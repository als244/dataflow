"""One training loop, two backends: the pytorch REFERENCE and the ENGINE
SERVICE (dataflowd). Both consume the same recipe, the same deterministic
token stream, and the same seeded init (the reference bridges the engine's
packed init bytes), so the only variable is the execution engine.

The reference is a plain pytorch loop over ``reference_models.Llama3`` with an
AdamW that mirrors the engine's ``ops.adamw_step`` (bf16 states, fp32 math,
weight decay on every field, cosine LR). The engine backend drives the
daemon: plan at a device budget → register (with the recipe hyper on the
resolver) → seed W/O in the store → per step put the round data + ``run`` +
read the per-round losses. Weights persist and evolve in the store across
``run`` calls; the driver owns the step loop.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

_ROOT = str(Path(__file__).resolve().parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from . import bridges
from .presets import cfg_dict, resolver_family, tokens_per_step
from .recipe import Recipe


@dataclass
class RunResult:
    backend: str                     # "reference" | "engine"
    losses: list[float] = field(default_factory=list)     # per-step mean CE
    tok_per_s: list[float] = field(default_factory=list)  # per-step, run only
    step_wall_s: list[float] = field(default_factory=list)
    budget_gib: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def steady_tok_per_s(self) -> float:
        tail = self.tok_per_s[1:] or self.tok_per_s
        return sum(tail) / len(tail) if tail else 0.0

    def save(self, path: str | os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def load_result(path: str | os.PathLike) -> RunResult:
    d = json.loads(Path(path).read_text())
    return RunResult(**d)


# =============================== reference ===================================

def _adamw_inplace(w, g, m, v, *, lr, beta1, beta2, eps, weight_decay, step):
    """In-place AdamW on one flat parameter, mirroring ops.adamw_step: bf16
    states, fp32 math, bias correction on the ROUND-TRIPPED states, weight
    decay applied to every element. Chunked to bound fp32 temporaries."""
    CHUNK = 1 << 24
    wf, gf_, mf_, vf_ = w.view(-1), g.view(-1), m.view(-1), v.view(-1)
    n = wf.numel()
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        wc, gc, mc, vc = wf[lo:hi], gf_[lo:hi], mf_[lo:hi], vf_[lo:hi]
        gf = gc.float()
        mfl = mc.float().mul_(beta1).add_(gf, alpha=1 - beta1)
        vfl = vc.float().mul_(beta2).addcmul_(gf, gf, value=1 - beta2)
        mc.copy_(mfl.to(mc.dtype))
        vc.copy_(vfl.to(vc.dtype))
        mhat = mc.float() / (1 - beta1 ** step)
        vhat = vc.float() / (1 - beta2 ** step)
        wff = wc.float()
        wff -= lr * (mhat / (vhat.sqrt() + eps) + weight_decay * wff)
        wc.copy_(wff.to(wc.dtype))


class ReferenceAdamW:
    """AdamW over an nn.Module's parameters, numerically matching the engine
    (bf16 m/v, per-field, weight decay everywhere, LR from the cosine
    schedule; bias-correction step is 1-indexed like the engine)."""

    def __init__(self, params, recipe: Recipe):
        self.params = [p for p in params if p.requires_grad]
        self.recipe = recipe
        self.state = {id(p): (torch.zeros_like(p), torch.zeros_like(p))
                      for p in self.params}

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self, opt_step: int) -> None:
        r = self.recipe
        lr = r.lr_at(opt_step)          # = peak * schedule.scale(opt_step + 1)
        bc = opt_step + 1               # 1-indexed bias correction (engine parity)
        for p in self.params:
            if p.grad is None:
                continue
            m, v = self.state[id(p)]
            _adamw_inplace(p.data, p.grad, m, v, lr=lr, beta1=r.beta1,
                           beta2=r.beta2, eps=r.eps,
                           weight_decay=r.weight_decay, step=bc)


NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315   # quintic Newton-Schulz
NS_ITERS = 5

# the engine recipe's adamw carve-outs (MuonRecipePolicy in
# tasks/optim.py) restated over reference parameter names — a test
# asserts the two stay identical
MUON_ADAMW_FRAGMENTS = ("norm", "embed", "head", "router", "idx")


def ns_orthogonalize(x: "torch.Tensor", *, eps: float = 1e-8,
                     iters: int = NS_ITERS) -> "torch.Tensor":
    """(..., r, c) fp32 -> per-matrix approximate UV^T. The reference
    twin of kernels/muon.py: transpose-to-wide, Frobenius normalize
    with eps inside the add, quintic iteration."""
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = x / x.norm(dim=(-2, -1), keepdim=True).add_(eps)
    for _ in range(iters):
        a = x @ x.mT
        b = NS_B * a + NS_C * (a @ a)
        x = NS_A * x + b @ x
    return x.mT if transposed else x


class ReferenceMuon:
    """The muon deployment split over a reference nn.Module, mirroring
    the engine end to end: matrix params (rank 2, and rank-3 expert
    stacks) run nesterov momentum + quintic Newton-Schulz with the
    Moonshot 0.2*sqrt(max(r, c)) scale and decoupled weight decay —
    momentum arithmetic in the (bf16) momentum buffer's dtype, NS in
    fp32 (kernels/muon.py semantics); embed/head tables, norms, and
    every other param take the exact ReferenceAdamW step. ``muon_lr``
    None means muon shares the scheduled adamw lr (the Moonshot scale
    is designed for that); when set it rides the same schedule via
    the peak-lr ratio."""

    def __init__(self, model, recipe: Recipe):
        self.recipe = recipe
        self.muon_lr = recipe.muon_lr
        self.momentum = recipe.momentum
        self.adamw_params: list = []
        self.muon_params: list = []
        for name, par in model.named_parameters():
            if not par.requires_grad:
                continue
            low = name.lower()
            if par.ndim not in (2, 3) \
                    or any(fr in low for fr in MUON_ADAMW_FRAGMENTS):
                self.adamw_params.append(par)
            else:
                if min(par.shape[-2:]) <= 1:
                    raise ValueError(
                        f"{name}: degenerate matrix {tuple(par.shape)} "
                        f"classified muon — name it into an adamw "
                        f"carve-out instead")
                self.muon_params.append(par)
        if not self.muon_params:
            raise ValueError("muon recipe found no matrix params — "
                             "wrong model or naming?")
        self.adamw_state = {id(par): (torch.zeros_like(par),
                                      torch.zeros_like(par))
                            for par in self.adamw_params}
        self.muon_state = {id(par): torch.zeros_like(par)
                           for par in self.muon_params}

    def zero_grad(self) -> None:
        for par in self.adamw_params + self.muon_params:
            par.grad = None

    def step(self, opt_step: int) -> None:
        r = self.recipe
        lr = r.lr_at(opt_step)
        bc = opt_step + 1
        for par in self.adamw_params:
            if par.grad is None:
                continue
            m, v = self.adamw_state[id(par)]
            _adamw_inplace(par.data, par.grad, m, v, lr=lr,
                           beta1=r.beta1, beta2=r.beta2, eps=r.eps,
                           weight_decay=r.weight_decay, step=bc)
        mlr = lr if self.muon_lr is None \
            else self.muon_lr * (lr / r.peak_lr)
        for par in self.muon_params:
            if par.grad is None:
                continue
            m = self.muon_state[id(par)]
            gm = par.grad.to(m.dtype)
            m.mul_(self.momentum).add_(gm)
            eff = gm.add(m, alpha=self.momentum)      # nesterov
            if eff.ndim == 2:
                eff = eff.unsqueeze(0)
            o = ns_orthogonalize(eff.float(), eps=r.eps)
            if r.weight_decay:
                w32 = par.data.float().mul_(1.0 - mlr * r.weight_decay)
                par.data.copy_(w32.to(par.dtype))
            scale = 0.2 * max(par.shape[-2:]) ** 0.5
            par.data.add_(o.reshape(par.shape).to(par.dtype),
                          alpha=-mlr * scale)


def reference_optimizer(model, cfg, recipe: Recipe):
    """The reference optimizer for the config's opt_policy: "muon"
    means the hybrid recipe (ReferenceMuon), anything unset/adamw the
    plain ReferenceAdamW. Other engine policies have no reference
    twin yet — refuse loudly."""
    policy = getattr(cfg, "opt_policy", None) or "adamw"
    if policy == "muon":
        return ReferenceMuon(model, recipe)
    if policy == "adamw":
        return ReferenceAdamW(model.parameters(), recipe)
    raise ValueError(f"opt_policy {policy!r} has no reference "
                     f"optimizer (have: adamw, muon)")


def run_reference(cfg, recipe: Recipe, stream, steps: int, *, seed: int = 11,
                  device: str = "cuda", grad_checkpoint: bool = False,
                  log=print, log_every: int = 10) -> RunResult:
    """Train ``reference_models.Llama3`` from the engine's seeded init on ``stream``."""
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow.training.families import resolve_family

    fam = resolve_family(cfg)
    dims = fam.dims_of(cfg)
    backend = CudaBackend()
    values = fam.initial_values(fam.lower(cfg), cfg, backend, seed=seed)
    model = bridges.build_reference_model(cfg, device=device)
    bridges.load_reference_init(model, cfg, dims,
                                bridges.get_bytes_from_values(values))
    for buf in values.values():        # only W_* were needed to bridge
        backend.free(buf)
    model.grad_checkpoint = grad_checkpoint
    model.train()
    opt = reference_optimizer(model, cfg, recipe)

    B, T = dims.tokens // dims.seq_len, dims.seq_len
    R = cfg.grad_accum_rounds
    # LBL-ON configs (MoE aux_coef > 0): train the composite CE + alpha*LBL
    # per round (the reference's per-round semantics match the engine's
    # per-round injection) but LOG THE CE CHANNEL — the pinned scalar
    # convention (the engine's loss_* objects are always pure CE).
    aux_coef = float(getattr(cfg, "aux_coef", 0.0) or 0.0)
    res = RunResult(backend="reference", budget_gib=None,
                    meta={"seed": seed, "grad_checkpoint": grad_checkpoint,
                          "aux_coef": aux_coef,
                          "opt_policy": getattr(cfg, "opt_policy", None)
                          or "adamw",
                          "tokens_per_step": tokens_per_step(cfg)})
    for step in range(steps):
        t0 = time.perf_counter()
        opt.zero_grad()
        step_loss = 0.0
        # GLOBAL-DENOMINATOR convention: every round's CE normalizes by
        # the STEP's valid-token total (rounds are a memory optimization,
        # not a semantics knob) — the engine side receives the same
        # denominator via run_args valid_rows. The per-round LBL term is
        # per-round BY DESIGN and is not rescaled.
        rounds = []
        step_valid = 0
        for r in range(R):
            tok, tgt = stream(step * R + r)
            rounds.append((tok, tgt))
            step_valid += int((tgt >= 0).sum())
        for tok, tgt in rounds:
            valid_r = int((tgt >= 0).sum())
            scale = valid_r / step_valid
            tok = tok.to(device, non_blocking=True).view(B, T)
            tgt = tgt.to(device, non_blocking=True).view(B, T)
            if aux_coef > 0:
                loss_r = model.loss(tok, tgt, aux_coef=aux_coef)
                lbl_r = model.load_balance_loss()
                ce_r = loss_r - aux_coef * lbl_r
                (ce_r * scale + aux_coef * lbl_r).backward()
            else:
                ce_r = model.loss(tok, tgt)
                (ce_r * scale).backward()
            step_loss += float(ce_r.detach()) * scale
        opt.step(step)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        res.losses.append(step_loss)
        res.step_wall_s.append(dt)
        res.tok_per_s.append(tokens_per_step(cfg) / dt)
        if step % log_every == 0 or step == steps - 1:
            log(f"[reference] step {step:4d}/{steps}  loss {step_loss:.4f}"
                f"  lr {recipe.lr_at(step):.2e}  {tokens_per_step(cfg) / dt:.0f} tok/s")
    del model, opt
    torch.cuda.empty_cache()
    return res


# ================================ engine =====================================

@contextmanager
def daemon_client(slab_gib: float = 100.0, *, socket: str | None = None,
                  device: int = 0, attach: bool = False, log=print,
                  peer_name: str | None = None,
                  peer_listen: str | None = None):
    """Yield a connected ``EngineClient``. By default boots an in-process
    dataflowd (the full service stack — wire protocol, store, dispatcher —
    hosted in a thread) with an EXPLICIT slab (never 'auto'); set
    ``attach=True`` to connect to an already-running daemon at ``socket``."""
    from dataflow.service import EngineClient, EngineConfig, Server

    if attach:
        if socket is None:
            raise ValueError("attach=True requires an explicit socket path")
        c = EngineClient(socket, client_name="pretrain")
        try:
            yield c
        finally:
            c.close()
        return

    sock = socket or f"/tmp/pretrain-dataflowd-{os.getpid()}.sock"
    log(f"[daemon] booting in-process dataflowd slab={slab_gib} GiB dev={device}")
    t0 = time.perf_counter()
    server = Server(EngineConfig(socket_path=sock, slab_backing_gib=slab_gib,
                                 device=device, fake=False,
                                 peer_name=peer_name,
                                 peer_listen=peer_listen))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ok = False
    for _ in range(6000):              # up to ~60s for the pinned slab to arm
        try:
            EngineClient(sock, client_name="probe").close()
            ok = True
            break
        except (ConnectionError, FileNotFoundError, OSError):
            time.sleep(0.01)
    if not ok:
        raise RuntimeError("dataflowd did not accept connections")
    log(f"[daemon] up in {time.perf_counter() - t0:.1f}s ({sock})")
    c = EngineClient(sock, client_name="pretrain")
    try:
        yield c
    finally:
        c.close()
        try:
            server.state.shutdown_requested.set()
            server.dispatcher.stop()
            if getattr(server.store, "slab", None) is not None:
                server.store.slab.free()
        except Exception as e:            # best-effort teardown
            log(f"[daemon] teardown warning: {e}")


def plan_at_budget(cfg, budget_gib: float, *, recompute: bool = True):
    """Plan the single-step program at a device budget (GiB). Returns the
    PlannedProgram; its placement is baked in -> a budget-specific prog_id.
    With recompute, the planner re-lowers the program at candidate recompute
    levels (``build_variant``) to fit tight budgets (roofline costs)."""
    from dataflow.training.families import resolve_family
    from dataflow.training.planning import plan_program

    fam = resolve_family(cfg)
    variant = (lambda levels: fam.lower(cfg, recompute_levels=levels)) if recompute else None
    return plan_program(fam.lower(cfg),
                        fast_memory_capacity=int(budget_gib * 1024 ** 3),
                        recompute=recompute, build_variant=variant)


def run_engine(client, cfg, recipe: Recipe, stream, steps: int, *,
               budget_gib: float, seed: int = 11, recompute: bool = True,
               log=print, log_every: int = 10) -> RunResult:
    """Train ``cfg`` through the daemon at ``budget_gib`` device budget."""
    from dataflow.core.jsonio import program_to_dict

    planned = plan_at_budget(cfg, budget_gib, recompute=recompute)
    prog_dict = program_to_dict(planned.program)
    cd = cfg_dict(cfg)
    fam = resolver_family(cfg)
    resolver = {"family": fam, "cfg": cd, "hyper": recipe.hyper_spec()}

    client.materialize_group({"kind": "family_init_all", "family": fam,
                              "cfg": cd, "seed": seed})
    R = cfg.grad_accum_rounds
    valid_by_step: dict[int, int] = {}

    def put_round(step: int, r: int) -> None:
        tok, tgt = stream(step * R + r)
        valid_by_step[step] = valid_by_step.get(step, 0) + int((tgt >= 0).sum())
        client.put_object(f"tokens_0_{r}", tok.numpy().tobytes())
        client.put_object(f"targets_0_{r}", tgt.numpy().tobytes())

    for r in range(R):                 # inputs must exist for register binding
        put_round(0, r)
    reg = client.register_program(prog_dict, resolver=resolver)
    missing = reg["bindings"]["missing_inputs"]
    if missing:
        raise RuntimeError(f"unbound inputs: {missing}")
    prog_id = reg["prog_id"]

    res = RunResult(backend="engine", budget_gib=budget_gib,
                    meta={"seed": seed, "prog_id": prog_id,
                          "peak_fast_bytes": planned.peak_fast_bytes,
                          "recompute": recompute,
                          "tokens_per_step": tokens_per_step(cfg)})
    fetch = [f"loss_0_{r}" for r in range(R)]
    for step in range(steps):
        if step > 0:
            for r in range(R):
                put_round(step, r)
        t0 = time.perf_counter()
        # GLOBAL-DENOMINATOR convention: one denominator for every round
        # of the step (scalar valid_rows); per-round loss objects then
        # hold Sum(nll_r)/valid_step, so the STEP loss is their plain sum
        out = client.run(prog_id,
                         args={"step": step,
                               "valid_rows": valid_by_step.pop(step)},
                         fetch=fetch)
        dt = time.perf_counter() - t0
        if out.get("state") != "done":
            raise RuntimeError(f"run step {step} state={out.get('state')}: {out}")
        step_loss = sum(out["fetched"][k] for k in fetch)
        res.losses.append(step_loss)
        res.step_wall_s.append(dt)
        res.tok_per_s.append(tokens_per_step(cfg) / dt)
        if step % log_every == 0 or step == steps - 1:
            log(f"[engine {budget_gib:g}GiB] step {step:4d}/{steps}  "
                f"loss {step_loss:.4f}  lr {recipe.lr_at(step):.2e}  "
                f"{tokens_per_step(cfg) / dt:.0f} tok/s")
    return res
