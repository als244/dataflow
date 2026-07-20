"""One training loop, two backends: the pytorch REFERENCE and the ENGINE
SERVICE (dataflowd). Both consume the same recipe, the same deterministic
token feed, and the same seeded init (the reference bridges the engine's
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

from ..model_families import bridges
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

def adamw_field_step(w, g, m, v, *, lr, beta1, beta2, eps, weight_decay, step):
    """In-place AdamW on one flat parameter, mirroring ops.adamw_step: bf16
    states, fp32 math, bias correction on the ROUND-TRIPPED states, weight
    decay applied to every element. Chunked to bound fp32 temporaries.
    THE exact-replica update both reference harnesses share (the twin
    trainer and the engine-vs-twin gradcheck)."""
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
        lr = r.lr(opt_step)          # = peak * schedule.scale(opt_step + 1)
        bc = opt_step + 1               # 1-indexed bias correction (engine parity)
        for p in self.params:
            if p.grad is None:
                continue
            m, v = self.state[id(p)]
            adamw_field_step(p.data, p.grad, m, v, lr=lr, beta1=r.beta1,
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
        lr = r.lr(opt_step)
        bc = opt_step + 1
        for par in self.adamw_params:
            if par.grad is None:
                continue
            m, v = self.adamw_state[id(par)]
            adamw_field_step(par.data, par.grad, m, v, lr=lr,
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


def save_reference_checkpoint(path, step: int, model, opt, res) -> None:
    """Atomic torch checkpoint for long reference runs: model + adamw
    moments (ordered like opt.params) + the curve so far. The data
    feed is a pure function of the round index and the twin has no
    dropout, so (states, step) is the complete resume record."""
    import os

    tmp = str(path) + ".tmp"
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "opt_mv": [(m, v) for (m, v) in
                   (opt.state[id(p)] for p in opt.params)],
        "losses": list(res.losses),
        "step_wall_s": list(res.step_wall_s),
        "tok_per_s": list(res.tok_per_s),
    }, tmp)
    os.replace(tmp, path)


def load_reference_checkpoint(path, model, opt, res) -> int:
    """Restore a save_reference_checkpoint record; returns the step to
    resume FROM (the first step not yet run)."""
    ck = torch.load(path, map_location="cuda", weights_only=False)
    model.load_state_dict(ck["model"])
    for p, (m, v) in zip(opt.params, ck["opt_mv"]):
        sm, sv = opt.state[id(p)]
        sm.copy_(m)
        sv.copy_(v)
    res.losses[:] = ck["losses"]
    res.step_wall_s[:] = ck["step_wall_s"]
    res.tok_per_s[:] = ck["tok_per_s"]
    return int(ck["step"])


def run_reference(cfg, recipe: Recipe, feed, steps: int, *, seed: int = 11,
                  device: str = "cuda", grad_checkpoint: bool = False,
                  checkpoint_every: int | None = None, checkpoint_dir=None,
                  resume: bool = False, partial_out=None,
                  log=print, log_every: int = 10) -> RunResult:
    """Train ``reference_models.Llama3`` from the engine's seeded init on ``feed``."""
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow_training.model_families.families import resolve_family

    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
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
    from dataflow_training.lowering.flops import flop_report

    flops = flop_report(cfg, fam.lower(cfg))
    f_eff, f_hw = flops.per_step()
    res = RunResult(backend="reference", budget_gib=None,
                    meta={"seed": seed, "grad_checkpoint": grad_checkpoint,
                          "aux_coef": aux_coef,
                          "opt_policy": getattr(cfg, "opt_policy", None)
                          or "adamw",
                          "tokens_per_step": tokens_per_step(cfg),
                          "flops_per_step": {"effective": f_eff,
                                             "hardware": f_hw}})
    ck_path = None
    if checkpoint_every and checkpoint_dir is not None:
        from pathlib import Path as _Path

        ck_path = _Path(checkpoint_dir) / "reference_ckpt.pt"
    start_step = 0
    if resume:
        if ck_path is None or not ck_path.exists():
            raise FileNotFoundError(
                f"--resume: no reference checkpoint at {ck_path}")
        start_step = load_reference_checkpoint(ck_path, model, opt, res)
        log(f"[reference] resumed from step {start_step} ({ck_path})")
    for step in range(start_step, steps):
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
        step_lens = {}
        for r in range(R):
            got = feed(step * R + r)
            rounds.append(got)
            step_valid += int((got[1] >= 0).sum())
            if len(got) > 2:
                step_lens[r] = got[2]
        for got in rounds:
            tok, tgt = got[0], got[1]
            # doc-aware feeds yield (tokens, targets, seq_lens): the
            # round runs PACKED (one (1, t) row, per-sequence positions,
            # block-diagonal attention) — the twins' varlen-native mode
            lens = got[2] if len(got) > 2 else None
            valid_r = int((tgt >= 0).sum())
            scale = valid_r / step_valid
            if lens is not None:
                tok = tok.to(device, non_blocking=True).view(1, -1)
                tgt = tgt.to(device, non_blocking=True).view(1, -1)
                loss_kw = {"seq_lens": lens}
            else:
                tok = tok.to(device, non_blocking=True).view(B, T)
                tgt = tgt.to(device, non_blocking=True).view(B, T)
                loss_kw = {}
            if aux_coef > 0:
                loss_r = model.loss(tok, tgt, aux_coef=aux_coef, **loss_kw)
                lbl_r = model.load_balance_loss()
                ce_r = loss_r - aux_coef * lbl_r
                (ce_r * scale + aux_coef * lbl_r).backward()
            else:
                ce_r = model.loss(tok, tgt, **loss_kw)
                (ce_r * scale).backward()
            step_loss += float(ce_r.detach()) * scale
        opt.step(step)
        # noaux balance-bias configs: apply the optimizer-time sign rule
        # on the STEP-AGGREGATE counts (the twin modules accumulated them
        # across this step's rounds; apply_bias_update clears them) —
        # matching the engine's once-per-step application
        speed = float(getattr(cfg, "bias_update_speed", 0.0) or 0.0)
        if speed > 0.0:
            for module in model.modules():
                if hasattr(module, "apply_bias_update"):
                    module.apply_bias_update(speed)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        res.losses.append(step_loss)
        res.step_wall_s.append(dt)
        res.tok_per_s.append(tokens_per_step(cfg) / dt)
        s_eff, s_hw = flops.per_step(
            step_lens or None, tokens=cfg.tokens, seq_len=cfg.seq_len)
        if step % log_every == 0 or step == steps - 1:
            log(f"[reference] step {step:4d}/{steps}  loss {step_loss:.4f}"
                f"  lr {recipe.lr(step):.2e}  {tokens_per_step(cfg) / dt:.0f} tok/s"
                f"  eff {s_eff / dt / 1e12:.1f} "
                f"hw {s_hw / dt / 1e12:.1f} TF/s")
        if ck_path is not None and (step + 1) % checkpoint_every == 0:
            save_reference_checkpoint(ck_path, step + 1, model, opt, res)
            if partial_out is not None:
                res.save(partial_out)
    del model, opt
    torch.cuda.empty_cache()
    return res


# ================================ engine =====================================

def init_model(client, family_name: str, cfg_dict: dict, *,
               seed: int = 0, object_sizes: dict | None = None,
               tp_view: dict | None = None, prog_id: str | None = None):
    """INIT IS A PROGRAM: build the family's one-task init program,
    register + run it through the ordinary verbs, and let the daemon's
    final-object capture persist every initial object (W_/O_/Aux_/data)
    into the store. Replaces the retired materialize_group verb; the
    bytes match in-process ``initial_values`` exactly (same code path).
    Returns the created object ids."""
    from dataflow.core.jsonio import program_to_dict
    from dataflow_training.model_families.families import (
        build_init_program,
        family,
    )
    from dataflow_training.register import canonical_spec

    fam = family(family_name)
    cfg = fam.config_type(**cfg_dict)
    program = build_init_program(fam, cfg, seed=seed,
                                 object_sizes=object_sizes,
                                 tp_view=tp_view)
    pid = prog_id or f"init-{family_name}-{seed}"
    reg = client.register_program(program_to_dict(program),
                                  resolver=canonical_spec(family_name,
                                                          cfg_dict),
                                  name=pid)
    missing = reg["bindings"]["missing_inputs"]
    if missing:
        raise RuntimeError(f"init program has unbound inputs: {missing}")
    out = client.run(reg["prog_id"], args={})
    if out.get("state") != "done":
        raise RuntimeError(f"init run failed: {out}")
    client.unregister_program(reg["prog_id"])
    return [o.id for t in program.tasks for o in t.outputs]


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

    from dataflow_training.register import register_all

    register_all()          # in-process server shares this registry
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


def measured_variant(fam, cfg, profiles, resolver, pcie, levels):
    """One re-lowered recompute variant with profiled task costs AND
    the executing box's measured PCIe bandwidths installed — the same
    treatment the base program gets, so the recompute search prices
    transfers honestly."""
    from dataclasses import replace as dc_replace

    from dataflow_training.run.profiling import apply_measured_costs

    prog = apply_measured_costs(fam.lower(cfg, recompute_levels=levels),
                                profiles, resolver)
    return dc_replace(prog, bandwidth_from_slow=pcie.bidi_h2d,
                      bandwidth_to_slow=pcie.bidi_d2h)


def plan_at_budget(cfg, budget_gib: float, *, recompute: bool = True,
                   measured: bool = False):
    """Plan the single-step program at a device budget (GiB). Returns the
    PlannedProgram; its placement is baked in -> a budget-specific prog_id.
    With recompute, the planner re-lowers the program at candidate recompute
    levels (``build_variant``). ``measured`` swaps the roofline cost seeds
    for PROFILED task costs (load_or_profile, disk-cached) AND the
    lowering's default transfer bandwidths for the box's measured PCIe
    numbers (cached_pcie) — the plan's makespan_us then IS the
    true-profiling simulator prediction. BIDI rates by doctrine
    (conservative): each lane is priced at its concurrent-saturation
    bandwidth, so predictions are floors and reality comes in at or
    better than expected (chain-ordered plans alternate directions
    enough that lanes often achieve uni rates — up to ~20% better
    than the bidi-priced prediction at the tightest budgets)."""
    import functools
    from dataclasses import replace as dc_replace

    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.lowering.planning import plan_program

    fam = resolve_family(cfg)
    if not measured:
        variant = (lambda levels: fam.lower(cfg, recompute_levels=levels)) if recompute else None
        return plan_program(fam.lower(cfg),
                            fast_memory_capacity=int(budget_gib * 1024 ** 3),
                            recompute=recompute, build_variant=variant)
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow_training.run.profiling import (apply_measured_costs,
                                                 cached_pcie,
                                                 load_or_profile)

    backend = CudaBackend()
    dims = fam.derive_dims(cfg)
    resolver = fam.build_resolver(dims)
    profiles = load_or_profile(fam.lower(cfg), resolver, backend)
    pcie = cached_pcie(backend)
    variant = (functools.partial(measured_variant, fam, cfg, profiles,
                                 resolver, pcie)
               if recompute else None)
    base = dc_replace(apply_measured_costs(fam.lower(cfg), profiles, resolver),
                      bandwidth_from_slow=pcie.bidi_h2d,
                      bandwidth_to_slow=pcie.bidi_d2h)
    return plan_program(base,
                        fast_memory_capacity=int(budget_gib * 1024 ** 3),
                        recompute=recompute, build_variant=variant)


def latest_engine_checkpoint(ckpt_dir) -> Path | None:
    """Newest COMPLETE solo checkpoint under ``ckpt_dir`` (the snapshot
    writer lands manifest.json last, so its presence is the
    completeness marker), or None."""
    if ckpt_dir is None:
        return None
    found = sorted(Path(ckpt_dir).glob("step_*/manifest.json"))
    return found[-1].parent if found else None


def run_engine(client, cfg, recipe: Recipe, feed, steps: int, *,
               budget_gib: float, seed: int = 11, recompute: bool = True,
               measured: bool = False,
               profile: dict | None = None,
               log=print, log_every: int = 10,
               checkpoint_every: int | None = None,
               checkpoint_dir=None, keep_last: int = 3,
               resume: bool = False) -> RunResult:
    """Train ``cfg`` through the daemon at ``budget_gib`` device budget.

    ``checkpoint_every``: snapshot W_*/O_* + the loss curve every N
    steps (host-local under ``checkpoint_dir``; keep-last pruning).
    ``resume``: restore the newest complete checkpoint AFTER init and
    registration, then continue — the engine is stateless per step,
    so bytes + step index + recorded losses are the whole trajectory
    (data rounds re-derive from the feed by step index)."""
    import shutil

    from dataflow.core.jsonio import program_to_dict

    planned = plan_at_budget(cfg, budget_gib, recompute=recompute,
                             measured=measured)
    n_rc = sum(1 for v in (planned.recompute_levels or {}).values() if v)
    ts = planned.transfer_stats or {}
    h2d = ts.get("from_slow", {})
    d2h = ts.get("to_slow", {})
    log(f"[plan] predicted {planned.makespan_us / 1e6:.2f} s/step "
        f"({'measured' if measured else 'roofline'} costs)  peak fast "
        f"{planned.peak_fast_bytes / 1024**3:.2f} GiB  peak backing "
        f"{planned.peak_backing_bytes / 1024**3:.2f} GiB  recompute "
        f"{n_rc}/{len(planned.recompute_levels or {})}  "
        f"h2d {h2d.get('bytes', 0) / 1e9:.1f} GB "
        f"({h2d.get('busy_us', 0.0) / planned.makespan_us * 100:.0f}%)  "
        f"d2h {d2h.get('bytes', 0) / 1e9:.1f} GB "
        f"({d2h.get('busy_us', 0.0) / planned.makespan_us * 100:.0f}%)")
    prog_dict = program_to_dict(planned.program)
    cd = cfg_dict(cfg)
    fam = resolver_family(cfg)
    resolver = {"kind": "model_family", "family": fam, "cfg": cd,
                "hyper": recipe.hyper_spec()}

    init_model(client, fam, cd, seed=seed)
    R = cfg.grad_accum_rounds
    valid_by_step: dict[int, int] = {}
    lens_by_step: dict[int, dict] = {}

    def put_round(step: int, r: int) -> None:
        got = feed(step * R + r)
        tok, tgt = got[0], got[1]
        if len(got) > 2:
            # doc-aware round: per-round cumulative boundaries ride
            # run_args ("seq_lens" wire form, round-keyed) — the
            # engine's prologue materializes Segments from them
            bounds = [0]
            for n in got[2]:
                bounds.append(bounds[-1] + int(n))
            lens_by_step.setdefault(step, {})[str(r)] = bounds
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

    from dataflow_training.lowering.flops import flop_report

    flops = flop_report(cfg, planned.program)
    f_eff, f_hw = flops.per_step()
    res = RunResult(backend="engine", budget_gib=budget_gib,
                    meta={"seed": seed, "prog_id": prog_id,
                          "peak_fast_bytes": planned.peak_fast_bytes,
                          "recompute": recompute,
                          "tokens_per_step": tokens_per_step(cfg),
                          "flops_per_step": {"effective": f_eff,
                                             "hardware": f_hw}})
    persist = sorted(s.id for s in planned.program.initial_objects
                     if s.id.startswith(("W_", "O_")))
    start_step = 0
    if resume:
        ck = latest_engine_checkpoint(checkpoint_dir)
        if ck is None:
            raise RuntimeError(f"resume: no complete checkpoint under "
                               f"{checkpoint_dir}")
        got = client.restore_snapshot(str(ck), overwrite=True)
        meta = got["client_meta"]
        if int(meta["seed"]) != seed:
            raise RuntimeError(f"resume: checkpoint seed {meta['seed']} "
                               f"!= run seed {seed}")
        start_step = int(meta["step"])
        res.losses = [float(x) for x in meta["losses"]]
        res.meta["resumed_from"] = str(ck)
        log(f"[engine] resumed @ step {start_step} from {ck}")
    fetch = [f"loss_0_{r}" for r in range(R)]
    prof_start = profile.get("start") if profile else None
    prof_stop = profile.get("stop") if profile else None
    for step in range(start_step, steps):
        if prof_start is not None and step == prof_start:
            client.profiler_control("start")
            log(f"[engine] profiler capture STARTED before step {step}")
        if step > 0:
            for r in range(R):
                put_round(step, r)
        t0 = time.perf_counter()
        # GLOBAL-DENOMINATOR convention: one denominator for every round
        # of the step (scalar valid_rows); per-round loss objects then
        # hold Sum(nll_r)/valid_step, so the STEP loss is their plain sum
        run_args = {"step": step, "valid_rows": valid_by_step.pop(step)}
        lens = lens_by_step.pop(step, None)
        if lens is not None:
            run_args["seq_lens"] = lens
        out = client.run(prog_id, args=run_args, fetch=fetch)
        dt = time.perf_counter() - t0
        if out.get("state") != "done":
            raise RuntimeError(f"run step {step} state={out.get('state')}: {out}")
        step_loss = sum(out["fetched"][k] for k in fetch)
        res.losses.append(step_loss)
        res.step_wall_s.append(dt)
        res.tok_per_s.append(tokens_per_step(cfg) / dt)
        if step % log_every == 0 or step == steps - 1:
            lens_lists = None
            if lens is not None:
                # wire form is cumulative bounds; the flop scaler wants lengths
                lens_lists = {r: [b[i + 1] - b[i] for i in range(len(b) - 1)]
                              for r, b in lens.items()}
            s_eff, s_hw = flops.per_step(
                lens_lists, tokens=cfg.tokens, seq_len=cfg.seq_len)
            log(f"[engine {budget_gib:g}GiB] step {step:4d}/{steps}  "
                f"loss {step_loss:.4f}  lr {recipe.lr(step):.2e}  "
                f"{tokens_per_step(cfg) / dt:.0f} tok/s  "
                f"eff {s_eff / dt / 1e12:.1f} "
                f"hw {s_hw / dt / 1e12:.1f} TF/s")
        if prof_stop is not None and step == prof_stop:
            client.profiler_control("stop")
            log(f"[engine] profiler capture STOPPED after step {step}")
        step_next = step + 1
        if checkpoint_every and step_next % checkpoint_every == 0 \
                and checkpoint_dir is not None:
            dest = Path(checkpoint_dir) / f"step_{step_next:06d}"
            out = client.snapshot("all", str(dest), ids=persist,
                                  client_meta={"step": step_next,
                                               "seed": seed,
                                               "losses": res.losses})
            done = client.wait_snapshot(out["snap_id"], timeout=600.0)
            if done["state"] != "done":
                raise RuntimeError(f"checkpoint @ {step_next} failed: "
                                   f"{done}")
            log(f"[engine] checkpoint @ step {step_next} -> {dest}")
            if keep_last > 0:
                complete = sorted(
                    Path(checkpoint_dir).glob("step_*/manifest.json"))
                for mf in complete[:-keep_last]:
                    shutil.rmtree(mf.parent, ignore_errors=True)
    return res
