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
    opt = ReferenceAdamW(model.parameters(), recipe)

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
                          "tokens_per_step": tokens_per_step(cfg)})
    for step in range(steps):
        t0 = time.perf_counter()
        opt.zero_grad()
        step_loss = 0.0
        for r in range(R):
            tok, tgt = stream(step * R + r)
            tok = tok.to(device, non_blocking=True).view(B, T)
            tgt = tgt.to(device, non_blocking=True).view(B, T)
            if aux_coef > 0:
                loss_r = model.loss(tok, tgt, aux_coef=aux_coef)
                ce_r = (float(loss_r.detach())
                        - aux_coef * float(model.load_balance_loss().detach()))
            else:
                loss_r = model.loss(tok, tgt)
                ce_r = float(loss_r.detach())
            loss_r.backward()
            step_loss += ce_r
        opt.step(step)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        res.losses.append(step_loss / R)
        res.step_wall_s.append(dt)
        res.tok_per_s.append(tokens_per_step(cfg) / dt)
        if step % log_every == 0 or step == steps - 1:
            log(f"[reference] step {step:4d}/{steps}  loss {step_loss / R:.4f}"
                f"  lr {recipe.lr_at(step):.2e}  {tokens_per_step(cfg) / dt:.0f} tok/s")
    del model, opt
    torch.cuda.empty_cache()
    return res


# ================================ engine =====================================

@contextmanager
def daemon_client(slab_gib: float = 100.0, *, socket: str | None = None,
                  device: int = 0, attach: bool = False, log=print):
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
                                 device=device, fake=False))
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

    def put_round(step: int, r: int) -> None:
        tok, tgt = stream(step * R + r)
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
        out = client.run(prog_id, args={"step": step}, fetch=fetch)
        dt = time.perf_counter() - t0
        if out.get("state") != "done":
            raise RuntimeError(f"run step {step} state={out.get('state')}: {out}")
        step_loss = sum(out["fetched"][k] for k in fetch) / R
        res.losses.append(step_loss)
        res.step_wall_s.append(dt)
        res.tok_per_s.append(tokens_per_step(cfg) / dt)
        if step % log_every == 0 or step == steps - 1:
            log(f"[engine {budget_gib:g}GiB] step {step:4d}/{steps}  "
                f"loss {step_loss:.4f}  lr {recipe.lr_at(step):.2e}  "
                f"{tokens_per_step(cfg) / dt:.0f} tok/s")
    return res
