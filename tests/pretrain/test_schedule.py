"""Gates for the LR schedule: warmup->cosine shape, and BYTE-exact
agreement with the engine's LRSchedule (the parity authority)."""
from dataflow.pretrain.recipe import Recipe
from dataflow.pretrain.schedule import CosineSchedule
from dataflow.tasks.optim import LRSchedule


def test_warmup_then_cosine_shape():
    s = CosineSchedule(peak_lr=3e-4, min_lr=3e-5, warmup_steps=100,
                       total_steps=1000)
    # linear warmup increases to the peak at step 99 (0-indexed)
    assert s(0) < s(50) < s(99)
    assert abs(s(99) - 3e-4) < 1e-12
    # cosine decays after the peak, reaching min at the horizon
    assert s(99) > s(100) > s(500) > s(999)
    assert abs(s(999) - 3e-5) < 1e-9
    # step 0 is a nonzero fraction of peak (no wasted zero-LR step)
    assert abs(s(0) - 3e-4 / 100) < 1e-12


def test_matches_engine_lrschedule_exactly():
    """The reference derives lr as peak * LRSchedule.scale(step+1); the
    engine's AdamWStep applies hyper.lr * schedule.scale(run_args['step']+1).
    They must be identical to the last bit or the curves drift."""
    peak = 3e-4
    s = CosineSchedule(peak_lr=peak, min_lr=3e-5, warmup_steps=100,
                       total_steps=1000)
    eng = LRSchedule(kind="cosine", warmup_steps=100, total_steps=1000,
                     min_lr_frac=0.1)
    for step in range(1000):
        assert s(step) == peak * eng.scale(step + 1)


def test_recipe_hyper_spec_matches_base_hyper():
    r = Recipe()
    h = r.base_hyper()
    spec = r.hyper_spec()
    assert h.lr == spec["lr"] == r.peak_lr
    assert h.weight_decay == spec["weight_decay"] == r.weight_decay
    assert h.beta1 == spec["beta1"] and h.beta2 == spec["beta2"]
    # the schedule baked into base_hyper equals the spec's schedule
    assert h.schedule.kind == spec["schedule"]["kind"] == "cosine"
    assert h.schedule.warmup_steps == spec["schedule"]["warmup_steps"]
    assert h.schedule.total_steps == spec["schedule"]["total_steps"]
    assert abs(h.schedule.min_lr_frac - spec["schedule"]["min_lr_frac"]) < 1e-12


def test_lr_at_delegates_to_schedule():
    r = Recipe(peak_lr=2e-4, min_lr=2e-5, warmup_steps=10, total_steps=100)
    for step in (0, 5, 9, 10, 50, 99):
        assert r.lr_at(step) == r.schedule()(step)
