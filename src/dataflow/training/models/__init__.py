"""Per-family training modules: one per model family — the Shaped
config + presets, dims, kind specs, lowering, and initial values.
Families register in ../families.py; the shared machinery
(shaped_program, warmup_program, lowering, planning, profiling,
train_loop) lives one level up."""
