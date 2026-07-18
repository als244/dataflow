"""Model families: everything a family contributes lives here — the
Shaped config + dims + lowering entry (llama3.py, dsv3.py, ...), the
block executables composing the shared templates (llama3_blocks.py,
...; templates in ../blocks/base_blocks.py), the parity bridges into
the isolated reference twins (bridges/), and the family registry
(families.py). The per-family X/ package consolidation and the
ModelFamily/Model objects land with the model_families milestone of
the split; this flat staging keeps module names stable meanwhile.
"""
