"""dataflow: a CPU-GPU dataflow runtime.

Layering (each layer importable without the ones above it):

- ``dataflow.core``     program IR, validation, JSON, simulator converters
- ``dataflow.runtime``  generic execution engine over a DeviceBackend
- ``dataflow.tasks``    task-executable library (ops -> blocks)
- ``dataflow.training`` DNN lowering, planning integration, profiling, testing

Isolated pure-torch reference twins live OUTSIDE this package, in the
repo-root ``reference_models/`` (deliberately independent ground truth).
"""

__version__ = "0.0.1"
