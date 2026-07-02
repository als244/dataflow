"""dataflow: a CPU-GPU dataflow runtime.

Layering (each layer importable without the ones above it):

- ``dataflow.core``     program IR, validation, JSON, simulator converters
- ``dataflow.runtime``  generic execution engine over a DeviceBackend
- ``dataflow.tasks``    task-executable library (ops -> blocks)
- ``dataflow.training`` DNN lowering, planning integration, profiling, testing
- ``dataflow.models``   model definitions + golden references
"""

__version__ = "0.0.1"
