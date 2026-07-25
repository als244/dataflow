"""dataflow: a CPU-GPU dataflow runtime.

Layering (each layer importable without the ones above it):

- ``dataflow.core``     program IR, validation, JSON, simulator converters
- ``dataflow.runtime``  generic execution engine over a DeviceBackend

Isolated pure-torch reference twins live OUTSIDE this package, in the
repo-root ``reference_models/`` (deliberately independent ground truth).
"""

__version__ = "0.0.1"

import os as _os

# Expandable segments, unless the caller has already chosen a policy.
#
# Without it torch's caching allocator holds ~1.7 GiB of segment slack --
# reserved from the device but not allocated to any tensor -- which occupies
# the device exactly like real residency and pushes a run past a budget it is
# otherwise inside. Set at package import rather than in the daemon because it
# only takes effect before torch's first CUDA allocation in the process, and
# in-process users (tests, tools, anything driving Engine directly) reach that
# point without ever constructing a Server.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
