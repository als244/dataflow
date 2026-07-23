"""Client-path parity reads.

Fetch engine-produced objects (gradients, block outputs, losses, MoE counts)
as HOST copies through the daemon client and map them into twin-named
comparison tensors — the same math the in-process ``gradcheck`` reads perform,
but sourced via ``client.get_object`` against objects retained on the backing
(host) tier. Because the backing tier IS host memory, the fetch is a plain host
copy: no engine device view is ever constructed or held, so this whole read
half is memory-safe by construction (the workload-test client contract).

Retention rule: an object is only capturable by ``get_object`` if the program
pins it to ``"backing"`` in ``final_locations`` (the engine copies the backing
slot into the store at run end). The in-process reads use ``"fast"`` and view
the device slot directly — that path is exactly what this module replaces.
"""
import dataclasses as dc
import functools

import torch

from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME
from dataflow_training.model_families import bridges


def with_backing_retention(program, ids):
    """Return a copy of ``program`` whose ``final_locations`` pins every id in
    ``ids`` to the backing (host) tier, so the run retains it and the engine
    captures it into the store for ``client.get_object`` to return."""
    return dc.replace(
        program,
        final_locations={**dict(program.final_locations),
                         **{oid: "backing" for oid in ids}})


def fetch_host_tensor(client, oid, dtype):
    """``client.get_object(oid)`` -> a 1-D host tensor of ``dtype`` over the
    returned bytes. The object must have been retained on the backing tier."""
    raw = client.get_object(oid)
    return torch.frombuffer(bytearray(raw), dtype=dtype)


def grad_shim(object_id, fabricated, weight_sizes):
    """Bridge name-map shim: hand ``to_reference_state_dict`` a weight-layout
    buffer whose field slots already hold the gradient values. A weight with no
    optimizer task (fully frozen) has no gradient storage, so fabricate a
    NaN-poisoned blank of the right size — every field derived from it reads
    NaN and is stripped by the caller."""
    made = fabricated.get(object_id)
    if made is not None:
        return made
    n = weight_sizes[object_id]
    blank = torch.zeros(n, dtype=torch.uint8, device="cuda")
    blank[: n - n % 2].view(torch.bfloat16).fill_(float("nan"))
    fabricated[object_id] = blank
    return blank


def client_grad_state_dict(client, cfg, program, resolver):
    """Engine dW backing -> twin-named gradient dict, sourced via the client.

    Mirror of ``testing.gradcheck.engine_grad_state_dict``: each optimizer
    task's executable resolves its own (weight_layout, grad_layout); the dW
    backing is read as HOST bytes with ``client.get_object`` (instead of an
    in-process device view), its gradient fields are packed into a fabricated
    weight-layout buffer, and the family bridge's ``to_reference_state_dict``
    supplies the twin name map. Requires the run to have retained every dW on
    the backing tier (see ``with_backing_retention``).
    """
    weight_sizes = program.object_sizes()
    fabricated: dict[str, torch.Tensor] = {}
    for task in program.tasks:
        if not task.id.startswith("optimizer_"):
            continue
        w_id = next(i for i in task.inputs if i.startswith("W_"))
        dw_id = next(i for i in task.inputs if i.startswith("dW"))
        executable = resolver(task)
        weight_layout, grad_layout = executable._layouts(
            task, weight_sizes[w_id])[:2]
        raw = fetch_host_tensor(client, dw_id, torch.uint8).cuda()
        grads = grad_layout.unpack_tensor(raw)
        grad_dtypes = {f.name: f.dtype for f in grad_layout.fields}
        fake = torch.empty(weight_layout.total_bytes, dtype=torch.uint8,
                           device="cuda")
        views = weight_layout.unpack_tensor(fake)
        for field in weight_layout.fields:
            if field.name in grads:
                if grad_dtypes[field.name] != field.dtype:
                    raise AssertionError(
                        f"{dw_id}:{field.name} grad dtype "
                        f"{grad_dtypes[field.name]} != weight dtype "
                        f"{field.dtype} — the fabricated-buffer shim would "
                        f"quantize")
                views[field.name].copy_(grads[field.name])
            else:
                views[field.name].fill_(
                    float("nan")
                    if TORCH_DTYPE_BY_NAME[field.dtype].is_floating_point
                    else 0)
        fabricated[w_id] = fake

    shim = functools.partial(grad_shim, fabricated=fabricated,
                             weight_sizes=weight_sizes)
    out = {}
    for name, tensor in bridges.to_reference_state_dict(cfg, shim).items():
        if tensor.float().isnan().all():
            continue                       # gradient-free field (frozen)
        out[name] = tensor
    return out
