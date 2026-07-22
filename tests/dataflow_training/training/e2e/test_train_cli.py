"""train.py CLI gates: the TOOL itself runs in the battery — the
subprocess layer (argparse -> conductor) that library tests cannot
cover (the uncovered-top-layer class). Zero-config world-1 train with
a v2 checkpoint, then peek reads it back.

Tests:
- test_train_cli_world1_writes_v2_checkpoint_and_peek_reads_it: the train CLI subprocess trains four steps, writes a format-2 checkpoint, and the peek subcommand reads it back.
"""
import json
import subprocess
import sys
from pathlib import Path
from dataflow_training.distributed.topology import repo_root

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)
pytest.importorskip("dataflow_sim")
pytest.importorskip("cuda.bindings")

REPO = repo_root()
TRAIN = REPO / "tools" / "train" / "train.py"
CORPUS = REPO / "datasets" / "fineweb10B"
needs_corpus = pytest.mark.skipif(
    not CORPUS.exists(), reason="needs the shard corpus")

pytestmark = [pytest.mark.fleet, pytest.mark.sim]


@pytest.mark.gpu
@pytest.mark.vram(gib=8)
@needs_corpus
def test_train_cli_world1_writes_v2_checkpoint_and_peek_reads_it(tmp_path):
    out = tmp_path / "cli_smoke.json"
    run = subprocess.run(
        [sys.executable, str(TRAIN), "train", "--preset", "gpt2",
         "--steps", "4", "--fast-budget", "4", "--backing-budget", "4",
         "--checkpoint-every", "2", "--out", str(out)],
        capture_output=True, text=True, timeout=600, cwd=REPO)
    assert run.returncode == 0, run.stderr[-2000:]
    curve = json.loads(out.read_text())
    assert len(curve["losses"]) == 4

    ck = REPO / "results" / "pretrain" / "checkpoints" / "cli_smoke"
    try:
        manifests = sorted(ck.glob("step_*/checkpoint_record.json"))
        assert manifests, "no v2 checkpoint written"
        m = json.loads(manifests[-1].read_text())
        assert m["format"] == 2
        assert m["launch"]["argv"][1:3] == [str(TRAIN), "train"] or \
            "train" in m["launch"]["argv"]

        peek = subprocess.run(
            [sys.executable, str(TRAIN), "peek", "cli_smoke"],
            capture_output=True, text=True, timeout=120, cwd=REPO)
        assert peek.returncode == 0, peek.stderr[-1000:]
        assert "steps recorded" in peek.stdout
    finally:
        import shutil

        shutil.rmtree(ck, ignore_errors=True)
        partial = REPO / "results" / "pretrain" / "cli_smoke_partial.json"
        partial.unlink(missing_ok=True)
