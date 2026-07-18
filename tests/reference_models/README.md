# tests/reference_models/

Home for suites whose SUBJECT is the reference twins themselves — the
isolated pure-torch models under the repo-root `reference_models/`
package (the truth tree).

Currently empty by design: every existing suite that touches the twins
uses them as a yardstick for something else, so it lives with its real
subject instead —

- `tests/dataflow_training/models/test_engine_vs_reference.py` judges
  the ENGINE against the twins (subject: the engine path);
- `tests/dataflow_training/pretrain/test_reference_muon.py` judges the
  reference muon optimizer in `dataflow_training.run.driver`, borrowing
  a twin as scaffolding (subject: the workload's yardstick optimizer).

A suite belongs here only when the twin is the thing under test — e.g.
golden cross-checks pinning a twin's forward/backward against
externally produced outputs, or twin-only API/behavior contracts that
hold with the engine absent.
