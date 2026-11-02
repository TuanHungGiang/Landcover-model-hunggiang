"""CLI, non-destructive eval-only, and automatic holdout structural regressions."""
import ast
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / "scripts/train_loveda_ddp_v3.py"
SRC = FILE.read_text()


def test_post_train_eval_after_all_epochs():
    assert SRC.index("for epoch in range(start, a.epochs):") < SRC.index(
        "run_final_evaluation(") < SRC.index("finally:")
    assert "start = a.epochs  # Skip training entirely" in SRC
    assert "Path(a.eval_only) if a.eval_only else output / \"best.pt\"" in SRC


def test_default_holdout_and_safe_eval_only_args():
    ast.parse(SRC)
    assert 'dest="evaluate_holdout", action="store_true", default=True' in SRC
    assert 'dest="evaluate_holdout", action="store_false"' in SRC
    assert 'p.add_argument("--profile-every", type=int, default=0' in SRC
    assert 'if sum(bool(x) for x in (a.resume, a.init_from, a.eval_only)) > 1' in SRC
    assert '"eval_args.json" if a.eval_only else "args.json"' in SRC
    assert "if not manifest.is_file():" in SRC
    assert "val_total_tiles={len(val_ds)}" in SRC


def test_holdout_independent_of_calibration():
    helper = (FILE.parent / "holdout_evaluation.py").read_text()
    assert helper.index("calibration_samples(") < helper.index("if args.evaluate_holdout and remaining:")
    assert 'output / "holdout_metrics.json"' in helper
    assert "if not args.no_calibrate:" in helper
