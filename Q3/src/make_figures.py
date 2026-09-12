"""Generate Q3 publication figures from saved results without solving."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dispatch_core.plotting import make_solution_figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution", nargs="?", default=str(ROOT / "Q3/output/full_solution.npz"))
    parser.add_argument("--sensitivity", type=Path, default=ROOT / "Q3/analysis/horizon_sensitivity.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Q3/figures")
    args = parser.parse_args(argv)
    sensitivity = args.sensitivity if args.sensitivity.exists() else None
    paths = make_solution_figures(args.solution, args.output_dir, question="Q3",
                                  dynamic_price=False, sensitivity_csv=sensitivity)
    print(f"generated {len(paths)} files in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

