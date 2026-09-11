"""Generate the four Q2 figures from saved results; no solver side effects."""
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent.parent
OUTPUT = SRC.parent / "output"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_support import configure, finish
from Q2.src.figures import annual, specified, emergency, storage


def main():
    configure()
    with np.load(OUTPUT / "prob2_solution.npz") as z:
        for renderer, filename in [
            (annual, "fig1_annual_cost_emergency.png"),
            (specified, "fig2_specified_days_dispatch.png"),
            (emergency, "fig3_emergency_heatmap.png"),
            (storage, "fig4_storage_soc.png"),
        ]:
            finish(renderer.render(z), OUTPUT, filename)
    print("Generated four figures in " + str(OUTPUT))


if __name__ == "__main__":
    main()
