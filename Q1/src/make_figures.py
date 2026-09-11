"""Generate the four Q1 figures from saved results; no solver side effects."""
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent.parent
OUTPUT = SRC.parent / "output"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plot_support import configure, finish
from Q1.src.figures import inputs, net_purchase, dispatch, storage


def main():
    configure()
    with np.load(OUTPUT / "prob1_solution.npz") as z:
        for renderer, filename in [
            (inputs, "fig1_price_load_pv.png"),
            (net_purchase, "fig2_netload_purchase.png"),
            (dispatch, "fig3_buy_charge_discharge.png"),
            (storage, "fig4_storage_energy.png"),
        ]:
            finish(renderer.render(z), OUTPUT, filename)
    print("Generated four figures in " + str(OUTPUT))


if __name__ == "__main__":
    main()
