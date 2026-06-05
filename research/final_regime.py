import json
import pandas as pd
import numpy as np


INPUT_PATH = "../output/regime_ml.json"
OUTPUT_PATH = "../output/final_regime.json"


# -----------------------------
# SIGNAL → PROBABILITY HELPERS
# -----------------------------
def volatility_to_prob(vol_ratio):
    """
    High vol → bear bias
    Low vol → bull bias
    """
    if vol_ratio is None:
        return [0.33, 0.34, 0.33]

    if vol_ratio > 1.3:
        return [0.6, 0.3, 0.1]  # bear heavy
    elif vol_ratio < 0.8:
        return [0.1, 0.3, 0.6]  # bull heavy
    else:
        return [0.2, 0.6, 0.2]  # sideways


def momentum_to_prob(ret_21d):
    """
    Momentum signal
    """
    if ret_21d is None:
        return [0.33, 0.34, 0.33]

    if ret_21d > 0.05:
        return [0.05, 0.25, 0.7]
    elif ret_21d < -0.05:
        return [0.7, 0.25, 0.05]
    else:
        return [0.2, 0.6, 0.2]


def confidence_weight(conf):
    """
    Reduce overconfidence impact
    """
    if conf is None:
        return 0.5
    return min(conf, 0.9)  # cap


# -----------------------------
# MAIN
# -----------------------------
def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    output = []

    for _, row in df.iterrows():

        # -----------------------------
        # ML PROBS
        # -----------------------------
        p_ml = np.array([
            row["prob_bear"] or 0,
            row["prob_sideways"] or 0,
            row["prob_bull"] or 0
        ])

        # -----------------------------
        # SIGNAL PROBS
        # -----------------------------
        p_vol = np.array(volatility_to_prob(row["vol_ratio"]))
        p_mom = np.array(momentum_to_prob(row["ret_21d"]))

        # -----------------------------
        # WEIGHTS
        # -----------------------------
        w_ml = confidence_weight(row["confidence"])
        w_vol = 0.25
        w_mom = 0.25

        # normalize weights
        total_w = w_ml + w_vol + w_mom
        w_ml /= total_w
        w_vol /= total_w
        w_mom /= total_w

        # -----------------------------
        # FINAL PROB
        # -----------------------------
        p_final = (
            w_ml * p_ml +
            w_vol * p_vol +
            w_mom * p_mom
        )

        # normalize
        p_final = p_final / np.sum(p_final)

        # -----------------------------
        # FINAL REGIME
        # -----------------------------
        labels = ["bear", "sideways", "bull"]

        max_p = np.max(p_final)
        idx = np.argmax(p_final)

        if max_p < 0.6:
            regime = "sideways"
        else:
            regime = labels[idx]

        output.append({
            "date": row["date"],
            "close": row["close"],
            "regime": regime,

            "prob_bear": float(p_final[0]),
            "prob_sideways": float(p_final[1]),
            "prob_bull": float(p_final[2]),
            "confidence": float(max_p),

            # debug
            "ml_conf": row["confidence"]
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Final regime saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()