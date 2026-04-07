import json
import pandas as pd


INPUT_PATH = "../output/regime_ml.json"


def main():
    # -----------------------------
    # LOAD DATA
    # -----------------------------
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    print(f"\nLoaded {len(df)} rows\n")

    # -----------------------------
    # BASIC CHECK
    # -----------------------------
    print("Columns:")
    print(df.columns.tolist())

    # -----------------------------
    # DROP EMPTY ROWS (important)
    # -----------------------------
    df_clean = df.dropna(subset=["regime", "confidence"]).copy()

    # -----------------------------
    # REGIME DISTRIBUTION
    # -----------------------------
    print("\n==============================")
    print("REGIME DISTRIBUTION")
    print("==============================")
    print(df_clean["regime"].value_counts(normalize=True))

    # -----------------------------
    # RETURNS BY REGIME
    # -----------------------------
    print("\n==============================")
    print("AVG ret_21d BY REGIME")
    print("==============================")
    print(df_clean.groupby("regime")["ret_21d"].mean())

    # -----------------------------
    # CONFIDENCE STATS
    # -----------------------------
    print("\n==============================")
    print("CONFIDENCE STATS")
    print("==============================")
    print(df_clean["confidence"].describe())

    # -----------------------------
    # LOW CONFIDENCE CASES
    # -----------------------------
    print("\n==============================")
    print("LOW CONFIDENCE CASES (< 0.6)")
    print("==============================")

    low_conf = df_clean[df_clean["confidence"] < 0.6]
    print(f"Count: {len(low_conf)}")

    if len(low_conf) > 0:
        print("\nSample:")
        print(low_conf.head(10)[[
            "date", "regime", "confidence",
            "prob_bull", "prob_bear", "prob_sideways"
        ]])

    # -----------------------------
    # EXTREME CONFIDENCE
    # -----------------------------
    print("\n==============================")
    print("EXTREME CONFIDENCE (> 0.99)")
    print("==============================")

    high_conf = df_clean[df_clean["confidence"] > 0.99]
    print(f"Count: {len(high_conf)}")

    # -----------------------------
    # PROBABILITY SUM CHECK
    # -----------------------------
    print("\n==============================")
    print("PROBABILITY SUM CHECK")
    print("==============================")

    df_clean["prob_sum"] = (
        df_clean["prob_bear"].fillna(0)
        + df_clean["prob_sideways"].fillna(0)
        + df_clean["prob_bull"].fillna(0)
    )

    print(df_clean["prob_sum"].describe())

    # -----------------------------
    # REGIME TRANSITIONS
    # -----------------------------
    print("\n==============================")
    print("REGIME TRANSITIONS")
    print("==============================")

    df_clean["prev_regime"] = df_clean["regime"].shift(1)
    transitions = df_clean[df_clean["regime"] != df_clean["prev_regime"]]

    print(f"Total transitions: {len(transitions)}")

    print("\nSample transitions:")
    print(transitions.head(10)[
        ["date", "prev_regime", "regime", "confidence"]
    ])


if __name__ == "__main__":
    main()