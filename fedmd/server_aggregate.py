import pandas as pd


def aggregate_logits():
    # Load client logits
    ckd = pd.read_csv("fedmd/logits/ckd_logits.csv")
    diabetes = pd.read_csv("fedmd/logits/diabetes_logits.csv")
    heart = pd.read_csv("fedmd/logits/heart_logits.csv")

    # Sanity check
    assert len(ckd) == len(diabetes) == len(heart), "Logit size mismatch!"

    # Combine into one dataframe
    agg_df = pd.DataFrame({
        "ckd_soft": ckd["ckd_logit"],
        "diabetes_soft": diabetes["diabetes_logit"],
        "heart_soft": heart["heart_logit"],
    })

    # Optional: global mean (not required but informative)
    agg_df["mean_soft"] = agg_df.mean(axis=1)

    agg_df.to_csv("fedmd/aggregated_logits.csv", index=False)

    print("✅ Logits aggregated successfully.")
    print(agg_df.head())


if __name__ == "__main__":
    aggregate_logits()