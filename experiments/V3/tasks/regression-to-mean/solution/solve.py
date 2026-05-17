import pandas as pd
import statsmodels.api as sm
import json, os


def main():
    df = pd.read_csv("employee_training.csv")

    # ANCOVA: post_score ~ pre_score + group
    # This controls for pre-score differences and corrects for
    # the regression-to-mean bias introduced by selecting the training
    # group on low pre_scores.
    X = pd.DataFrame({
        "pre_score": df["pre_score"].values,
        "is_training": (df["group"] == "training").astype(int).values,
    })
    X = sm.add_constant(X)
    ancova = sm.OLS(df["post_score"].values, X).fit()

    beta_group = float(ancova.params["is_training"])
    p_group = float(ancova.pvalues["is_training"])

    result = {
        "training_effective": bool(p_group < 0.05),
        "p_value": p_group,
        "mean_improvement": beta_group,
        "test_used": "ANCOVA (post ~ pre + group)"
    }

    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
