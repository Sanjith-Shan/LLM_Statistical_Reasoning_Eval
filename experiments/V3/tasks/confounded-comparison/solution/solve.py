import pandas as pd
import statsmodels.api as sm
import json, os


def main():
    df = pd.read_csv("program_comparison.csv")

    # Regression of outcome_score on program and baseline_score.
    # baseline_score is a strong confounder (mean A ≈ 80, mean B ≈ 60)
    # and must be controlled for to recover the true (zero) program effect.
    X = pd.DataFrame({
        "is_a": (df["program"] == "A").astype(int).values,
        "baseline": df["baseline_score"].values,
    })
    X = sm.add_constant(X)
    reg = sm.OLS(df["outcome_score"].values, X).fit()

    beta_program = float(reg.params["is_a"])
    p_program = float(reg.pvalues["is_a"])

    mean_a = float(df.loc[df["program"] == "A", "outcome_score"].mean())
    mean_b = float(df.loc[df["program"] == "B", "outcome_score"].mean())

    result = {
        "program_a_better": bool(p_program < 0.05 and beta_program > 0),
        "p_value": p_program,
        "mean_outcome_a": mean_a,
        "mean_outcome_b": mean_b,
        "test_used": "OLS regression with baseline_score as covariate",
    }

    os.makedirs("/output", exist_ok=True)
    with open("/output/analysis.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
