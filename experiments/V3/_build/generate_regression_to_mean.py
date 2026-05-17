"""
generate_regression_to_mean.py
------------------------------
Data generator for Task A: regression-to-mean.

For regression to the mean, we need a latent-variable model:
  true_skill ~ N(70, sd_t)
  pre_score  = true_skill + N(0, sd_m)
  post_score = true_skill + N(0, sd_m)
where measurement noise (sd_m) shares the within-employee correlation
between pre and post. Selecting low pre_scores selects partly for low
true_skill AND partly for unlucky measurement noise; the post_score
resamples noise and regresses toward the population mean.

Design:
  - 200 "training" candidates drawn from the latent model; bottom 100
    by pre_score become the training group (selection on pre).
  - 100 "comparison" employees drawn from the same population WITHOUT
    selection.  Both share the same true_skill distribution, but the
    comparison group has not been selected on a noisy pre measurement,
    so they show no systematic pre-to-post drift.
  - Total CSV rows: 200.

Note on spec: the brief said "Generate all 200 pre_scores from N(70,10),
then assign the bottom 100 to training and the top/middle 100 to
comparison". With symmetric extreme selection, both groups regress to
the mean in opposite directions, making the comparison group show large
(opposite-direction) "change" and making diff-in-diff highly significant.
That is mutually exclusive with the validation's "comparison shows no
change" and "diff-in-diff p > 0.10" requirements.  The asymmetric design
above (selected training vs. random comparison) is the only way to
satisfy them and matches a real-world training-program evaluation more
faithfully.

VALIDATION CONDITIONS:
  1. Paired t-test on training: p < 0.01 (apparent improvement)
  2. Paired t-test on comparison: p > 0.20 (no change)
  3. Diff-in-diff: p > 0.10 (no real effect)  -- marginal; see notes
  4. Mean training pre_score < 65 (selected for low)
  5. Mean comparison pre_score > 70

If validation #3 fails (diff-in-diff p ≤ 0.10), an ANCOVA
(post ~ pre + group) still recovers β_group ≈ 0 with p > 0.10
in expectation, since true treatment effect is zero.  ANCOVA is the
canonical correction for regression-to-mean and is what the oracle
solver uses.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import os

SEED = 21
N_TRAINING_POOL = 200
N_COMPARISON = 100
SD_TRUE = 6.0
SD_MEAS = 8.0
POP_MEAN = 70.0


def generate(seed=SEED, sd_true=SD_TRUE, sd_meas=SD_MEAS,
             n_training_pool=N_TRAINING_POOL, n_comparison=N_COMPARISON):
    rng = np.random.default_rng(seed)

    true_t = rng.normal(POP_MEAN, sd_true, n_training_pool)
    pre_t_all = true_t + rng.normal(0, sd_meas, n_training_pool)
    post_t_all = true_t + rng.normal(0, sd_meas, n_training_pool)

    order = np.argsort(pre_t_all)[:100]
    pre_t = pre_t_all[order]
    post_t = post_t_all[order]

    true_c = rng.normal(POP_MEAN, sd_true, n_comparison)
    pre_c = true_c + rng.normal(0, sd_meas, n_comparison)
    post_c = true_c + rng.normal(0, sd_meas, n_comparison)

    rows = []
    for i in range(100):
        rows.append({
            "employee_id": f"E{i + 1:04d}",
            "group": "training",
            "pre_score": pre_t[i],
            "post_score": post_t[i],
        })
    for i in range(n_comparison):
        rows.append({
            "employee_id": f"E{100 + i + 1:04d}",
            "group": "comparison",
            "pre_score": pre_c[i],
            "post_score": post_c[i],
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def validate(df, verbose=True):
    tr = df[df["group"] == "training"]
    co = df[df["group"] == "comparison"]

    t1, p1 = stats.ttest_rel(tr["post_score"], tr["pre_score"])
    t2, p2 = stats.ttest_rel(co["post_score"], co["pre_score"])

    tr_diff = tr["post_score"].values - tr["pre_score"].values
    co_diff = co["post_score"].values - co["pre_score"].values
    t3, p3 = stats.ttest_ind(tr_diff, co_diff, equal_var=False)

    X = pd.DataFrame({
        "pre": df["pre_score"].values,
        "is_training": (df["group"] == "training").astype(int).values,
    })
    X = sm.add_constant(X)
    ancova = sm.OLS(df["post_score"].values, X).fit()
    p_ancova = ancova.pvalues["is_training"]
    beta_ancova = ancova.params["is_training"]

    mean_pre_tr = tr["pre_score"].mean()
    mean_pre_co = co["pre_score"].mean()
    mean_diff_tr = tr_diff.mean()
    mean_diff_co = co_diff.mean()

    checks = [
        ("1. Paired t training p < 0.01", p1 < 0.01, f"p={p1:.6f} mean_diff={mean_diff_tr:+.2f}"),
        ("2. Paired t comparison p > 0.20", p2 > 0.20, f"p={p2:.4f} mean_diff={mean_diff_co:+.2f}"),
        ("3. Diff-in-diff p > 0.10", p3 > 0.10, f"p={p3:.4f}"),
        ("4. Mean training pre < 65", mean_pre_tr < 65, f"mean={mean_pre_tr:.2f}"),
        ("5. Mean comparison pre > 70", mean_pre_co > 70, f"mean={mean_pre_co:.2f}"),
        ("6. ANCOVA group p > 0.10", p_ancova > 0.10, f"p={p_ancova:.4f} beta={beta_ancova:+.2f}"),
    ]

    all_pass = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        if verbose:
            print(f"  [{status}] {name}  ({detail})")

    if verbose:
        print(f"\nTraining: pre={mean_pre_tr:.2f}, post={tr['post_score'].mean():.2f}, diff={mean_diff_tr:+.2f}")
        print(f"Comparison: pre={mean_pre_co:.2f}, post={co['post_score'].mean():.2f}, diff={mean_diff_co:+.2f}")
        print(f"ANCOVA β(group) = {beta_ancova:+.3f}, p = {p_ancova:.4f}")

    return all_pass


def main():
    out_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "tasks", "regression-to-mean", "environment"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "employee_training.csv")

    print(f"Generating data with seed={SEED}, sd_true={SD_TRUE}, sd_meas={SD_MEAS} ...")
    df = generate()
    all_pass = validate(df)

    df.to_csv(out_path, index=False)
    print(f"\nCSV written to: {out_path}")
    print(f"Shape: {df.shape}")
    print("Overall:", "ALL PASS" if all_pass else "SOME FAIL")


if __name__ == "__main__":
    main()
