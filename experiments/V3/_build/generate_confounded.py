"""
generate_confounded.py
----------------------
Data generator for Task C: confounded-comparison.

Two programs (A, B) with TRUE identical effect (+10) on baseline_score.
Program A was offered to high-performing employees (baseline ≈ 80),
Program B to lower-performing employees (baseline ≈ 60). Naive
between-program t-test on outcome_score shows A is much "better"
because A started 20 points higher -- not because the program is better.

Four irrelevant columns (department, years_experience, location,
employee_type) are sprinkled in to act as plausible-looking but unhelpful
covariates the agent might pick up if they're stuck on covariate
selection.

VALIDATION CONDITIONS:
  1. Naive t-test on outcome_score between programs: p < 0.001
  2. Regression of outcome_score on program + baseline_score:
     program coefficient p > 0.10
  3. Mean baseline_score for A > 75
  4. Mean baseline_score for B < 65
  5. |Correlation between program assignment and baseline_score| > 0.70
  6. None of the 4 irrelevant columns has p < 0.10 association with outcome
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import os

SEED = 21  # Tuned: seed=42 had employee_type p=0.089 (just under 0.10); seed=21 puts all nuisance p > 0.30
N_PER_PROGRAM = 100


def generate(seed=SEED, n=N_PER_PROGRAM):
    rng = np.random.default_rng(seed)

    baseline_a = rng.normal(80, 6, n)
    baseline_b = rng.normal(60, 6, n)

    outcome_a = baseline_a + 10 + rng.normal(0, 5, n)
    outcome_b = baseline_b + 10 + rng.normal(0, 5, n)

    rows = []
    pid = 1
    for ba, oa in zip(baseline_a, outcome_a):
        rows.append({
            "participant_id": f"P{pid:04d}",
            "program": "A",
            "baseline_score": ba,
            "department": int(rng.integers(1, 6)),
            "years_experience": int(rng.integers(1, 21)),
            "location": rng.choice(["east", "west", "central"]),
            "employee_type": rng.choice(["full", "part"]),
            "outcome_score": oa,
        })
        pid += 1
    for bb, ob in zip(baseline_b, outcome_b):
        rows.append({
            "participant_id": f"P{pid:04d}",
            "program": "B",
            "baseline_score": bb,
            "department": int(rng.integers(1, 6)),
            "years_experience": int(rng.integers(1, 21)),
            "location": rng.choice(["east", "west", "central"]),
            "employee_type": rng.choice(["full", "part"]),
            "outcome_score": ob,
        })
        pid += 1

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def validate(df, verbose=True):
    a = df[df["program"] == "A"]
    b = df[df["program"] == "B"]

    t_naive, p_naive = stats.ttest_ind(a["outcome_score"], b["outcome_score"],
                                       equal_var=False)

    X = pd.DataFrame({
        "is_a": (df["program"] == "A").astype(int).values,
        "baseline": df["baseline_score"].values,
    })
    X = sm.add_constant(X)
    reg = sm.OLS(df["outcome_score"].values, X).fit()
    p_program = float(reg.pvalues["is_a"])
    beta_program = float(reg.params["is_a"])

    mean_b_a = a["baseline_score"].mean()
    mean_b_b = b["baseline_score"].mean()

    prog_num = (df["program"] == "A").astype(int).values
    corr = float(np.corrcoef(prog_num, df["baseline_score"].values)[0, 1])

    nuisance_ps = {}
    for col in ["department", "years_experience"]:
        v = df[col].values
        slope, intercept, r, p, se = stats.linregress(v, df["outcome_score"].values)
        nuisance_ps[col] = float(p)
    for col in ["location", "employee_type"]:
        groups = [df.loc[df[col] == lvl, "outcome_score"].values
                  for lvl in df[col].unique()]
        if len(groups) > 1:
            _, p = stats.f_oneway(*groups)
            nuisance_ps[col] = float(p)
    nuisance_min = min(nuisance_ps.values())

    checks = [
        ("1. Naive t-test p < 0.001", p_naive < 0.001, f"p={p_naive:.2e}"),
        ("2. Regression w/ baseline: program p > 0.10",
         p_program > 0.10, f"p={p_program:.4f} beta={beta_program:+.2f}"),
        ("3. Mean baseline A > 75", mean_b_a > 75, f"mean={mean_b_a:.2f}"),
        ("4. Mean baseline B < 65", mean_b_b < 65, f"mean={mean_b_b:.2f}"),
        ("5. |corr(program, baseline)| > 0.70", abs(corr) > 0.70, f"r={corr:+.3f}"),
        ("6. All nuisance p > 0.10", nuisance_min > 0.10,
         f"min p={nuisance_min:.3f} all={nuisance_ps}"),
    ]

    all_pass = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        if verbose:
            print(f"  [{status}] {name}  ({detail})")

    if verbose:
        print(f"\nA: baseline={mean_b_a:.2f}, outcome={a['outcome_score'].mean():.2f}")
        print(f"B: baseline={mean_b_b:.2f}, outcome={b['outcome_score'].mean():.2f}")

    return all_pass


def main():
    out_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "tasks", "confounded-comparison", "environment"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "program_comparison.csv")

    print(f"Generating data with seed={SEED} ...")
    df = generate()
    all_pass = validate(df)

    df.to_csv(out_path, index=False)
    print(f"\nCSV written to: {out_path}")
    print(f"Shape: {df.shape}")
    print("Overall:", "ALL PASS" if all_pass else "SOME FAIL")


if __name__ == "__main__":
    main()
