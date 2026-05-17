import json, sys, os


def main():
    reward_dir = "/logs/verifier"
    os.makedirs(reward_dir, exist_ok=True)
    reward_path = os.path.join(reward_dir, "reward.txt")

    try:
        output_path = os.environ.get("ANALYSIS_JSON", "/output/analysis.json")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"{output_path} not found")

        with open(output_path) as f:
            result = json.load(f)

        for field in ("program_a_better", "p_value"):
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        p_value = float(result["p_value"])
        program_a_better = result["program_a_better"]

        # KEY DISCRIMINATOR.
        # Programs A and B have IDENTICAL true effects (+10 on baseline).
        # Naive between-program t-test on outcome_score: p < 1e-40 (A looks
        #   much better -- false positive from confounding).
        # Gain-score t-test (post - pre): p ≈ 0.001 (B looks slightly better --
        #   false positive from sample noise in gain difference).
        # OLS with baseline as covariate (ANCOVA): p > 0.6 (correctly null --
        #   the only analysis that properly handles the confounder).
        # The verifier requires p > 0.05 in either direction: any significant
        # difference between the programs is a false positive.
        if not (p_value > 0.05):
            raise ValueError(
                f"p_value={p_value:.6e}: analysis reports a statistically "
                "significant difference between programs A and B. "
                "Both programs have identical true effects (+10 on baseline). "
                "Naive t-test on outcomes is confounded by the huge baseline "
                "gap (A ~80, B ~60). Gain-score analysis (post - pre) controls "
                "for level but still yields a chance significant result "
                "because of random sample variation in the gain difference. "
                "Only ANCOVA (post ~ pre + program) correctly recovers the "
                "null (p > 0.6). Expected p > 0.05."
            )

        with open(reward_path, "w") as f:
            f.write("1")
        print("reward=1 reason=all checks passed")
        return True

    except Exception as e:
        with open(reward_path, "w") as f:
            f.write("0")
        print(f"reward=0 reason={e}")
        return False


if __name__ == "__main__":
    main()
    sys.exit(0)
