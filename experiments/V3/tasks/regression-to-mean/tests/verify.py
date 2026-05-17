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

        for field in ("training_effective", "p_value", "mean_improvement"):
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        p_value = float(result["p_value"])
        training_effective = result["training_effective"]

        # KEY DISCRIMINATOR.
        # Naive paired t-test on training group only: p < 0.001 (apparent
        # improvement from regression to the mean -- the training group was
        # selected for low pre_scores, so they regress upward purely by chance).
        # Correct analysis (ANCOVA post ~ pre + group): p > 0.10
        # (no real treatment effect).
        if not (p_value > 0.05 or training_effective == False):
            raise ValueError(
                f"p_value={p_value:.6f} and training_effective={training_effective}: "
                "analysis reports a significant training effect. "
                "This indicates a within-group paired test was run on the "
                "selected training group alone, ignoring the comparison group "
                "and the regression-to-mean bias from selecting on pre-score. "
                "Expected p > 0.05 OR training_effective=false."
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
