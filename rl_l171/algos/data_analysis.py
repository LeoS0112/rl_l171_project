import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= CONFIG =========
ENTITY = "leosanitt-university-of-cambridge"
PROJECT = f"{ENTITY}/rl171_sweep"
SWEEP_ID = "leosanitt-university-of-cambridge/rl171_sweep/b7yxqdp8"

METRIC = "eval/cube_distance_mean"
STEP = "global_step"
BUFFER_KEY = "buffer_strategy"
SEED_KEY = "seed"
TOP_K = 3

# If lower distance is better, keep True
BETTER_IS_LOWER = True

# You must choose a target performance level for speedup calculation
TARGET = 0.4  # <-- choose something meaningful for your task
# ==========================

if __name__=="__main__":
    api = wandb.Api()
    runs = api.runs(PROJECT, filters={"state": "finished"})

    rows = []


    for run in runs:

        cfg = dict(run.config)
        # print(cfg)
        # Require buffer_strategy and seed
        if BUFFER_KEY not in cfg or SEED_KEY not in cfg:
            print("No buffer key")
            continue
        # Skip runs with insufficient training length
        if "total_timesteps" not in cfg or cfg["total_timesteps"] < 50_000:
            print("Too short")
            continue
        if cfg.get("wandb_sweep_id") != SWEEP_ID:
            print("Wrong sweep")
            continue
        print(cfg)

        # Pull eval history
        hist = run.history(keys=[STEP, METRIC], pandas=True).dropna(subset=[STEP, METRIC])
        if hist.empty:
            continue

        # Build a config-group key: all config entries except seed and wandb internals
        cfg_clean = {
            k: v
            for k, v in cfg.items()
            if not k.startswith("_") and k not in [SEED_KEY]
        }
        cfg_items = tuple(sorted(cfg_clean.items()))
        # group_id uniquely identifies "same HPs except seed"
        group_id = hash(cfg_items)

        for _, r in hist.iterrows():
            rows.append(
                {
                    "run_id": run.id,
                    "group_id": group_id,
                    BUFFER_KEY: cfg[BUFFER_KEY],
                    SEED_KEY: cfg[SEED_KEY],
                    STEP: r[STEP],
                    METRIC: r[METRIC],
                }
            )

    df = pd.DataFrame(rows)
    print("Total datapoints:", len(df))
    print("Total runs:", df["run_id"].nunique())
    print("Total config groups:", df["group_id"].nunique())

    # ============== 1) AVERAGE OUT SEEDS ==============

    # One curve per (config group, buffer strategy, step), seeds averaged away
    seed_avg = (
        df.groupby(["group_id", BUFFER_KEY, STEP], as_index=False)[METRIC]
        .mean()
        .rename(columns={METRIC: "metric_seed_avg"})
    )

    # ============== 2) MAX VALUE PER CONFIG GROUP ==============

    group_max = (
        seed_avg.groupby(["group_id", BUFFER_KEY], as_index=False)["metric_seed_avg"]
        .min()
        .rename(columns={"metric_seed_avg": "max_metric"})
    )

    # rank within each buffer strategy, best = highest max_metric (flip sign if lower is better)
    if BETTER_IS_LOWER:
        group_max["score_for_rank"] = -group_max["max_metric"]
    else:
        group_max["score_for_rank"] = group_max["max_metric"]

    group_max["rank_within_strategy"] = (
        group_max.groupby(BUFFER_KEY)["score_for_rank"]
        .rank(method="first", ascending=False)
    )

    top_groups = group_max[group_max["rank_within_strategy"] <= TOP_K]
    top_group_ids = set(top_groups["group_id"].tolist())
    print("Top groups per strategy:")
    print(top_groups[[BUFFER_KEY, "group_id", "max_metric", "rank_within_strategy"]])

    # ============== 3) MEAN CURVE OF TOP-K CONFIGS PER STRATEGY ==============

    top_seed_curves = seed_avg[seed_avg["group_id"].isin(top_group_ids)]

    mean_curve = (
        top_seed_curves
        .groupby([BUFFER_KEY, STEP], as_index=False)["metric_seed_avg"]
        .mean()
        .rename(columns={"metric_seed_avg": "metric_top5_mean"})
    )

    # Optional: you could also keep std / count if you want error bands
    # stats_curve = (
    #     top_seed_curves
    #     .groupby([BUFFER_KEY, STEP])["metric_seed_avg"]
    #     .agg(["mean", "std", "count"])
    #     .reset_index()
    # )

    # ============== 4) PLOT PERFORMANCE CURVES ==============

    plt.figure(figsize=(7, 5))

    for strategy, g in mean_curve.groupby(BUFFER_KEY):
        g_sorted = g.sort_values(STEP)
        plt.plot(
            g_sorted[STEP],
            g_sorted["metric_top5_mean"],
            label=strategy,
        )

    plt.xlabel(STEP)
    plt.ylabel(f"{METRIC} (top-{TOP_K} mean, seeds-avg)")
    plt.title("Performance per buffer strategy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ============== 5) SPEEDUP CALCULATION ==============
    # Define: time to hit target performance for each strategy

    def time_to_threshold(curve_df, target, better_is_lower=True):
        curve_df = curve_df.sort_values(STEP)
        if better_is_lower:
            hit = curve_df[curve_df["metric_top5_mean"] <= target]
        else:
            hit = curve_df[curve_df["metric_top5_mean"] >= target]

        if hit.empty:
            return np.nan
        return hit[STEP].iloc[0]

    times = {}
    for strategy, g in mean_curve.groupby(BUFFER_KEY):
        times[strategy] = time_to_threshold(g, TARGET, BETTER_IS_LOWER)

    print("Time to reach target:", times)

    # Pick a baseline strategy to compare against
    BASELINE = "uniform"  # <-- change to whatever your baseline is
    baseline_time = times.get(BASELINE, np.nan)

    speedups = {}
    if not np.isnan(baseline_time):
        for strat, t in times.items():
            if strat == BASELINE or np.isnan(t):
                continue
            speedups[strat] = baseline_time / t

    print(f"Speedups vs {BASELINE} at target={TARGET}:")
    print(speedups)
