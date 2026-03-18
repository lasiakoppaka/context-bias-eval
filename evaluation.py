import logging
from typing import Optional
from statistics import mean, stdev

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_contamination_experiment(
    essay_neutral: str,
    essay_anchor: str,
    demographic_label: str,
    model: str,
    num_trials: int = 3,
    score_range: tuple[float, float] = (0.0, 10.0),
) -> dict:
    """
    Measures context contamination bias in LLM essay evaluation.

    Runs two conditions:
      - Baseline: evaluates essay_neutral with no prior context.
      - Anchored: evaluates essay_neutral after the model has already seen
        essay_anchor (which contains a demographic signal), measuring whether
        the prior context "contaminates" the neutral evaluation.

    The contamination score is the difference (anchored - baseline), normalized
    to the scoring range so results are comparable across rubrics/models.

    Args:
        essay_neutral:     The essay to evaluate (no demographic signals).
        essay_anchor:      The essay used to prime the model with a demographic context.
        demographic_label: Label identifying the demographic group being tested (e.g. "Black female").
        model:             The LLM model identifier to use for evaluation.
        num_trials:        Number of times to repeat each condition for averaging (default: 3).
                           Higher values reduce noise at the cost of more API calls.
        score_range:       The (min, max) of the scoring rubric, used to normalize
                           the contamination score (default: 0–10).

    Returns:
        A dict containing:
            demographic_anchor      - The demographic label tested
            baseline_score          - Mean score across trials with no anchor
            anchored_score          - Mean score across trials with anchor context
            contamination_score     - Raw difference: anchored_score - baseline_score
            contamination_pct       - Contamination as % of total score range
            contamination_direction - "positive", "negative", or "none"
            baseline_scores_raw     - Individual trial scores (baseline)
            anchored_scores_raw     - Individual trial scores (anchored)
            baseline_stdev          - Score variability for baseline trials
            anchored_stdev          - Score variability for anchored trials
            num_trials              - Number of trials run
            model                   - Model used
    """

    # --- Input validation ---
    if not essay_neutral or not essay_neutral.strip():
        raise ValueError("essay_neutral cannot be empty.")
    if not essay_anchor or not essay_anchor.strip():
        raise ValueError("essay_anchor cannot be empty.")
    if not demographic_label or not demographic_label.strip():
        raise ValueError("demographic_label cannot be empty.")
    if num_trials < 1:
        raise ValueError(f"num_trials must be at least 1, got {num_trials}.")

    score_min, score_max = score_range
    if score_min >= score_max:
        raise ValueError(f"score_range must be (min, max) with min < max, got {score_range}.")

    logger.info(
        f"Starting contamination experiment | demographic='{demographic_label}' | "
        f"model='{model}' | trials={num_trials}"
    )

    # --- Condition 1: Baseline (no anchor) ---
    baseline_scores = []
    for trial in range(num_trials):
        try:
            score = evaluate_single(essay_neutral, model)
            baseline_scores.append(score)
            logger.debug(f"Baseline trial {trial + 1}/{num_trials}: score={score}")
        except Exception as e:
            logger.error(f"Baseline trial {trial + 1} failed: {e}")
            raise RuntimeError(f"Baseline evaluation failed on trial {trial + 1}: {e}") from e

    # --- Condition 2: Anchored (model primed with demographic essay) ---
    anchored_scores = []
    for trial in range(num_trials):
        try:
            conversation_history = evaluate_single(essay_anchor, model, return_history=True)
            score = evaluate_with_history(essay_neutral, model, history=conversation_history)
            anchored_scores.append(score)
            logger.debug(f"Anchored trial {trial + 1}/{num_trials}: score={score}")
        except Exception as e:
            logger.error(f"Anchored trial {trial + 1} failed: {e}")
            raise RuntimeError(f"Anchored evaluation failed on trial {trial + 1}: {e}") from e

    # --- Aggregate results ---
    baseline_mean = mean(baseline_scores)
    anchored_mean = mean(anchored_scores)
    contamination_score = anchored_mean - baseline_mean

    # Normalize contamination to scoring range (e.g., -1.0 to +1.0 for a 0–10 rubric)
    score_span = score_max - score_min
    contamination_pct = (contamination_score / score_span) * 100

    # Direction label for easy filtering/grouping in analysis
    if contamination_score > 0:
        direction = "positive"
    elif contamination_score < 0:
        direction = "negative"
    else:
        direction = "none"

    # Standard deviation (requires >1 trial; skip if only 1 trial)
    baseline_std = stdev(baseline_scores) if num_trials > 1 else 0.0
    anchored_std = stdev(anchored_scores) if num_trials > 1 else 0.0

    logger.info(
        f"Results | baseline={baseline_mean:.3f} | anchored={anchored_mean:.3f} | "
        f"contamination={contamination_score:+.3f} ({contamination_pct:+.1f}%) | direction={direction}"
    )

    return {
        "demographic_anchor": demographic_label,
        "baseline_score": round(baseline_mean, 4),
        "anchored_score": round(anchored_mean, 4),
        "contamination_score": round(contamination_score, 4),
        "contamination_pct": round(contamination_pct, 2),
        "contamination_direction": direction,
        "baseline_scores_raw": baseline_scores,
        "anchored_scores_raw": anchored_scores,
        "baseline_stdev": round(baseline_std, 4),
        "anchored_stdev": round(anchored_std, 4),
        "num_trials": num_trials,
        "model": model,
    }