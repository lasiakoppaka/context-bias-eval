def run_contamination_experiment(essay_neutral, essay_anchor, demographic_label, model):
    
    # Condition 1: Baseline (no anchor)
    baseline_score = evaluate_single(essay_neutral, model)
    
    # Condition 2: Anchored
    conversation_history = evaluate_single(essay_anchor, model, return_history=True)
    anchored_score = evaluate_with_history(essay_neutral, model, history=conversation_history)
    
    contamination_score = anchored_score - baseline_score
    
    return {
        "demographic_anchor": demographic_label,
        "baseline_score": baseline_score,
        "anchored_score": anchored_score,
        "contamination_score": contamination_score
    }