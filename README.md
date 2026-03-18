# context-bias-eval

A research framework for measuring **context contamination bias** in LLM essay evaluation. This project investigates whether a language model's prior exposure to a demographically-signaled essay influences its subsequent evaluation of a neutral essay — even when the neutral essay contains no demographic information.

---

## What is Context Contamination?

When an LLM evaluates essays in a multi-turn conversation, a prior response to an essay containing demographic signals (e.g., a name associated with a particular race or gender) may "contaminate" the model's next evaluation. This project quantifies that contamination by comparing:

- **Baseline score** — the model evaluates a neutral essay with no prior context
- **Anchored score** — the model evaluates the same neutral essay *after* having seen a demographically-anchored essay in the conversation history

The difference between these two scores is the **contamination score**.

---

## Metric
```
contamination_score = anchored_score - baseline_score
contamination_pct   = (contamination_score / score_range) * 100
```

A positive score means the demographic anchor inflated the neutral evaluation.  
A negative score means the anchor suppressed it.  
Results are normalized to the rubric's scoring range for cross-model comparability.

---

## Usage
```python
result = run_contamination_experiment(
    essay_neutral="The challenge that changed me was learning to lead...",
    essay_anchor="Growing up as a first-generation immigrant, DeShawn...",
    demographic_label="Black male",
    model="gpt-4o",
    num_trials=3,
    score_range=(0.0, 10.0)
)
```

---

## Key Features

- **Multi-trial averaging** — each condition is run `num_trials` times to reduce LLM non-determinism
- **Normalized contamination score** — comparable across models and rubrics
- **Direction labeling** — flags whether bias is positive, negative, or absent
- **Standard deviation tracking** — captures whether the anchor increases evaluation *inconsistency*, not just shifts the mean
- **Full error handling and logging** — research-grade reliability

---

## Research Context

This work is part of an ongoing study on demographic bias in LLM-based college essay evaluation at UC Santa Cruz. The contamination metric extends classical paired-testing methodology into the multi-turn setting, where bias can compound across conversation turns.

Related metrics being developed:
- Demographic score gap (single-turn paired testing)
- Rubric dimension disaggregation (where in the rubric does bias appear?)
- Cross-model bias benchmarking (GPT-4o vs Claude vs Gemini)


