# Legacy Exploratory Data (LEGACY_EXPLORATORY_V0)

## Status

These runs were started **before** the prospective prediction-lock protocol was defined. They are tagged `LEGACY_EXPLORATORY_V0` and may **not** be used as confirmatory evidence.

## What happened

On September 17, 2026, a 40-run core experiment was launched at q=113 testing:
- Group: {additive, multiplicative}
- Tokenizer: {integer, bit}
- Optimization: {Paper 1 progressive WD, Paper 2 fixed WD}
- Seeds: {42, 123, 456, 789, 2026}

The experiment design predates the new prediction study protocol. The runs began before any prediction lock existed. Therefore they cannot serve as prospective confirmatory evidence.

## Quarantine action

- The launcher was stopped on September 17, 2026 at ~21:44 UTC.
- 3 runs completed (additive + integer + paper1 opt, seeds 42/123/456).
- 1 run was stopped at 16.7% progress (seed 789) — not >=80% and not the smallest seed.
- 36 runs were never started.

## Permitted uses

These runs may be used for:
- Debugging data pipelines
- Runtime calibration and training budget estimation
- Exploratory predictor development (feature engineering, hyperparameter selection)
- Checking that the training loop produces valid metrics

## Forbidden uses

These runs must NOT be used in:
- Prospective accuracy claims
- Confirmatory predictor evaluation
- Blind test metrics
- Primary statistical significance tests
- Any figure or table presented as confirmatory evidence

## Files

- `legacy_40_status_<timestamp>.csv` — Run status at time of quarantine
- `../results/matched_isomorphic/` — The actual run data (metrics, configs, summaries)

## Note

The additive-vs-multiplicative comparison is no longer the primary causal representation intervention. The main experiment now uses GF(2) affine bijections on fixed-length binary representations, which provide a cleaner causal intervention (same bits, same length, same information, same labels, different coordinate geometry). The additive-vs-multiplicative comparison is retained as a secondary demonstration only (CONF_D).
