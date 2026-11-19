# Stage 31: Decision-focused secondary audit

This package adds **no new posterior fitting**. It relies on the prior accepted Stage29 Gaussian–MAR and fixed-gamma Gaussian–MNAR masked-report posterior predictive ensembles, and on the accepted Stage26 full-data cellwise maps. A separate original Stage30 ZIP is retained byte-for-byte in this package; it contains the Stage29 archive. We additionally provide the two small Stage26 accepted source CSVs so that the new analyses are self-contained.

## Reproduction

Python dependencies: numpy, pandas, scikit-learn, matplotlib. In this directory run:

```
python evaluate_decisions_v31.py
python plot_decision_v31.py
python validate_stage31.py
```

The 20,000 paired day-block bootstrap replicates use seed 20261031. Both read-only sources are identified by SHA-256 in `SHA256SUMS.txt`. Re-fitting the model is unnecessary for this secondary reanalysis. The evaluators read the separately stored withheld outcomes only to compute scores, not to fit the original models. Exact event definition: recorded reading *strictly greater than* 100, 120 or 150.

## Outputs

- `heldout_decision_curve_v31.csv`: true/false positives, false negatives and illustrative net benefits at eight probability cutoffs for three events
- `heldout_discrimination_v31.csv`: event counts, Brier, AUROC, average precision
- `heldout_brier_decomposition_v31.csv`: outcome-stratified and overall paired day-block uncertainty
- `area_weighted_decision_sensitivity_v31.csv`: accepted Stage26 cell-map area classifications at four probability cutoffs, all three dates and three subindex thresholds
- `regional_event_tipping_illustration_v31.csv`: accepted joint-area event probabilities and purely illustrative action cutoff
- manuscript and SI contain an additional dedicated analysis, with a derived publication figure

## Important inferential boundary

Masked reported readings have known R=1. This is a post-hoc decision audit of the same validation set as Stages29/30, NOT independent validation at genuinely missing R=0 outcomes. The Stage26 full-data map classification is assumption-conditional risk sensitivity, NOT observed map accuracy. A tiny favorable Brier(150) difference does not establish superior detection and cannot be selected as confirmatory evidence. Submission of underlying raw data/GIS requires source verification and redistribution authorization.
