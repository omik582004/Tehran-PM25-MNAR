# Sensitivity-Aware Bayesian Spatio-Temporal Exceedance Mapping under Informative Missingness

Omid Karimi and Fatemeh Hosseini, Semnan University.

This archive accompanies the retrospective reconstruction of the PM2.5 AQI subindex in Tehran, 1–14 January 2024. The response is an index, not a mass concentration. The maps condition on the reporting assumptions specified in the paper.

## Verify the reported results

From this directory, using Python 3 with NumPy installed, run:

    python -m pip install numpy
    python verify_results.py

The checks compare the archived input panels, inspect diagnostic gates and regional area summaries, and recompute held-out scores from saved predictive ensembles. See verification.txt for the expected output. This command does not rerun MCMC.

## Files

- data_original/: supplied source data and GIS files.
- ORIGINAL_INPUTS/: archived analysis inputs.
- DIRECT_RESULTS/: compact tables and diagnostic summaries used to check the paper.
- stages/: empirical model runs and sensitivity analysis (25–26), masked-report evaluation (29), threshold-weighted scores (30), and decision comparisons (31). These directories retain the historical scripts and results; some development scripts refer to the original machine paths or ZIP layouts and may require path changes for refitting.
- stage16_pilot/: historical pilot code and outputs, separate from the final empirical fits.

## Data access

The original monitoring and boundary files are third-party inputs. Their inclusion in this author handoff does not confer redistribution rights. Before publishing this repository, confirm permission for both the supplied files and their copies in the stage archives. If public redistribution is not permitted, replace restricted files with authorized access instructions and amend the manuscript availability statement accordingly.
