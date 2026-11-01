# Data

Committed (small, needed at runtime):
- `documentation.json` – the 29 knowledge-base articles the system retrieves from.
- `validation_tickets.json` – the 80 validation tickets used for the full unattended run.

Not committed (copy from `Capstone_Pack/05_Datasets/` if you need them):
- `development_tickets.json` – 500 labelled tickets for development and calibration.
- `ground_truth_responses.json` – 200 reference answers used in evaluation.

The hidden evaluation set is never stored here; the harness takes its path as `--input`.
