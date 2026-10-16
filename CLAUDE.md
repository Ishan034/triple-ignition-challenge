# CLAUDE.md

Read `docs/project-context.md` before working on this repo. It holds the client background,
architecture, MVP scope and validation rules, imported from the team's Notion.

## Key rules for code in this repo

- Scope is contamination **detection + diagnosis decision support** for MaverickX enzyme
  production. The earlier lunar/asteroid mining and mineral prediction-map ideas are shelved:
  do not reintroduce them.
- Never present an anomaly score as a contamination probability, and never claim a specific
  organism was identified from process variables (pH, temperature, DO) alone.
- Keep anomaly score and contamination evidence separate, and keep confidence and severity
  separate.
- Keep batch outcomes in a separate table from measurements to avoid label leakage. Split
  train/test by whole batch, and only use data available at alert time.
- Missing values stay explicit: never fill them with zero, and never silently drop unusual values.
- Prefer simple, explainable methods first (robust thresholds, then EWMA/CUSUM). Multivariate
  and ML methods are only comparisons.
- Alerts are structured objects (variable, observed vs reference, magnitude, algorithm and
  reference version, data-quality warnings, persistence, confirmation status).
- Label synthetic-data results as software demonstrations, not real-world detection performance.
