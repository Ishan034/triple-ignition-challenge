# Triple Ignition Challenge

A contamination monitoring and diagnosis system for **MaverickX**, an enzyme company, to help address contamination in its enzyme production.

## Goal

Build a decision-support prototype that analyses enzyme-production batch data to:

1. **Detect** batches that deviate from their normal production profile.
2. **Trace** plausible contamination sources from batch history (inputs, equipment, process steps, timing).
3. **Assess** confidence and severity as separate dimensions.
4. **Recommend** the next confirmatory check, leaving final decisions to scientists.

The system investigates contamination; it does not predict future contamination or replace laboratory confirmation. See [`docs/project-context.md`](docs/project-context.md) for the full background, architecture, MVP scope and validation plan.

## Status

Layer 1 (anomaly detection) is working end to end on **synthetic** α-amylase fermentation data with simulated bacterial (*E. coli*) contamination, while real MaverickX batch data is requested. Next: diagnostic rules, source tracing and the Streamlit dashboard. Deadline: 28 October 2026.

## Running the demo

```bash
python -m contamination.demo     # writes data/synthetic/ and outputs/
```

This generates synthetic batches, validates them, fits a frozen reference on the 30 earliest (normal) batches, monitors the 25 later batches, evaluates against the separate outcomes table, and writes `outputs/alerts.csv`, `batch_summary.csv`, `detector_comparison.csv`, `reference_profile.json` and figures in `outputs/figures/`. Results on synthetic data show the software behaves as intended. They are **not** real-world detection performance.

## Code map

| Module | Layer | What it does |
|---|---|---|
| `schema.py` | data model | Variables, units, allowed codes; measurements / outcomes / batches schemas |
| `synthetic.py` | data | Synthetic batches: normal, slow growth, contamination, pH sensor spike, DO probe drift, missing-data gaps |
| `ingest.py` | 1 · evidence acquisition | Loads CSVs; flags (never deletes) missing, suspect, duplicate and wrong-unit readings |
| `reference.py` | 3 · reference memory | Robust per-hour reference (median, 1.4826·MAD), frozen and versioned |
| `detectors.py` | 4 · anomaly detection | Robust z threshold, EWMA, two-sided CUSUM |
| `monitor.py` | 4 · anomaly detection | Leave-one-out limit calibration; scoring; structured alerts; batch summary; `as_of_h` replay |
| `alerts.py` | 4 · output | `Alert` object: evidence, persistence, corroboration, data-quality warnings, versions |
| `evaluate.py` | validation | Sensitivity, specificity, precision, false alerts per batch, lead time vs today's method |
| `plots.py` | reporting | Reference band vs observed trajectory with alerts |

## Planned structure

```
data/        Synthetic and (later) authorised batch data
notebooks/   Exploration and experiments
src/         Ingestion, preprocessing, anomaly detection and diagnostic rules
app/         Streamlit investigation dashboard
tests/       pytest suite
docs/        Project context and technical notes
```

## Planned stack

Python, pandas, NumPy/SciPy, scikit-learn, Streamlit, Plotly, NetworkX, pytest.

## Getting started

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.lock   # exact pinned versions
pip install -e ".[dev]"            # installs the project package
pytest                             # run tests
```

Dependencies are declared in `pyproject.toml`; `requirements.lock` pins the versions the team uses.

## Contributors

- Ishan Singh ([@Ishan034](https://github.com/Ishan034))
- Aisha Jawad
- Beren Abedan
- Trisha
