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

The MVP works end to end on **synthetic** α-amylase fermentation data with simulated bacterial (*E. coli*) contamination: anomaly detection, diagnosis, source tracing and an investigation dashboard. It has also been evaluated on external data: the IndPenSim industrial penicillin simulation (100 batches, labelled equipment faults) and 406 real erythromycin plant batches. See [`docs/external-validation.md`](docs/external-validation.md). Real MaverickX batch data has been requested. Synthetic results and design decisions are in [`docs/mvp-results.md`](docs/mvp-results.md). Deadline: 28 October 2026.

## Investigation dashboard

```bash
python -m contamination.demo        # once, to generate data/synthetic/
streamlit run app/dashboard.py      # opens http://localhost:8501
```

- **Campaign overview**: every monitored batch with its status, process deviation, leading explanation, contamination evidence and affected scope.
- **Batch investigation**: signal timelines against the reference band with alerts; a **replay** slider that shows what the system knew at any hour; ranked explanations, evidence, next checks and possible sources; a form to **record check results** (the assessment updates immediately); a downloadable report.
- **Source tracing**: a genealogy graph of contamination cases and the resources linking them, clusters, and the full candidate table.
- **How it works**: a plain-language explanation and the system's limits.

A "Show simulated ground truth" toggle (synthetic data only) shows the known answers for evaluation. The model never sees them.

### Deploying (Streamlit Community Cloud)

On [share.streamlit.io](https://share.streamlit.io), create an app from this repo: branch `main`, main file `app/dashboard.py`, Python 3.13 (under Advanced settings). Dependencies come from `requirements.txt`, and the simulated campaign in `data/synthetic/` is committed, so nothing else is needed. Set sharing to invite-only to keep it to the team.

## Running the demo

```bash
python -m contamination.demo                      # writes data/synthetic/ and outputs/
python -m contamination.robustness --seeds 11 31  # multi-seed evaluation (~4 min)
```

The demo generates synthetic batches, validates them, fits a frozen reference on the 30 earliest (normal) batches, monitors and diagnoses the 25 later batches, evaluates against the separate outcomes table, and writes `outputs/alerts.csv`, `batch_summary.csv`, `detector_comparison.csv`, `diagnosis.csv`, `diagnosis_evaluation.csv`, `batch_statuses.csv`, `source_candidates.csv`, `reference_profile.json`, one investigation report per flagged batch in `outputs/reports/`, and figures in `outputs/figures/`. Results on synthetic data show the software behaves as intended. They are **not** real-world detection performance.

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
| `diagnosis.py` | 5 · diagnostic reasoning, 3 · confidence/severity, 4 · next checks | Evidence items (present/absent/unknown), transparent hypothesis ranking, safety rule, investigation reports |
| `knowledge/catalogue.yaml` | 5 · knowledge base | Hypotheses, evidence matrix, weights, checks: a **draft** for scientist review |
| `tracing.py` | 2 · source tracing | Batch genealogy graph (NetworkX); candidate sources ranked by over-representation (Fisher's exact test, Bonferroni, confounding check); clusters; affected scope |
| `datasets/indpensim.py`, `datasets/erythromycin.py` | external data | Download and convert external datasets into campaign folders; evaluate on them |
| `knowledge/profiles/*.yaml` | 5 · knowledge base | Process profiles mapping the rule book's roles (biomass, oxygen, pH, product, temperature) to each process's variables |
| `robustness.py` | validation | Multi-seed evaluation of detection and diagnosis |
| `evaluate.py` | validation | Sensitivity, specificity, precision, false alerts per batch, lead time vs today's method |
| `pipeline.py` | orchestration | Load a campaign folder; run detection → diagnosis → tracing; per-batch replay |
| `app/dashboard.py` | 6 · decision support | Streamlit investigation dashboard |
| `plots.py` | reporting | Reference band vs observed trajectory with alerts |

## Structure

```
data/        Synthetic and (later) authorised batch data
notebooks/   Exploration and experiments
src/         Ingestion, preprocessing, anomaly detection and diagnostic rules
app/         Streamlit investigation dashboard
tests/       pytest suite
docs/        Project context and technical notes
```

## Stack

Python, pandas, NumPy/SciPy, pandera, scikit-learn (comparison only), NetworkX, PyYAML, Streamlit, Plotly, Matplotlib, pytest, ruff.

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
