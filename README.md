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

Early stage. The MVP is being built on synthetic α-amylase fermentation data with simulated bacterial (*E. coli*) contamination while real MaverickX batch data is requested. Deadline: 28 October 2026.

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
