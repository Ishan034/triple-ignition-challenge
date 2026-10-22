# Data template

Example exports in the shapes plants usually have, plus the `mapping.yaml` that tells the
importer which file and column holds what. The values are **simulated** (4 batches from the
synthetic demo: 2 normal, 1 contaminated, 1 with a drifting oxygen probe). They show the
format, not real data.

| File | What it holds | Layout |
|---|---|---|
| `historian_export.csv` | Online signals (pH, temperature, dissolved oxygen) | one column per signal |
| `lab_results.csv` | Offline results, with the time the sample was taken **and** the time the result came back | one row per result |
| `operator_notes.csv` | Timestamped observations (smell, colour, foam) | one row per note |
| `batch_records.csv` | Batch start (inoculation) time, vessel, medium lot, seed lot | one row per batch |
| `batch_outcomes.csv` | What happened to each batch, kept apart from the measurements | one row per batch |
| `mapping.yaml` | Which file and column holds what | |

Your own exports can have other file names, column names and extra columns: change the
mapping to match, and add controller outputs (base pump, cooling) and any other signals
under `measurements` and `variables`. Then:

```bash
python -m contamination.importer convert path/to/mapping.yaml --out data/maverickx/campaign
python -m contamination.importer readiness data/maverickx/campaign
```

or use **Load your data** in the dashboard. Nothing is filled, rescaled or deleted:
unusual values are flagged for review, and columns the mapping does not use are listed.
