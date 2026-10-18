# External validation: IndPenSim and real erythromycin plant data

Until now every result came from our own synthetic data, where the simulator and the
rule book share the same assumptions. This is the first test on data we did not write.
**Neither dataset contains contamination**, so this tests false alarms, early detection
of genuine process deviations, and whether the system avoids calling a non-biological
problem "contamination". Contamination detection itself still needs MaverickX's
records.

Reproduce (downloads ~510 MB, kept out of git):

```bash
python -m contamination.datasets.indpensim prepare && python -m contamination.datasets.indpensim evaluate
python -m contamination.datasets.erythromycin prepare && python -m contamination.datasets.erythromycin evaluate
```

## 1. IndPenSim: 100 batches of a simulated 100,000 L penicillin process

**Data.** [IndPenSim](https://data.mendeley.com/datasets/pdnjz7zz5x/2) (Goldrick et al.
2015, CC BY 4.0) is a high-fidelity simulation built on historical records of an
industrial penicillin process. It has 100 batches of 167–290 h with realistic
batch-to-batch variation under three control strategies: recipe (batches 1–30),
operator (31–60) and advanced process control (61–90). Batches 91–100 contain labelled
equipment faults. We monitored only process outputs: pH, temperature, dissolved oxygen,
OUR, CER, off-gas CO₂, substrate, and offline biomass and penicillin (12-hourly, 4 h
result delay). The faulted inputs themselves are not monitored, because watching the
changed input would make detection trivial.

**Split.** The reference was built from 20 normal batches of each control strategy (60).
We tested on the other 30 normal batches and all 10 fault batches.

| Measure | Result |
|---|---|
| Normal held-out batches flagged | **1 / 30** (one isolated temperature reading, read as a likely measurement glitch) |
| Fault batches flagged after the fault started | **8 / 10** |
| …during the first fault window | **8 / 10** |
| Median time from fault onset to first alert | **5 h** |
| Alerts before a fault started | 0 |
| Batches wrongly diagnosed as contamination | **0 / 40** |

| Fault (batches) | Caught | Hours to alert | Leading explanation |
|---|---|---|---|
| Aeration rate (3) | 3 / 3 | 0 | Process-condition change (1), unresolved (2) |
| Substrate feed rate (3) | 3 / 3 | 5–6 | Unresolved (2), delayed/slow growth (1) |
| Coolant flow rate (2) | 2 / 2 | 10–11 | Process-condition change (2) |
| Base flow rate (2) | **0 / 2** | – | – |

**Reading it.**

- **Low false alarms across all three control styles, with fast detection.** Most faults
  were caught within hours of starting.
- **No equipment fault was mistaken for contamination.** The closest correct reading,
  "raw-material or process-condition change", led in 3 of 8. "Unresolved" in 4 of 8 is
  the safe answer: the rule book has no "actuator fault" explanation yet.
- **Both base-flow faults were missed.** The pH control loop compensates for the faulty
  base pump, so the culture barely changes. Catching this kind of fault needs
  monitoring of the controller outputs (pump and valve demands) against the recipe.
  That's a planned extension.

**Two flaws this dataset exposed, now fixed:**

1. **Comparing beyond the reference's coverage.** Two normal batches that ran longer than
   almost all reference batches (264 h and 290 h) were scored against references built
   from only one or two batches. That produced huge deviations and **two false
   "contamination" diagnoses**. The system now doesn't score hours covered by fewer than
   5 reference batches, and reports that the batch is beyond the reference instead. This
   also stopped those batches from inflating the calibrated limits, which made detection
   much faster (median 20.5 h → 5 h). The fix was found on this test set, so these
   IndPenSim numbers are no longer fully "unseen".
2. **Elapsed time counted from the first log entry, not the batch start.** That gave a
   one-hour offset that made three fault alerts look as if they came before the fault.
   Elapsed time now counts from each batch's recorded start time.

## 2. Erythromycin fermentation: 406 real industrial batches

**Data.** [Zenodo 14619074](https://zenodo.org/records/14619074) (CC BY 4.0) contains
hourly plant-historian data from an erythromycin plant in 2022, logged from about hour
30 of each batch. **There are no fault or contamination labels.** Column codes are
undocumented, so only clear signals were monitored: temperature, online and offline pH,
dissolved oxygen, agitation, OUR and CER. Final potency was **held out** and used only
as the outcome.

**Split.** Chronological: the earliest 243 batches formed the reference, and the later
163 were monitored.

| Measure | Result |
|---|---|
| Logger artefacts flagged by ingestion (−1 / 9999 sentinels, capped OUR/CER) | 2,016 readings, none deleted |
| Later batches flagged for review | **14 / 163 (9%)**, almost all on temperature |
| Flagged among low-potency batches (below the reference's 10th percentile) | **8 / 48 (17%)** |
| Flagged among the rest | 6 / 115 (5%) |
| Median final potency: flagged vs not flagged | 10,434 vs 10,848 |
| Is the association chance? | Fisher p = 0.02 (odds ratio 3.6); Mann-Whitney p = 0.001 |

**Reading it.**

- **On genuine plant data, the batches the system flags really did turn out worse.** This
  is our first real-data evidence that its flags point at meaningful deviations. It's
  modest: only 14 flagged batches.
- **Diagnoses stayed cautious.** 13 of 14 led with "process-condition change", none with
  contamination.
- **The limits came out very wide on real data** (robust z ≈ 53, CUSUM ≈ 2,700), so only
  gross deviations alert, and DO and pH never did. Calibration takes the single worst of
  243 reference batches, and real plant data always has a few outliers. The planned fix
  is to set limits from a high percentile (e.g. 99th) instead of the maximum, and to
  re-run both evaluations.
- **Potency drifted down over the year** (median 11,103 in the reference period vs
  10,844 later). A frozen reference must be reviewed and refreshed periodically, which
  the system already supports through versioning.

## What this changes for the project

- We can now say: **on 40 held-out industrial-scale simulated batches, 1 false alarm,
  8 of 10 equipment faults caught (median 5 h), and none mistaken for contamination.**
  On real plant data, **flagged batches had significantly lower final potency.**
- We still cannot say anything about **contamination** detection on real data. That
  needs MaverickX's batch records with confirmed contamination cases.
- Next model changes: percentile-based limits; monitoring controller outputs; an
  "equipment/actuator fault" explanation in the rule book.
