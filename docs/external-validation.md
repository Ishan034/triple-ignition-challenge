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
equipment faults. We monitored the process outputs (pH, temperature, dissolved oxygen,
OUR, CER, off-gas CO₂, substrate, and offline biomass and penicillin, 12-hourly with a
4 h result delay) and the outputs of the pH and temperature controllers (base flow,
cooling-water flow), which a plant historian records. The recipe inputs that the
aeration and substrate-feed faults act on are not monitored, because watching the
changed input would make detection trivial. The base-flow and coolant faults act on the
controller outputs themselves; for those, what is tested is mainly the diagnosis.

**Split.** The reference was built from 20 normal batches of each control strategy (60).
We tested on the other 30 normal batches and all 10 fault batches.

| Measure | Before (Oct 2026 first run) | Now |
|---|---|---|
| Normal held-out batches flagged | 1 / 30 | **1 / 30** (one isolated temperature reading) |
| Fault batches flagged after the fault started | 8 / 10 | **8 / 10** |
| Median time from fault onset to first alert | 5 h | **2 h** |
| Alerts before a fault started | 0 | 0 |
| Batches wrongly diagnosed as contamination | 0 / 40 | **0 / 40** |
| Faults read as an equipment or actuator fault | (no such explanation) | **2 / 10** (both cooling faults) |

| Fault (batches) | Caught | Hours to alert | Leading explanation |
|---|---|---|---|
| Aeration rate (3) | 3 / 3 | 0 | Process-condition change (1), unresolved (2) |
| Substrate feed rate (3) | 3 / 3 | 2 (was 5–6) | Unresolved (2), delayed/slow growth (1) |
| Coolant flow rate (2) | 2 / 2 | 10–11 | **Equipment or actuator fault (2)** |
| Base flow rate (2) | **0 / 2** | – | – |

**Reading it.**

- **Watching the controllers helps.** The base pump's output changed 2 h after the
  substrate-feed faults began, 3–4 h before the culture signals did, so these faults are
  now caught at 2 h instead of 5–6 h. For the cooling faults, temperature rose while the
  cooling-water flow did not respond: the rule book now reads that pattern as an
  **equipment or actuator fault**, the correct explanation, instead of a
  process-condition change.
- **Still no equipment fault mistaken for contamination.** One substrate-feed fault leads
  with "delayed / slow growth", which is the culture's response to less feed: right
  about the culture, but it does not name the pump.
- **Both base-flow faults are still missed, and the earlier explanation was wrong.** We
  said the pH loop compensates so the culture barely changes. In fact pH drops from 6.5
  to below 6.0 during the fault, but the pH limit was very wide (robust z ≈ 56) because
  a few normal reference batches have similar dips. Tightening the limit catches them
  (see the calibration trade-off below), at the cost of more false alarms.

**Two flaws this dataset exposed earlier, now fixed:**

1. **Comparing beyond the reference's coverage.** Two normal batches that ran longer than
   almost all reference batches (264 h and 290 h) were scored against references built
   from only one or two batches. That produced huge deviations and **two false
   "contamination" diagnoses**. The system now doesn't score hours covered by fewer than
   5 reference batches, and reports that the batch is beyond the reference instead. This
   also stopped those batches from inflating the calibrated limits, which made detection
   much faster (median 20.5 h → 5 h).
2. **Elapsed time counted from the first log entry, not the batch start.** That gave a
   one-hour offset that made three fault alerts look as if they came before the fault.
   Elapsed time now counts from each batch's recorded start time.

These fixes, the controller-output rules and the calibration below were all developed
while looking at this dataset, so the IndPenSim numbers are **no longer unseen**.

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
  243 reference batches, and real plant data always has a few outliers. Tightening them
  is a trade-off, shown in the next section; under the default 5% false-alarm budget
  this dataset's results are unchanged.
- **Potency drifted down over the year** (median 11,103 in the reference period vs
  10,844 later). A frozen reference must be reviewed and refreshed periodically, which
  the system already supports through versioning.

## 3. Calibration trade-off: how tight should the limits be?

The limits used to sit just above the **single worst** approved reference batch. We
tested tighter limits on all three datasets. "Spike limit" is the single-reading
(robust threshold) limit; trend detectors (EWMA, CUSUM) were kept at the worst batch
except in the first row.

| Setting | IndPenSim: normal flagged | IndPenSim: faults caught | IndPenSim: base-flow faults | Erythromycin: later batches flagged | Synthetic (tuning seeds): normal-category flagged |
|---|---|---|---|---|---|
| All limits at 95th percentile | 6 / 30 | 10 / 10 | 2 / 2 | – | 54 / 130 |
| Worst batch (old default) | 1 / 30 | 8 / 10 | 0 / 2 | 14 / 163 | 41 / 130 |
| Spike limit at 95th percentile | 4 / 30 | 10 / 10 | 2 / 2 (at 4 h) | 97 / 163 (60%) | 45 / 130 |
| **False-alarm budget 5% (new default)** | **1 / 30** | **8 / 10** | 0 / 2 | **14 / 163** | 43 / 130 |
| False-alarm budget 10% | 3 / 30 | 10 / 10 | 2 / 2 (only at 76 h) | 54 / 163 (33%) | 44 / 130 |

IndPenSim figures in the first two rows come from runs before the controller outputs were added; the others include them.

**Reading it.**

- **A fixed percentile behaves very differently by dataset.** The same "95th percentile"
  setting that catches every IndPenSim fault flags 60% of the erythromycin plant's later
  batches: far too many reviews for a real plant.
- **A false-alarm budget is predictable on stable processes.** The limit is tightened only
  while at most a set share of the approved batches would have alerted (on any variable).
  On IndPenSim the held-out rate matched the budget (1/30 at 5%, 3/30 at 10%).
- **It is not predictable when the process drifts.** On the erythromycin plant the
  later batches drifted from the earlier ones (potency fell over the year), so a 10%
  budget gave 33% flagged. A frozen reference must be reviewed and refreshed.
- **The default is a 5% budget.** It never did worse than the old default on any dataset,
  and it is a number MaverickX can set themselves ("how many normal batches in 100 can we
  afford to review?"). Catching the base-flow faults needs about 10%.

## What this changes for the project

- We can now say: **on 40 industrial-scale simulated batches, 1 false alarm, 8 of 10
  equipment faults caught (median 2 h), none mistaken for contamination, and both
  cooling faults correctly read as equipment faults.** On real plant data, **flagged
  batches had significantly lower final potency.**
- We still cannot say anything about **contamination** detection on real data. That
  needs MaverickX's batch records with confirmed contamination cases.
- Done since the first run: limits set by a false-alarm budget; controller outputs
  monitored; an "equipment or actuator fault" explanation in the rule book.
- The data request now asks MaverickX for controller outputs (pump and valve demands) and
  for their acceptable false-alarm rate.
