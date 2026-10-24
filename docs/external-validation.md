# External validation: IndPenSim and real erythromycin plant data

Until now every result came from our own synthetic data, where the simulator and the
rule book share the same assumptions. This is the first test on data we did not write.
**Neither dataset contains contamination**, so this tests false alarms, early detection
of genuine process deviations, and whether the system avoids calling a non-biological
problem "contamination". Contamination detection itself still needs MaverickX's
records.

Both datasets ship as converted campaign folders (`data/indpensim/`, `data/erythromycin/`,
CC BY 4.0, attribution in each folder's README) and can be explored in the dashboard.
Rebuild them from the original downloads (~510 MB, kept out of git) and re-run the
evaluations with:

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

| Measure | First run | With controller outputs | With the control-loop check (now) |
|---|---|---|---|
| Normal held-out batches flagged | 1 / 30 | 1 / 30 | **1 / 30 (95% range 1–17%)** |
| Fault batches flagged after the fault started | 8 / 10 | 8 / 10 | **10 / 10 (72–100%)** |
| Median time from fault onset to first alert | 5 h | 2 h | **1.5 h (0–5.5 h)** |
| Alerts before a fault started | 0 | 0 | 0 |
| Batches wrongly diagnosed as contamination | 0 / 40 | 0 / 40 | **0 / 40 (0–9%)** |
| Faults read as an equipment or actuator fault | (no such explanation) | 2 / 10 | **4 / 10** (all pump and cooling faults) |

| Fault (batches) | Caught | Hours to alert | Leading explanation |
|---|---|---|---|
| Aeration rate (3) | 3 / 3 | 0 | Process-condition change (1), unresolved (2) |
| Substrate feed rate (3) | 3 / 3 | 2 (was 5–6) | Unresolved (2), delayed/slow growth (1) |
| Coolant flow rate (2) | 2 / 2 | 9–10 | **Equipment or actuator fault (2)** |
| Base flow rate (2) | **2 / 2** (were missed) | **1** | **Equipment or actuator fault (2)** |

**Reading it.**

- **Watching the controllers helps.** The base pump's output changed 2 h after the
  substrate-feed faults began, 3–4 h before the culture signals did, so these faults are
  caught at 2 h instead of 5–6 h.
- **The control-loop check catches the base-pump faults 1 h after they start.** A
  healthy pH loop answers a falling pH by pumping more base. In all 90 normal batches, pH
  never moved far from normal while the base pump failed to respond (robust z at most
  9.3); in both base-flow faults it reached 10.6 one hour after onset and 28 by hour 4.
  The new detector watches exactly that pattern, so it needs no tighter limits and adds
  no false-alarm batches. The same pattern on temperature and cooling water flags the
  cooling faults. All four pump and cooling faults are now read as **equipment or
  actuator faults**.
- **Still no equipment fault mistaken for contamination.** One substrate-feed fault leads
  with "delayed / slow growth", which is the culture's response to less feed: right
  about the culture, but it does not name the pump.
- **A correction.** We first said the base-flow faults were missed because the pH loop
  compensates. In fact pH drops from 6.5 to below 6.0, but the pH limit was very wide
  (robust z ≈ 56) because a few normal batches have similar dips, *with* the controller
  responding. The control-loop check uses that difference.

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

These fixes, the controller-output rules, the control-loop check and the calibration
below were all developed while looking at this dataset, so the IndPenSim numbers are
**no longer unseen**.

## 2. Erythromycin fermentation: 406 real industrial batches

**Data.** [Zenodo 14619074](https://zenodo.org/records/14619074) (CC BY 4.0) contains
hourly plant-historian data from an erythromycin plant in 2022, logged from about hour
30 of each batch. **There are no fault or contamination labels.** Column codes are
undocumented, so only clear signals were monitored: temperature, online and offline pH,
dissolved oxygen, agitation, OUR and CER. Final potency was **held out** and used only
as the outcome.

**Split.** Chronological: the earliest 243 batches formed the reference, and the later
163 were monitored.

| Measure | Without the linked-signal detector | With it (current default) |
|---|---|---|
| Logger artefacts flagged by ingestion (−1 / 9999 sentinels, capped OUR/CER) | 2,016 readings, none deleted | same |
| Later batches flagged for review | **14 / 163 (9%)**, almost all on temperature | **26 / 163 (16%)** |
| Flagged among low-potency batches (below the reference's 10th percentile) | **8 / 48 (17%)** | 10 / 48 (21%) |
| Flagged among the rest | 6 / 115 (5%) | 16 / 115 (14%) |
| Median final potency: flagged vs not flagged | 10,434 vs 10,848 | 10,690 vs 10,850 |
| Is the association chance? | Fisher p = 0.02 (odds ratio 3.6); Mann-Whitney p = 0.001 | Fisher p = 0.19; Mann-Whitney p = 0.04 |
| Batches called contamination | 0 | 0 |

**Reading it.**

- **On genuine plant data, the batches the system flags turned out worse.** With the
  original detectors the association was clear (14 flagged batches). The linked-signal
  detector (section 5b), added to catch contamination earlier, also flags 12 batches whose
  respiration changed mid-run without hurting potency, so the association weakens
  (flagged batches still have lower potency, p = 0.04). On a plant, those are reviews
  that find a process change rather than a quality problem.
- **Diagnoses stayed cautious.** Flagged batches led with "process-condition change" or
  "unresolved"; none with contamination.
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

IndPenSim figures in the first two rows come from runs before the controller outputs were added; the others include them. All rows are before the control-loop check, which now catches the base-flow faults under the 5% budget (section 1).

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
- **The default is a 5% budget**, now split equally between the single-reading threshold
  and the linked-signal detector. It never did worse than the old default on any dataset,
  and it is a number MaverickX can set themselves ("how many normal batches in 100 can we
  afford to review?"). Catching the base-flow faults needs about 10%.

## 4. Benchmark: would a standard multivariate method do better?

Reviewers will ask. `python -m contamination.multivariate` fits PCA with Hotelling's T²
and SPE (the textbook multivariate monitoring method) on the same hourly z-scores,
calibrated by cross-validation under the same 5% false-alarm budget. Offline assays are
left out (PCA needs a complete row every hour), and our detectors are scored on the same
hourly signals. **Comparison only**: the project keeps the simple detectors. Our
figures here are from before the linked-signal detector (section 5b) was added.

| Dataset | Method | Normal batches flagged | Problem batches caught | Onset to alert |
|---|---|---|---|---|
| IndPenSim | Ours (same signals) | 1/30 (1–17%) | **10/10 (72–100%)** | 1.5 h (0–5.5 h) |
| IndPenSim | PCA T²/SPE | 0/30 (0–11%) | 7/10 (40–89%) | 2 h (1–2 h) |
| Synthetic, seeds 11–30 | Ours (pH, temperature, DO) | 75/260 (24–35%) | 120/120 contaminations | 7.7 h |
| Synthetic, seeds 11–30 | PCA T²/SPE | 69/260 (22–32%) | 120/120 contaminations | 7.7 h |

PCA missed both base-pump faults and one cooling fault, and on the synthetic data it
matched our detectors. It does not say which variable moved or why without a further
contribution analysis. So there is no case for the more complex method on this evidence.

## 5. Semi-synthetic contamination test: how big must a contaminant be?

Neither external dataset contains contamination, and on our own simulator a perfect score
partly reflects shared assumptions. `python -m contamination.datasets.contaminate` takes
the 30 held-out normal IndPenSim batches and adds the effects of a contaminant growing
from a random onset hour (logistic growth, half effect 12 h after onset): extra oxygen
uptake and CO₂, lower dissolved oxygen, extra biomass, extra acid (the base pump works
harder), extra heat (more cooling) and less penicillin. **The effects are our
assumptions**; the batch-to-batch variation, control strategies, sampling and 12-hourly
lab results with a 4 h delay are IndPenSim's. The reference is fitted on untouched
batches only. At strength 1 the contaminant raises CO₂ output and oxygen uptake by 20%
and lowers dissolved oxygen by 25%; other strengths scale these.

| Contaminant strength | Caught: original detectors | Caught: with linked-signal detector (default) | Onset to first alert (with) | Named as contamination (with) | Onset to named (with) |
|---|---|---|---|---|---|
| ½ (+10% respiration) | 1/30 (1–17%) | 1/30 (1–17%) | – | 0/30 | – |
| 1 (+20%) | 5/30 (7–34%), 49 h | **27/30 (74–97%)** | **19 h** | 4/30 | 34 h |
| 2 (+40%) | 28/30 (79–98%), 16 h | **30/30 (89–100%)** | **13 h** | **29/30** | **26 h** |
| 4 (+80%) | 30/30 (89–100%), 11 h | **30/30 (89–100%)** | **11 h** | **30/30** | **21.5 h** |

The unmodified copies of the same batches: 1/30 flagged, 0/30 named contamination, with
or without the linked-signal detector.

**Reading it.**

- **There is a detection threshold, and section 5b lowered it.** With the original
  detectors, a contaminant that changes respiration by 20% or less was mostly invisible on
  realistic industrial variation. Requiring linked signals to change together catches
  27 of 30 at +20%, within about 19 h, with no extra false alarms on these batches. Below
  about +10% it is still invisible. The 120/120 on our own simulator reflected a process
  with much less batch-to-batch variation, not a general capability.
- **At +20% the alert usually comes without a name.** The model raises the alert but
  leaves the explanation "unresolved" (4/30 named contamination), because the confirming
  evidence, extra biomass, comes from a lab assay every 12 h. That is the intended safe
  behaviour: the recommended checks include a contamination test.
- **When the evidence is there, the diagnosis is right.** At strengths 2 and 4,
  contamination became the leading explanation in 59 of 60 batches, using rules written
  for a different process (α-amylase), and no normal batch was called contamination.
- **The first sign is almost always CO₂ output (CER).** Off-gas analysis is the most
  valuable signal to ask MaverickX for, followed by more frequent biomass/OD readings.
- **What we tried that did not help:** a separate reference per control strategy caught
  about the same as a pooled reference with an equally loose false-alarm budget (12/30
  vs 11/30 at +20%, both with 4/30 false alarms), so it was not adopted. MaverickX
  products that differ a lot can be monitored as separate campaigns.

## 5b. The linked-signal detector

A contaminant changes several physically linked signals at once: more oxygen uptake,
more CO₂, lower dissolved oxygen. Ordinary batch-to-batch differences are mostly
*persistent offsets* (a batch runs a little high all the way through), and ordinary
mid-batch wobble usually moves *one* signal. The linked-shift detector therefore:

1. compares each hourly reading with the same batch's own recent past (the median of
   the 24 h ending 12 h earlier), so a persistent offset cancels;
2. averages that change over 6 h and scales it by its normal spread;
3. takes the **second-largest** change across the linked group, so it only rises when at
   least two linked signals change together.

The linked group is set per process: CO₂ output, oxygen uptake and dissolved oxygen for
IndPenSim (off-gas CO₂ was left out because it measures the same thing as CO₂ output),
the same three for the erythromycin plant, and dissolved oxygen and pH for the synthetic
α-amylase process. Its limit is calibrated leave-one-out like the others, and the
false-alarm budget is split equally between it and the single-reading threshold. It
reports which signals moved, and counts as "several signals deviate together" in the
diagnosis, never on its own as contamination.

| Dataset | Without | With (default) |
|---|---|---|
| Semi-synthetic contamination, +20% | 5/30 caught | **27/30 caught** |
| IndPenSim faults / normal batches flagged | 10/10, 1/30 | 10/10, 1/30 |
| Synthetic tuning seeds: normal-category flagged | 43/130 | 41/130 |
| Erythromycin plant: later batches flagged | 14/163 (9%) | 26/163 (16%) |

The cost is on the real plant: more reviews of batches whose respiration genuinely
changed without a quality problem. Developed while looking at the semi-synthetic test,
so its numbers there are not unseen; the erythromycin and synthetic figures were not
used to design it.

## 6. Optional: comparing each batch with its own past

A natural fix for batch-to-batch offsets is to compare each reading with the same batch's
recent past, so a persistent offset cancels and a change partway through stands out.
This single-signal "shift" detector is implemented but **off by default** (the
linked-signal detector in section 5b is the version that is on). Figures below are
without the linked-signal detector:

| Dataset | Without shift | With shift |
|---|---|---|
| Semi-synthetic contamination, strength 1 (first run) | 6/30 caught, 105 h after onset | 12/30 caught, 34 h after onset |
| IndPenSim faults and normal batches | 10/10, 1/30 | 10/10, 1/30 |
| Erythromycin plant: later batches flagged | **14/163 (9%)** | **63/163 (39%)** |
| Synthetic tuning seeds: normal-category flagged | 43/130 | 47/130 |

Real plant batches change mid-run for ordinary reasons, so on the erythromycin plant it
would bury scientists in reviews. It can be switched on per process
(`MonitorConfig.detectors`) once MaverickX's own false-alarm rate with it is known.

## 7. Reference refresh

`python -m contamination.refresh data/erythromycin` proposes monitored batches that could
join the approved reference (no alerts, no problem outcome), shows how "normal" and the
limits would move, and with `--apply --note "..."` writes the new list and a line in
`reference_history.csv`. Nothing is approved automatically. On the erythromycin plant,
149 of the 163 later batches qualify (the other 14 had alerts to review first); the
reference trajectories would barely move (median shift 0–0.2 robust SD), while the
single-reading limits would tighten (e.g. pH 9.1 → 7.0) because more batches define
normal. The potency drift noted above is not visible in the monitored signals.

## What this changes for the project

- We can now say: **on 40 industrial-scale simulated batches, 1 false alarm, all 10
  equipment faults caught (median 1.5 h), none mistaken for contamination, and all four
  pump and cooling faults correctly read as equipment faults.** On real plant data,
  **flagged batches had significantly lower final potency.** A standard multivariate
  method (PCA) did no better.
- We still cannot say anything about **contamination** detection on real data. That
  needs MaverickX's batch records with confirmed contamination cases. On realistic
  simulated variation, a contaminant that changes respiration by about 20% is caught in
  27 of 30 batches within about 19 h, and one at 10% is missed (sections 5 and 5b).
- Done since the first run: limits set by a false-alarm budget; controller outputs
  monitored; a control-loop check; a linked-signal detector; an "equipment or actuator
  fault" explanation in the rule book; a PCA benchmark; the semi-synthetic contamination
  test; 95% ranges on every count; a reference refresh workflow.
- The data request now asks MaverickX for controller outputs (pump and valve demands) and
  for their acceptable false-alarm rate.
