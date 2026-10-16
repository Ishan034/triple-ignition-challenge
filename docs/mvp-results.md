# MVP results: detection and diagnosis on synthetic data

> **Synthetic data only.** These numbers show that the software behaves as intended on
> simulated α-amylase batches. They are **not** real-world contamination-detection
> performance. The process model, the simulated contaminant and every rule and weight
> in the knowledge base are student-team drafts that MaverickX scientists have not reviewed.

Reproduce with:

```bash
python -m contamination.demo                      # one dataset (seed 7), reports + figures
python -m contamination.robustness --seeds 11 31  # 20 fresh datasets
```

## What was evaluated

Each synthetic dataset has 30 reference batches (all normal, the earliest chronologically)
and 25 later test batches: 10 normal, 3 slow growth (legitimate variation), 6 contaminated,
3 with a faulty pH reading and 3 with a drifting DO probe. About 20% of all batches also have
a gap of missing readings. The reference and detector limits are fitted on the reference
batches only. The outcomes table is used only for evaluation.

Thresholds and knowledge-base weights were adjusted while looking at seeds 1–10. The
results below come from **seeds 11–30**, which were not used for tuning.

## Layer 4: detection (seeds 11–30, 500 test batches)

| Measure | Result |
|---|---|
| Contaminated batches flagged after onset | 120 / 120 |
| Non-contamination anomalies flagged (sensor spike, probe drift) | 120 / 120 |
| "Normal"-category batches flagged (includes the 60 slow-growth batches) | 81 / 260 |
| Median time from onset to first alert | ~7.8 h |
| Median lead time vs the simulated visual/odour note | 7.5 h (per-seed medians 5–9 h) |
| False alerts per test batch | 1.9 |

The simple robust threshold gives the earliest alerts. EWMA and CUSUM add little on these
simulated signals. CUSUM's calibrated limits are very high because each batch has a small
constant offset that CUSUM keeps accumulating.

## Layer 5: diagnosis (process evidence only, seeds 11–30)

The leading explanation is compared with the simulated scenario at several review times,
anchored on the first alert at or after the simulated onset. "Unresolved" means no
explanation was strong enough to lead, so the scientist is asked to investigate broadly
(the safe failure). "Wrong" means the report pointed the investigation the wrong way.

| Review time | Scenario | Batches | Matched | Unresolved | Wrong |
|---|---|---|---|---|---|
| at first alert | contamination | 120 | 12 | 103 | 5 |
| | probe drift | 60 | 0 | 59 | 1 |
| | sensor spike | 60 | 2 | 57 | 1 |
| | slow growth | 60 | 0 | 60 | 0 |
| first alert + 4 h | contamination | 120 | 82 | 35 | 3 |
| | probe drift | 60 | 0 | 57 | 3 |
| | sensor spike | 60 | 60 | 0 | 0 |
| | slow growth | 60 | 0 | 58 | 2 |
| first alert + 8 h | contamination | 120 | 114 | 4 | 2 |
| | probe drift | 60 | 56 | 1 | 3 |
| | sensor spike | 60 | 60 | 0 | 0 |
| | slow growth | 60 | 11 | 48 | 1 |
| end of batch | contamination | 120 | 120 | 0 | 0 |
| | probe drift | 60 | 58 | 1 | 1 |
| | sensor spike | 60 | 60 | 0 | 0 |
| | slow growth | 60 | 58 | 2 | 0 |

**How to read it:**

- **At the first alert the system mostly says "unresolved, too early to tell".** That's
  the intended behaviour: one signal moving on its own fits contamination, a probe fault
  and normal variation alike. The recommended checks at that point are chosen to tell
  those apart, and a contamination check is always included while contamination isn't
  contradicted (the safety rule).
- **Contamination becomes the leading explanation within about 4–8 h** of the first
  post-onset alert, and in every contaminated batch by end of batch.
- **A sensor spike is recognised once the next readings return to normal.** A probe
  drift needs about 6 h before it counts as "decoupled" from the signals it should move with.
- **Slow growth needs most of the growth phase** before the delayed-trajectory fit is
  reliable, so it stays unresolved for longer.
- **The remaining wrong cases are mostly early alerts explained by other real evidence**
  in the batch, for example a small genuine temperature offset (read as a process-condition
  change) or noise just after onset (read as a measurement glitch).

## Source tracing (seeds 11–30)

Each synthetic campaign plants one common source: 4 of the 6 contaminated test batches
use medium lot `M-12`. The other 2 contaminations are independent. Batch statuses come
from the end-of-batch diagnoses (process evidence only), and reference batches count as
approved normal. Each medium lot, inoculum lot and bioreactor is compared between users
and non-users (one-sided Fisher's exact test, Bonferroni-corrected), and then checked for
confounding with any stronger lead.

| Measure | Result |
|---|---|
| Planted lot ranked as the top **strong lead** | 20 / 20 seeds |
| Planted cases linked to the planted lot | 80 / 80 |
| Seeds with a false strong lead | 1 / 20 (inoculum lot I-11 in seed 23) |
| Independent cases left without a strong lead | 38 / 40 |
| "Possible leads" per campaign (shared by ≥ 2 cases, relative risk ≥ 2, not significant) | 2.3 on average |

The false strong lead in seed 23 is a genuine chance cluster: the two independent cases
ran inside the same 5-batch inoculum lot. The data can't tell that apart from a real
common source, so it's reasonable for the system to raise it as a lead. The possible leads
are mostly the same kind of chance co-occurrence among consecutive batches. That's why
the report calls them leads for investigation, not proof.

Two rules came out of development:

- **Over-representation, not just sharing.** Both bioreactors are always shared by several
  cases, but cases are no more common there than elsewhere (relative risk ≈ 1). They're
  labelled "not over-represented" rather than "possible lead".
- **Confounding check.** An inoculum lot overlapping the planted medium-lot batches looked
  significant only because of that overlap (in 3 of 20 seeds). Further strong leads must
  stay significant once batches that used a stronger lead are excluded. Otherwise they're
  downgraded, with a note explaining the overlap.

Tracing fills the report's **affected scope** (linked cases and the resources linking
them, plus any other batches that used a strong-lead resource and need review). It also
adds a **Possible sources** section with a suggested check per resource type.

## Design decisions made while building the diagnosis layer

These came from failures found during development. Each is a parameter or rule in
`src/contamination/knowledge/catalogue.yaml`, so it can be reviewed.

1. **Decoupling needs time** (`decoupled_min_duration_h: 6`). Without this, five of six
   contaminated batches were first diagnosed as a probe fault.
2. **A single weak item can't lead** (`leading_min_score: 2`).
3. **A leading explanation needs specific evidence** (`leading_requires_any`). Without
   this, slow-growth batches were sometimes led by "contamination" on non-specific
   evidence ("several signals deviate", "low activity").
4. **Safety rule** (`always_consider: H01`). While contamination isn't contradicted, the
   recommended checks include one that is informative about it.
5. **Delay is fitted on the growth phase only**, and activity is compared with the
   time-shifted reference when the batch is delayed. That way late product formation isn't
   mistaken for low activity.
6. **Only independent tests can confirm.** Contamination evidence reads "unconfirmed"
   until a contamination test (e.g. microscopy, plating, qPCR) is recorded. An instrument
   check result counts as evidence about a measurement fault, not about contamination.

## Known limitations

- The contaminant's effects (extra biomass, oxygen demand, acidification, lower activity)
  are assumptions written into the simulator. With real data the evidence links may need
  to change, and some may not hold at all.
- Detector limits are set from 30 reference batches. Fewer batches give looser references
  and more unresolved diagnoses (seen in the 20-batch test fixture).
- Source tracing only knows the genealogy recorded in the batches table (medium lot,
  inoculum lot, bioreactor). Water, transfer lines, operators and environmental samples
  aren't modelled. With a small campaign, chance clusters of consecutive batches can't be
  told apart from a real common source.
- Statuses for tracing come from end-of-batch diagnoses (a retrospective campaign review).
  A live view would need statuses built with `as_of_h`.
- None of the check turnaround times or costs are real MaverickX figures.
