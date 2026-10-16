# Project context

Imported from the team's Notion workspace (page **Triple Ignition Challenge**, with its
*Research* and *System* databases) on 2026-10-16. Notion remains the source of truth for
research notes; this file captures what the code in this repository needs to know.

Challenge hub: <https://tksworld.notion.site/TKS-Triple-Ignition-Challenge-3b20b470b01080e69b62eb65570f89d7>

## The client: MaverickX

- Enzyme biotech startup (Y Combinator backed). Started in mining, pivoted to oil & gas for
  faster cash flow; mining remains a long-term goal. Public products: PetroX (oil & gas),
  LithX (mining), HydroX (water treatment).
- Small team of ~4–10 researchers: strong science, weaker on the business side.
- Risk-averse: prefers sequential focus over parallel workstreams; only onboards partners
  whose name it already knows.
- Main contact on the discovery call: Suman.

## The problem we chose

Five problems came out of the discovery call, ranked by severity:

1. **Contamination in enzyme production** (most severe; Suman's biggest day-to-day time sink).
   Batches get contaminated in a small facility with limited controls and must be remade. The
   source is often only found after several failed batches. Contaminated enzyme deployed in
   an oil tanker ruins the whole recovery process. Today it is spotted by smell, pigment and
   colour changes, and individual expertise.
2. Scaling from bench to factory / tech transfer to their custom bioreactor.
3. Slow, costly analytical testing (outsourced HPLC; results take days). The notes disagree
   on cost: "~$10,000 per test" in the team meeting vs "~$500 per 10 samples" in the
   discovery-call notes.
4. Inefficient R&D from too many experimental variables.
5. Partners want a clear financial ROI (treated as a symptom, not a root cause).

**Decision (meeting 11 Oct 2026):** focus only on #1. All earlier lunar / asteroid mining and
ilmenite prediction-mapping ideas are shelved, and so is the product-fit map / recipe generator
/ problem-type predictor idea for now (it addresses pre/post-production, not the core
bottleneck). This repo was created fresh so that the earlier lunar-mining context doesn't bias it.

## What we are building

A **hybrid contamination monitoring and diagnosis decision-support system**:

> analyses batch data to detect abnormalities, trace potential contamination sources and
> assess the evidence for batch impact, helping scientists investigate problems faster and
> more consistently.

It is a **detection + diagnosis** system, not a predictor of future contamination (that would
need historical data with confirmed cases). It supports scientists and never makes the
batch-release decision itself.

### Four-layer architecture (agreed 14 Oct 2026)

| Layer | Question | Outputs |
|---|---|---|
| 1. Anomaly detection | Is this batch behaving differently from its normal profile? | **Anomaly score** and **independent evidence of contamination**, kept as two separate outputs |
| 2. Source tracking | Where might it have entered? | Ranked candidate sources from sample timing, batch identity, process records and equipment history, or "source undetermined" |
| 3. Confidence & severity | How sure are we, and how much does it matter? | **Confidence** (reliability of the finding) and **severity** (effect on yield/purity/cost), never merged |
| 4. Recommendation | What should the scientist do next? | Suggested confirmatory test, "further investigation", or "uncertain"; multi-factor, no single variable decides |

Notes:

- One baseline does not fit all: each product/organism needs its own normal profile.
- Colour or smell changes alone are not reliable indicators. Combine several independent signals.
- Beren's review suggests regrouping into three modules: (1) data collection & reference
  profile, (2) detection & evaluation, (3) diagnosis & decision support.

The implementation guide expands this into six software layers: evidence acquisition → signal
processing → reference & case memory → anomaly detection → diagnostic reasoning → decision
support / reporting.

### MVP scope (version 1)

The core is Layer 1. Layers 2–4 get thin, transparent, rule-based versions.

| Layer | Version 1 scope |
|---|---|
| Anomaly detection | Build and evaluate the core model |
| Source identification | Record sample location and batch history; defer automated attribution (a batch-genealogy table/graph is the planned next step) |
| Severity & confidence | Report anomaly score and confidence separately; add production impact only when data supports it |
| Recommendations | Flag for review and suggest confirmation tests; release decisions stay with qualified staff |

**The MVP will:** standardise batch measurements and observations, compare a batch with the
right reference, flag unusual values *and* trajectories, show the evidence behind every flag,
and record scientist assessments and outcomes.

**The MVP will not:** claim to identify every contaminant, treat an anomaly score as a
probability of contamination, train a supervised classifier without reliable labels,
decide automatically to discard a batch, or claim "E. coli detected" from pH, temperature
or DO alone.

### Proxy dataset choice

There is no MaverickX data yet, so we start with **synthetic data**:

- **Target product:** α-amylase production (gives a measurable quality variable: enzyme activity).
- **Candidate contaminant:** *E. coli* as an example of bacterial contamination.
- Bacterial contamination is the primary concern and mixed microbial contamination the
  secondary one. Bacteriophage and yeast/mould are lower priority.
- Alternative raised on 14 Oct: **lipase** (relevant to oil & gas) with sulfate-reducing
  bacteria. Pick whichever enzyme/contaminant pairing has the richest public data
  (pH, temperature, oxygen). **Still to be confirmed with the team.**

This is a prototype design choice, not a claim about what MaverickX produces or has seen.

## Technical approach (from the MVP implementation guide)

### Data model (CSV first, SQLite later if needed)

Measurements table, one row per measurement:

| Field | Example | Purpose |
|---|---|---|
| `batch_id` | B001 | Link records to a batch |
| `timestamp` | 2026-10-01 10:00 | Sequence and timing |
| `stage` | Stage A | Compare like with like |
| `variable` | pH | What was measured |
| `value` | 7.2 | Numerical value |
| `unit` | — | Never mix units |
| `source` | instrument / manual | Trace the evidence |
| `quality_flag` | valid / missing / suspect | Keep bad data from silently affecting results |
| `observation` | free-text visual note | Keep qualitative evidence |
| `outcome_status` | unknown / reviewed / confirmed | How mature the evidence is |

Keep a **separate batch-outcome table** (one row per batch) so the answer can't leak into
the detector. Later tables: batch genealogy (batch ↔ input lots, equipment, process steps)
and an investigation case library.

Candidate variables: pH, temperature, dissolved oxygen, biomass, colour/pigment, enzyme
activity, yield, microbiological test results.

### Ingestion rules

- Flag or reject duplicates. Never fill missing values with zero. Keep missingness explicit.
- Keep the original value and unit. Never silently drop unusual values.
- Tell a real observation apart from an instrument error or an unverified note.
- Store when each measurement *became available*, so retrospective analysis can't use future data.

### Signal processing

- Compare batches within the same process stage. Don't time-warp batches to look alike.
- Standardise with z-scores, or robust z = 0.6745·(x − median)/MAD for small or outlier-heavy
  data, with a defined fallback when MAD = 0.
- Features: deviation from baseline, rate of change, rolling variability, time to milestone,
  cross-signal residual (two signals that stop moving together), missing-data indicator.

### Reference profiles

- Phase I: build the reference from scientist-approved normal batches and record which ones
  were included and why. Phase II: **freeze** it and monitor new batches against it. Update only
  through a documented review, so slow drift can't become the new normal.

### Detectors, in this order

1. Robust single-variable thresholds/deviations (interpretable baseline).
2. EWMA, E_t = λ·x_t + (1 − λ)·E_{t−1}, and CUSUM for gradual or persistent shifts.
3. Multivariate monitoring (Hotelling's T², robust Mahalanobis, PCA), only with enough comparable batches.
4. Isolation Forest, only as a research comparison.

No neural nets until we can explain what the simpler methods miss.

Each **alert** is an object, not a boolean: batch ID, timestamp, responsible variable or
pattern, observed vs reference value, deviation size, algorithm + reference version,
data-quality warnings, whether the signal is isolated, persistent or corroborated, and
whether contamination has been independently confirmed. Anomaly severity and diagnostic
confidence are separate fields.

### Diagnostic reasoning (rules first)

- A scientist-reviewed **hypothesis catalogue**: biological contamination, expected
  process variation, raw-material or condition change, measurement/equipment fault, unknown.
- An **evidence matrix** where each piece of evidence is *present / absent / contradictory / unknown*.
  Missing is not the same as negative.
- A rule-based decision table, with Bayesian networks or case-based retrieval later.
- Source tracing compares shared inputs, equipment, steps and timelines across abnormal and
  normal batches. A shared history is a lead, not proof.
- Severity is shown as several dimensions, not one weighted score: contamination evidence
  (unconfirmed/suspected/confirmed), process deviation, quality impact, affected scope,
  confidence, next step.

### Dashboard

Streamlit: batch overview, signal timeline with reference bands and flags, alert explanation,
investigation record. A **batch replay** feature steps through the timeline to show when the
first alert would have fired using only the data available at that point.

### Stack

Python · pandas · NumPy/SciPy · scikit-learn (optional comparison) · CSV → SQLite ·
Streamlit · Plotly/Matplotlib · NetworkX (genealogy graph) · pytest.

### Data sources

- **Now:** synthetic time series with normal noise, legitimate batch-to-batch variation,
  missing values, isolated spikes, gradual drifts, broken cross-signal relationships and
  documented labels.
- **For testing generic code:** UCR Time Series Anomaly Archive; NASA Prognostics Data Repository.
  These are not contamination data.
- **Goal:** de-identified MaverickX batch records with independently confirmed outcomes.
- Never merge unrelated public data with company data as if it were one biological training set.

### Validation

- Compare threshold vs EWMA (vs multivariate) on the same data.
- Split by **whole batch**, chronologically where possible. Fit preprocessing on reference data
  only. Freeze thresholds before testing. Use only data available at alert time.
- Metrics: sensitivity, specificity, precision, false alerts per batch, detection lead time,
  investigation time. Report uncertainty when confirmed cases are few. With rare events,
  accuracy is misleading.
- Agree success targets with MaverickX before testing, and don't invent them beforehand.
- Synthetic-data results show the software behaves as intended. They do not show real-world
  contamination detection.

## Team & timeline

- Team: Aisha Jawad, Beren Abedan, Ishan Singh, Trisha. Directors / reviewers: Pranav and Shea.
- **Aisha & Beren:** research, system architecture, validator outreach.
  **Ishan & Trisha:** prototype build (from Wed 14 Oct).
- Suggested workstreams: data & process mapping, anomaly detection, diagnostic reasoning,
  interface & validation. Agree the data schema and alert format before coding separately.
- **Deadline: 28 October 2026.** Target: first MVP build in about a week, then a week of testing.
- Sunday 19 Oct: team review of the prototype and feedback, outreach meetings begin.
- Deliverables: **prototype**, **one-pager** for MaverickX, **technical document** (including the science).
- Final Sunday: share with Pranav & Shea and incorporate feedback. Final Mon/Tue: polish.

### First build steps

1. One-page data request to MaverickX (records, measurements, confirmed outcomes, confidentiality).
2. Synthetic batch dataset + ingestion/validation pipeline.
3. Robust-threshold and EWMA detectors, with a chart of reference vs observed trajectory and alerts.

The first demo stays modest: load a batch, validate it, compare to the reference, flag a
deviation and explain why.

## Competitive positioning

- Main rival team (Alex, Octavian, Mohamed) has a strong biology/protein background and will
  likely pursue enzyme modification or contamination-resistant enzymes.
- Our edge: the technical, predictive and interdisciplinary layer, usable by non-scientists, with
  an easy UI and interpretable outputs. MaverickX said it needs non-scientist perspectives.

## Open questions for MaverickX

- Which production organism do they use? Are past contamination cases confirmed and
  identified, or suspected from smell, colour and behaviour?
- What do they already record (measurements, logs, timestamps, stages)?
- What counts as a confirmed outcome, and how is it established today?
- What is the real bottleneck: early detection, identification, telling contamination apart from
  normal variation, or choosing the next test?
- How much scientist time does one investigation take, and what is the main consequence
  (lost batches, delays, unreliable results)?
- What turnaround time would make a test useful, and what budget is acceptable?
- How does the biological R&D relate to the chemistry-focused products on their website?
