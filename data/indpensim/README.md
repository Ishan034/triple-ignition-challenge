# IndPenSim campaign (converted)

100 simulated batches of a 100,000 L fed-batch penicillin process, converted for this
project by `python -m contamination.datasets.indpensim prepare`.

Source: Goldrick, S. et al. (2015) J. Biotechnol. 193:70-82; data: IndPenSim,
Mendeley Data, doi:10.17632/pdnjz7zz5x.2, licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Changes: kept hourly online values and
the offline assays (with a 4 h result delay), temperature converted to degC, a subset of
columns, synthetic start dates, and batch outcomes derived from the fault reference.

Batches 91-100 contain equipment faults (not contamination). The approved reference is
in reference_batches.txt (the first 20 normal batches of each control strategy).
