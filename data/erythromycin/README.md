# Erythromycin fermentation campaign (converted)

406 real industrial batches from an erythromycin plant (2022), converted for this
project by `python -m contamination.datasets.erythromycin prepare`.

Source: erythromycin fermentation process (EFP) data, Zenodo record 14619074
(https://zenodo.org/records/14619074), licensed CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Changes: kept 7 clearly identified
signals (temperature, online/offline pH, dissolved oxygen, agitation, OUR, CER),
reshaped to one row per reading, and held final potency out as the batch outcome.

There are no fault or contamination labels. The approved reference is in
reference_batches.txt (the earliest 60% of batches by start time).
