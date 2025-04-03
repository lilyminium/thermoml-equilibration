# Final filter

This directory contains the final filtering step to extract the training, validation, and combined datasets in `output/`.

## Scripts

The actual run file can be found in `run.sh` and logs in `logs/`.

* renumber-properties.py: Properties were renumbered from `../01_curate-data/intermediate/continued-filtered-without-high-viscosities.csv` to `input/renumbered-dataset.csv` because the IDs were hard to parse and remember.
* filter-data-training.py: Does a final filter using, among others, the `exclude-molecules.smi` list in the previous step. Yields `output/training-set.[csv|json]`
* filter-data-validation.py: Does a final filter using, among others, the `exclude-molecules.smi` list in the previous step. Yields `output/validation-set.[csv|json]`
* profile-dataset-chemical-groups.py: Profiles the chemical groups of the training set with checkmol, results are in `chemical-groups/`.
* combine-datasets.py: combines the training and validation datasets, results in `output/combined-set.[json|csv]`.
