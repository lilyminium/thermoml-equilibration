# Initial data curation

This directory contains scripts for initial downloads and filtering of ThermoML. The final output of this is `output/broad-dataset.json`.


## Scripts

Scripts are quickly documented below in the order they were run. Log files are in `logs/`.

* download-initial-dataset.py: downloads the initial datasets and filters out any properties where SMILES are disconnected (contains a period). Creates `input/thermoml.csv` and `input/initial-thermoml.csv`.
* filter-data-initial.py: Some initial filters pulled from the Sage 2.0 process, but more permissive. Results in `intermediate/initial-filtered-thermoml.csv`.
* filter-data-continued.py: More filtering. Gives `intermediate/continued-filtered-thermoml.csv`
* blacklist-high-viscosities.py: Filters out any properties containing a component with a viscosity over 0.3 Pa * s. Yields `intermediate/continued-filtered-without-high-viscosities.csv`
* filter-data-broad.py: An initial broad filter to select for pure and 50/50 mixed properties so we can do an initial sweep and look for slowly-equilibrating properties. Gives `output/broad-dataset.[json|csv]`.
