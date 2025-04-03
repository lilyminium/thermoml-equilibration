#!/bin/bash

# python renumber-properties.py > logs/renumber-properties.log
python filter-data-training.py -np 4 > logs/filter-data-training.log
python filter-data-validation.py -np 4 > logs/filter-data-validation.log
# python profile-dataset-chemical-groups.py -i output/training-set.csv > logs/profile-dataset-chemical-groups.log


python combine-datasets.py output/training-set.json output/validation-set.json > logs/combine-datasets.log