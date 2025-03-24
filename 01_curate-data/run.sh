#!/bin/bash


# python download-initial-dataset.py -np 4 > logs/download-initial-dataset.log
# python filter-data-initial.py -np 4 > logs/filter-data-initial.log
# python filter-data-continued.py -np 4 > logs/filter-data-continued.log

# python blacklist-high-viscosities.py > logs/blacklist-high-viscosities.log

# python profile-dataset-chemical-groups.py > logs/profile-dataset-chemical-groups.log

# python filter-data-broad.py -np 4 > logs/filter-data-broad.log

python filter-for-rerun.py -np 4 > logs/filter-for-rerun.log