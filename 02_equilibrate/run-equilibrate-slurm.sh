#!/usr/bin/env bash
#SBATCH -J eq-100
#SBATCH -p standard
#SBATCH -t 5-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --account dmobley_lab
#SBATCH --output slurm-%x.%A.out

. ~/.bashrc

# Use the right conda environment
conda activate evaluator-test-env-openff

python equilibrate-slurm.py -s 10 -d ../01_curate-data/output/dataset.json -p 8108 -n 1000
