#!/usr/bin/env bash
#SBATCH -J eq-100
#SBATCH -p cpu
#SBATCH -t 7-00:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb
#SBATCH --output slurm-%x.%A.out

. ~/.bashrc

# Use the right conda environment
conda activate evaluator-test-env-openff

python equilibrate-k8s.py --dataset-path ../01_curate-data/output/broad-dataset.json
