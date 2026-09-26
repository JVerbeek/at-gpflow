#!/usr/bin/env bash
#SBATCH --partition=das
#SBATCH --account=das
#SBATCH --qos=das-normal
#SBATCH --account=das
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=logs/speedtest-variable-%j.out
#SBATCH --error=logs/speedtest-variable-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# hyperparameters
. $HOME/miniconda3/etc/profile.d/conda.sh

conda activate gp

python speedtest-script.py $1

