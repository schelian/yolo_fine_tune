#!/bin/bash
#SBATCH --job-name=FineTune
#SBATCH --partition=check
#SBATCH --time=00:40:00
#SBATCH --output=PyTestJob_%j.log
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --gres=gpu:1  

module purge > /dev/null 2>&1

conda activate yolo_fine_tune
python yolov9_fine_tune.py

echo "Job Complete"
