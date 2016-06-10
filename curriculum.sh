#!/bin/bash
#SBATCH -A transferlearning
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH -p plgrid-gpu
#SBATCH --gres=gpu

module add plgrid/tools/python/2.7.9
module add plgrid/apps/cuda/7.0

cd $PLG_GROUP_STORAGE/plggaghcvg/TransferRL
python trainer.py --model_name $1 --world_path $2 --curriculum_name $3
