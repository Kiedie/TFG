#!/bin/bash

#SBATCH --job-name signals
#SBATCH --partition dios
#SBATCH -c 4
#SBATCH --gres=gpu:1
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/naguiler/Environments/tf-pt

python3 experimentacion3.py

mail -s "Proceso finalizado" juanjoha@correo.ugr.es <<< "El proceso ha finalizado"
