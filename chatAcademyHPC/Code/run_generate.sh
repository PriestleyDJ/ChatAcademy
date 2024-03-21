#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1 
#SBATCH --gres=gpu:4
#SBATCH --mem=334GB  
#SBATCH --mail-user=INSERT EMAIL@sheffield.ac.uk
#SBATCH --mail-type=ALL

module load Anaconda3/2022.05
module load CUDA/11.8.0 

source activate chatAcademy

python chatacademyimplementation.py $1
