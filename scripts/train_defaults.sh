#!/bin/bash
#SBATCH --qos=blanca-ccn
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=16

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /pl/active/ccnlab/conda/etc/profile.d/conda.sh
conda activate pytorch_mpi

export MKL_NUM_THREADS=16 OMP_NUM_THREADS=16

echo "MKL_NUM_THREADS: "
echo $MKL_NUM_THREADS
echo "OMP_NUM_THREADS: "
echo $OMP_NUM_THREADS

python train_test.py \
--results_dir train_results \
--out_data_file train_defaults.json \
--checkpoint_path ../model_weights/defaults.pt
