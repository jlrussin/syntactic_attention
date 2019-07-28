#!/usr/bin/env bash
#SBATCH -p localLimited
#SBATCH -A ecortex
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 4

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.0

python train_test.py \
--dataset MT \
--flip False \
--train_data_file data/MT/train.daxy \
--test_data_file data/MT/test.daxy \
--load_vocab_json vocab_fra_MT_daxiste.json \
--num_iters 1000000 \
--syn_act True \
--learning_rate 0.0001 \
--results_dir train_results \
--out_data_file train_synact_fra_MT_lr0p0001.json \
--checkpoint_path ../model_weights/synact_fra_MT_lr0p0001.pt
