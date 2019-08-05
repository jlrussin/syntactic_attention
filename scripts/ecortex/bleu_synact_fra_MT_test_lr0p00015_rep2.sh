#!/usr/bin/env bash
#SBATCH -p localBackground
#SBATCH -A ecortex
#SBATCH --mem=10G
#SBATCH --time=6:00:00
#SBATCH -c 4

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.0

python evaluate_bleu.py \
--dataset MT \
--flip False \
--train_data_file data/MT/train \
--val_data_file data/MT/test_no_unk.txt \
--test_data_file data/MT/test \
--load_vocab_json vocab_fra_MT_all.json \
--syn_act True \
--load_weights_from ../model_weights/synact_fra_MT_test_lr0p00015_rep2.pt
