#!/usr/bin/env bash
#SBATCH -p localBackground
#SBATCH -A ecortex
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH -c 4

export HOME=`getent passwd $USER | cut -d':' -f6`
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1.0

python evaluate_bleu.py \
--dataset MT \
--flip True \
--test_data_file data/MT/train.daxy \
--load_vocab_json vocab_eng_MT_daxiste.json \
--load_weights_from ../model_weights/defaults_fra_MT_lr0p0001.pt
