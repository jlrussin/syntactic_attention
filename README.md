# Syntactic Attention

This repository contains the code associated with [this paper](https://arxiv.org/abs/1904.09708)

Title: Compositional generalization in a deep seq2seq model by separating syntax and semantics

Abstract: Standard methods in deep learning for natural language processing fail to capture the compositional structure of human language that allows for systematic generalization outside of the training distribution. However, human learners readily generalize in this way, e.g. by applying known grammatical rules to novel words. Inspired by work in neuroscience suggesting separate brain systems for syntactic and semantic processing, we implement a modification to standard approaches in neural machine translation, imposing an analogous separation. The novel model, which we call Syntactic Attention, substantially outperforms standard methods in deep learning on the SCAN dataset, a compositional generalization task, without any hand-engineered features or additional supervision. Our work suggests that separating syntactic from semantic learning may be a useful heuristic for capturing compositional structure.

## Summary

This directory contains the following:
- data: directory containing the add-jump split of the SCAN dataset
- data.py: classes for building PyTorch dataset and dataloader with SCAN
- SyntacticAttention.py: Syntactic Attention model
- train_test.py: Training script with periodic validation and testing
- results: contains results of one run of Syntactic Attention on add-jump split
- utils.py: contains helper function
- vocab.json: contains vocabulary for SCAN

### Requirements
- Python 3 ([here](https://scipy.org/install.html))
- Install NumPy ([here](https://scipy.org/install.html))
- Install PyTorch ([pytorch.org](https://pytorch.org/))
```bash
conda install pytorch torchvision -c pytorch
```
- The add-jump dataset is already in the `data` directory, but the rest of the dataset can be downloaded [here](https://github.com/brendenlake/SCAN)

## Training and testing
To train and test the Syntactic Attention model on the add-jump split with the default hyperparameters:

```bash
python train_test.py
```
This will train the model and produce a json file in the results directory with loss data, along with periodic tests on the full training, validation, and test sets. The default hyperparameters and descriptions can be found in `train_test.py`.

## Usage
```
usage: train_test.py [--train_data_file data/tasks_train_addprim_jump.txt]
                     [--test_data_file data/tasks_test_addprim_jump.txt]
                     [--load_vocab_json vocab.json][--batch_size 1]
                     [--num_iters 200000][--rnn_type LSTM]
                     [--m_hidden_dim 120][--x_hidden_dim 200][--n_layers 1]
                     [--dropout_p 0.5]
                     [--seq_sem False][--syn_act False][--sem_mlp False]
                     [--load_weights_from CHECKPOINT.pt][--learning_rate 0.001]
                     [--clip_norm 5.0]
                     [--results_dir results][--out_data_file RESULTS.json]
                     [--checkpoint_path CHECKPOINT.pt][--checkpoint_every 5]
                     [--record_loss_every 400]
```
