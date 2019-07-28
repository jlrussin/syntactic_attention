import os
import argparse
import json
import numpy as np
from nltk.translate import bleu_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import ScanDataset,MTDataset,SCAN_collate
from SyntacticAttention import *
from utils import *

parser = argparse.ArgumentParser()
# Data
parser.add_argument('--dataset', choices=['MT'],
                    default='MT',
                    help='MT only')
parser.add_argument('--flip', type=str2bool, default=False,
                    help='Flip source and target for MT dataset')
parser.add_argument('--test_data_file',
                    default='data/MT/train.daxy',
                    help='Path to test set')
parser.add_argument('--load_vocab_json',default='vocab_fra_MT_daxiste.json',
                    help='Path to vocab json file')

# Model hyperparameters
parser.add_argument('--rnn_type', choices=['GRU', 'LSTM'],
                    default='LSTM', help='Type of rnn to use.')
parser.add_argument('--m_hidden_dim', type=int, default=120,
                    help='Number of hidden units in semantic embeddings')
parser.add_argument('--x_hidden_dim', type=int, default=200,
                    help='Number of hidden units in syntax rnn')
parser.add_argument('--n_layers', type=int, default=1,
                    help='Number of layers in RNNs')
parser.add_argument('--dropout_p', type=float, default=0.5,
                    help='Dropout rate')
parser.add_argument('--seq_sem', type=str2bool, default=False,
                    help='Semantic embeddings also processed with RNN.')
parser.add_argument('--syn_act', type=str2bool, default=False,
                    help='Syntactic information also used for action')
parser.add_argument('--sem_mlp', type=str2bool, default=False,
                    help='Nonlinear semantic layer with ReLU')
parser.add_argument('--load_weights_from', default=None,
                    help='Path to saved weights')

# Output options
#parser.add_argument('--results_dir', default='results',
#                    help='Results subdirectory to save results')
#parser.add_argument('--out_data_file', default='results.json',
#                    help='Name of output data file')

def main(args):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Vocab
    with open(args.load_vocab_json,'r') as f:
        vocab = json.load(f)
    out_idx_to_token = vocab['out_idx_to_token']

    # Dataset
    batch_size = 1
    test_data = MTDataset(args.test_data_file,vocab,args.flip)
    test_loader = DataLoader(test_data,batch_size,
                             shuffle=True,collate_fn=SCAN_collate)
    in_vocab_size = len(vocab['in_token_to_idx'])
    out_vocab_size = len(vocab['out_idx_to_token'])
    max_tar_len = max([len(b[3]) for b in test_data])

    # Reference dictionary
    reference_dict = {}
    for sample in test_data:
        src = sample[2]
        tar = sample[3]
        assert src[0] == '<SOS>'
        assert src[-1] == '<EOS>'
        assert tar[0] == '<SOS>'
        assert tar[-1] == '<EOS>'
        src = src[1:-1]
        tar = tar[1:-1]
        key = '_'.join(src)
        if key not in reference_dict:
            reference_dict[key] = [tar]
        else:
            reference_dict[key].append(tar)

    # Model
    model = Seq2SeqSynAttn(in_vocab_size, args.m_hidden_dim, args.x_hidden_dim,
                           out_vocab_size, args.rnn_type, args.n_layers,
                           args.dropout_p, args.seq_sem, args.syn_act,
                           args.sem_mlp, max_tar_len, device)

    if args.load_weights_from is not None:
        model.load_state_dict(torch.load(args.load_weights_from))
    model.to(device)
    model.eval()

    # Evaluation loop:
    hypotheses = []
    references_list = []
    with torch.no_grad():
        for sample_count,sample in enumerate(test_loader):
            # Forward pass
            instructions, true_actions, _, _ = sample
            if len(true_actions) < 6:
                continue # Don't include if less than 4 words (without SOS, EOS)
            instructions = [ins.to(device) for ins in instructions]
            true_actions = [ta.to(device) for ta in true_actions]
            actions,padded_true_actions = model(instructions,true_actions)
            # Get hypothesis
            max_actions = torch.argmax(acts,dim=1)
            max_actions = max_actions[0,:-1] # Remove <EOS>
            max_actions = max_actions.cpu().numpy()
            hypothesis = [out_idx_to_token[a] for a in max_actions]
            hypotheses.append(hypothesis)
            # Get references
            ins_tokens = ins[0][1:-1] # Remove <EOS> and <SOS>
            key = '_'.join(ins_tokens)
            references = reference_dict[key]
            references_list.append(references)

    # Compute BLEU
    print("Computing BLEU score...")
    bleu = bleu_score.corpus_bleu(references_list,hypotheses)
    print("BLEU score: ", bleu*100)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
