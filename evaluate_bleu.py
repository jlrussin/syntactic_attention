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
parser.add_argument('--train_data_file',
                    default='data/MT/train',
                    help='Path to test set')
parser.add_argument('--val_data_file',
                    default='data/MT/test_no_unk.txt',
                    help='Path to test set')
parser.add_argument('--test_data_file',
                    default='data/MT/test',
                    help='Path to test set')
parser.add_argument('--load_vocab_json',default='vocab_eng_MT_all.json',
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

def get_reference_dict(dataset):
    reference_dict = {}
    for sample in dataset:
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
    return reference_dict

def evaluate_bleu(dataloader,vocab,reference_dict,model,max_len,device):
    # Setup
    out_idx_to_token = vocab['out_idx_to_token']
    model.max_len = max_len
    print("Getting predictions...")
    hypotheses = []
    references_list = []
    with torch.no_grad():
        for sample_count,sample in enumerate(dataloader):
            # Forward pass
            instructions, true_actions, ins_tokens, act_tokens = sample
            instructions = [ins.to(device) for ins in instructions]
            true_actions = [ta.to(device) for ta in true_actions]
            if len(true_actions[0]) < 6:
                continue # Don't include if less than 4 words (without SOS, EOS)
            actions,padded_true_actions = model(instructions,true_actions)
            # Get hypothesis
            max_actions = torch.argmax(actions,dim=1)
            max_actions = max_actions.squeeze(0).cpu().numpy()
            out_tokens = [out_idx_to_token[str(a)] for a in max_actions]
            if '<EOS>' in out_tokens:
                eos_index = out_tokens.index('<EOS>')
            else:
                eos_index = len(out_tokens)
            hypothesis = out_tokens[:eos_index]
            hypotheses.append(hypothesis)
            # Get references
            ins_words = ins_tokens[0][1:-1] # Remove <EOS> and <SOS>
            key = '_'.join(ins_words)
            references = reference_dict[key]
            references_list.append(references)

    # Compute BLEU
    print("Computing BLEU score...")
    bleu = bleu_score.corpus_bleu(references_list,hypotheses)
    bleu = bleu*100

    # Return model max_len to None
    model.max_len = None

    return bleu

def main(args):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Vocab
    with open(args.load_vocab_json,'r') as f:
        vocab = json.load(f)
    in_vocab_size = len(vocab['in_token_to_idx'])
    out_vocab_size = len(vocab['out_idx_to_token'])

    # Dataset
    batch_size = 1
    train_data = MTDataset(args.train_data_file,vocab,args.flip)
    val_data = MTDataset(args.val_data_file,vocab,args.flip)
    test_data = MTDataset(args.test_data_file,vocab,args.flip)
    train_loader = DataLoader(train_data,batch_size,
                             shuffle=True,collate_fn=SCAN_collate)
    val_loader = DataLoader(val_data,batch_size,
                             shuffle=True,collate_fn=SCAN_collate)
    test_loader = DataLoader(test_data,batch_size,
                             shuffle=True,collate_fn=SCAN_collate)

    # Max lengths
    train_max_len = max([len(b[3]) for b in train_data])
    val_max_len = max([len(b[3]) for b in val_data])
    test_max_len = max([len(b[3]) for b in test_data])

    # Reference dicts
    train_ref_dict = get_reference_dict(train_data)
    val_ref_dict = get_reference_dict(val_data)
    test_ref_dict = get_reference_dict(test_data)

    # Model
    model = Seq2SeqSynAttn(in_vocab_size, args.m_hidden_dim, args.x_hidden_dim,
                           out_vocab_size, args.rnn_type, args.n_layers,
                           args.dropout_p, args.seq_sem, args.syn_act,
                           args.sem_mlp, None, device)

    if args.load_weights_from is not None:
        model.load_state_dict(torch.load(args.load_weights_from))
    model.to(device)
    model.eval()

    # Get BLEU scores
    train_bleu = evaluate_bleu(train_loader,vocab,train_ref_dict,model,
                               train_max_len, device)
    print("Training set: %s" % args.train_data_file)
    print("Train BLEU: %f" % train_bleu)

    val_bleu = evaluate_bleu(val_loader,vocab,val_ref_dict,model,
                               val_max_len, device)
    print("Validation set: %s" % args.val_data_file)
    print("Validation BLEU: %f" % val_bleu)

    test_bleu = evaluate_bleu(test_loader,vocab,test_ref_dict,model,
                              test_max_len, device)
    print("Test set: %s" % args.train_data_file)
    print("Test BLEU: %f" % train_bleu)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
