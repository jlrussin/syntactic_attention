import os
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from torch.utils.data import DataLoader
from data import ScanDataset,MTDataset,SCAN_collate
from SyntacticAttention import *
from utils import *

parser = argparse.ArgumentParser()
# Data
parser.add_argument('--dataset', choices=['SCAN','MT'],
                    default='SCAN',
                    help='Dataset class to use')
parser.add_argument('--flip', type=str2bool, default=False,
                    help='Flip source and target for MT dataset')
parser.add_argument('--train_data_file',
                    default='data/SCAN/tasks_train_addprim_jump.txt',
                    help='Path to training set')
parser.add_argument('--val_data_file',
                    default='data/SCAN/tasks_test_addprim_jump.txt',
                    help='Path to validation set')
parser.add_argument('--test_data_file',
                    default='data/SCAN/tasks_test_addprim_jump.txt',
                    help='Path to test set')
parser.add_argument('--load_vocab_json',default='vocab.json',
                    help='Path to vocab json file')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Samples per batch')
parser.add_argument('--num_iters', type=int, default=200000,
                    help='Number of optimizer steps before stopping')
parser.add_argument('--seed', type=int, default=1,
                    help='Seed for random number generators')

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

# Optimization
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Fixed learning rate for Adam optimizer')
parser.add_argument('--clip_norm', type=float, default=5.0,
                    help='Maximum L2-norm at which gradients will be clipped.')

# Output options
parser.add_argument('--results_dir', default='results',
                    help='Results subdirectory to save results')
parser.add_argument('--out_data_file', default='results.json',
                    help='Name of output data file')
parser.add_argument('--checkpoint_path',default=None,
                    help='Path to output saved weights.')
parser.add_argument('--checkpoint_every', type=int, default=5,
                    help='Epochs before evaluating model and saving weights')
parser.add_argument('--record_loss_every', type=int, default=400,
                    help='iters before printing and recording loss')

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Vocab
    with open(args.load_vocab_json,'r') as f:
        vocab = json.load(f)

    # Datasets
    if args.dataset == 'SCAN':
        all_train_data = ScanDataset(args.train_data_file,vocab)
        split_id = int(0.8*len(all_train_data))
        train_data = [all_train_data[i] for i in range(split_id)]
        val_data = [all_train_data[i] for i in range(split_id,len(all_train_data))]
        test_data = ScanDataset(args.test_data_file,vocab)
    elif args.dataset == 'MT':
        train_data = MTDataset(args.train_data_file,vocab,args.flip)
        val_data = MTDataset(args.val_data_file,vocab,args.flip)
        test_data = MTDataset(args.test_data_file,vocab,args.flip)


    # Dataloaders
    train_loader = DataLoader(train_data,args.batch_size,
                              shuffle=True,collate_fn=SCAN_collate)
    val_loader = DataLoader(val_data,args.batch_size,
                            shuffle=True,collate_fn=SCAN_collate)
    test_loader = DataLoader(test_data,args.batch_size,
                             shuffle=True,collate_fn=SCAN_collate)

    in_vocab_size = len(vocab['in_token_to_idx'])
    out_vocab_size = len(vocab['out_idx_to_token'])
    # Model
    model = Seq2SeqSynAttn(in_vocab_size, args.m_hidden_dim, args.x_hidden_dim,
                           out_vocab_size, args.rnn_type, args.n_layers,
                           args.dropout_p, args.seq_sem, args.syn_act,
                           args.sem_mlp, None, device)

    if args.load_weights_from is not None:
        model.load_state_dict(torch.load(args.load_weights_from))
    model.to(device)

    # Loss function
    loss_fn = nn.NLLLoss(reduction='mean',ignore_index=-100)
    loss_fn = loss_fn.to(device)

    # Optimizer
    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.learning_rate)

    # Training loop:
    iter = 0
    epoch_count = 0
    loss_data, train_errors, val_errors, test_errors = [],[],[],[]
    best_val_error = 1.1 # best validation error - for early stopping
    while iter < args.num_iters:
        epoch_count += 1
        for sample_count,sample in enumerate(train_loader):
            iter += 1
            # Forward pass
            instructions, true_actions, _, _ = sample
            instructions = [ins.to(device) for ins in instructions]
            true_actions = [ta.to(device) for ta in true_actions]
            optimizer.zero_grad()
            actions,padded_true_actions = model(instructions,true_actions)
            # Compute NLLLoss
            true_actions = padded_true_actions.to(device)
            loss = loss_fn(actions,padded_true_actions)
            # Backward pass
            loss.backward()
            if args.clip_norm is not None:
                clip_grad_norm_(params,max_norm=args.clip_norm)
            optimizer.step()
            # Record loss
            if iter % args.record_loss_every == 0:
                loss_datapoint = loss.data.item()
                print('Epoch:', epoch_count,
                      'Iter:', iter,
                      'Loss:', loss_datapoint)
                loss_data.append(loss_datapoint)
        # Checkpoint
        last_epoch = (iter >= args.num_iters)
        if epoch_count % args.checkpoint_every == 0 or last_epoch:
            print("Checking training error...")
            train_error = check_accuracy(train_loader, model, device, args)
            print("Training error is ", train_error)
            train_errors.append(train_error)
            print("Checking validation error...")
            val_error = check_accuracy(val_loader, model, device, args)
            print("Validation error is ", val_error)
            val_errors.append(val_error)
            print("Checking test error...")
            test_error = check_accuracy(test_loader, model, device, args)
            print("Test error is ", test_error)
            test_errors.append(test_error)

            # Write stats file
            results_path = '../results/%s' % (args.results_dir)
            if not os.path.isdir(results_path):
                os.mkdir(results_path)
            stats = {'loss_data':loss_data,
                     'train_errors':train_errors,
                     'val_errors':val_errors,
                     'test_errors':test_errors}
            results_file_name = '%s/%s' % (results_path,args.out_data_file)
            with open(results_file_name, 'w') as f:
                json.dump(stats, f)

            # Save model weights
            if val_error < best_val_error: # use val (not test) to decide to save
                best_val_error = val_error
                if args.checkpoint_path is not None:
                    torch.save(model.state_dict(),
                               args.checkpoint_path)


def check_accuracy(dataloader, model, device, args):
    model.eval()
    with torch.no_grad():
        all_correct_trials = [] # list of booleans indicating whether correct
        for sample in dataloader:
            instructions, true_actions, _, _ = sample
            batch_size = len(instructions)
            out_vocab_size = model.out_vocab_size
            instructions = [ins.to(device) for ins in instructions]
            true_actions = [ta.to(device) for ta in true_actions]
            actions,padded_true_actions = model(instructions, true_actions)

            # Manually unpad with mask to compute accuracy
            mask = padded_true_actions == -100
            max_actions = torch.argmax(actions,dim=1)
            correct_actions = max_actions == padded_true_actions
            correct_actions = correct_actions + mask # Add boolean mask
            correct_actions = correct_actions.cpu().numpy()
            correct_trials = np.all(correct_actions,axis=1).tolist()
            all_correct_trials = all_correct_trials + correct_trials
    model.train()
    return 1.0 - np.mean(all_correct_trials)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
