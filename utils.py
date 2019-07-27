from data import ScanDataset,MTDataset
import json

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def generate_vocab_json(dataset,data_file,flip,vocab_out_json):
    if dataset == 'SCAN':
        data = ScanDataset(data_file)
    elif dataset == 'MT':
        data = MTDataset(data_file,flip)
    vocab = data.vocab
    with open(vocab_out_json,'w') as f:
        json.dump(vocab,f)
