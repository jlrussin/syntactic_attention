from data import ScanDataset
import json

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def generate_vocab_json(data_file,vocab_out_json):
    dataset = ScanDataset(data_file)
    vocab = dataset.vocab
    with open(vocab_out_json,'w') as f:
        json.dump(vocab,f)
