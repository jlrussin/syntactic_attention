import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SynAttnEncoder(nn.Module):
    def __init__(self, in_vocab_size, m_hidden_size, x_hidden_size, rnn_type,
                 num_layers, dropout_p, seq_sem, device):
        super(SynAttnEncoder, self).__init__()
        self.in_vocab_size = in_vocab_size # size of input vocabulary
        self.m_hidden_size = m_hidden_size # semantic hidden size
        self.x_hidden_size = x_hidden_size # syntax hidden size
        self.rnn_type = rnn_type # type of RNN (GRU or LSTM)
        self.num_layers = num_layers # number of layers in RNNs
        self.dropout_p = dropout_p # dropout rate
        self.seq_sem = seq_sem # sequential semantics option
        self.device = device

        # Embeddings for instructions (equivalent to eq. 1 in paper)
        self.W_m = nn.Embedding(in_vocab_size, m_hidden_size)
        self.W_x = nn.Embedding(in_vocab_size, x_hidden_size//2)

        # Dropout
        self.dropout = nn.Dropout(dropout_p) # used for dropout on embeddings
        rnn_dropout = dropout_p if num_layers > 1 else 0.0

        # RNN for sequential semantic processing
        if seq_sem:
            if rnn_type == 'GRU':
                self.semantic_rnn = nn.GRU(m_hidden_size, m_hidden_size//2,
                                           num_layers, dropout=rnn_dropout,
                                           bidirectional=True)
            elif rnn_type == 'LSTM':
                self.semantic_rnn = nn.LSTM(m_hidden_size, m_hidden_size//2,
                                            num_layers, dropout=rnn_dropout,
                                            bidirectional=True)
        # SOS and EOS
        self.SOS_embed = nn.Embedding(1,x_hidden_size)
        self.EOS_embed = nn.Embedding(1,x_hidden_size)

        # Recurrent networks for processing syntactic instructions (2 layers)
        if rnn_type == 'GRU':
            self.rnn1= nn.GRU(x_hidden_size//2, x_hidden_size//2, num_layers,
                              dropout=rnn_dropout, bidirectional=True)
            self.rnn2 = nn.GRU(x_hidden_size, x_hidden_size, num_layers,
                               dropout=rnn_dropout, bidirectional=True)
        elif rnn_type == 'LSTM':
            self.rnn1 = nn.LSTM(x_hidden_size//2, x_hidden_size//2, num_layers,
                                dropout=rnn_dropout, bidirectional=True)
            self.rnn2 = nn.LSTM(x_hidden_size, x_hidden_size, num_layers,
                                dropout=rnn_dropout, bidirectional=True)

        # Fully connected layer for producing first s_i
        self.fc0 = nn.Linear(2*x_hidden_size,2*x_hidden_size)
        self.relu = nn.ReLU()

    def forward(self, instructions):
        batch_size = len(instructions)

        # Pad sequences in instructions
        seq_lens = [ins.shape[0] for ins in instructions]
        instructions = pad_sequence(instructions)
        max_len = instructions.shape[0]

        # Embeddings
        m_i = self.W_m(instructions)
        m_i = self.dropout(m_i)
        sx_i = self.W_x(instructions)
        sx_i = self.dropout(sx_i)

        # Sequential semantics: see section 3.4 of paper
        if self.seq_sem:
            packed_sem = pack_padded_sequence(m_i,seq_lens)
            m_i, _ = self.semantic_rnn(packed_sem)
            m_i, _ = pad_packed_sequence(m_i)

        # First layer of syntax RNN
        packed = pack_padded_sequence(sx_i,seq_lens)
        h1_j, _ = self.rnn1(packed) # Don't need hidden
        h1_j, seq_lens = pad_packed_sequence(h1_j)

        # Take only contextual information (eq. 2 from paper)
        h1_j = h1_j.view(max_len,batch_size,2,self.x_hidden_size//2)
        context = torch.zeros(max_len,batch_size,2,
                              self.x_hidden_size//2,device=self.device)
        for forward_i,backward_i in zip(range(1,max_len,1),range(max_len-2,-1,-1)):
            # Fill with previous forward hidden
            context[forward_i,:,0,:] = h1_j[forward_i-1,:,0,:]
            # Fill with previous backward hidden
            context[backward_i,:,1,:] = h1_j[backward_i+1,:,1,:]
        h1_j = context.view(max_len,batch_size,self.x_hidden_size)

        # <SOS> and <EOS> roles
        h1_j[0,:,:] = self.SOS_embed(torch.tensor(0,device=self.device))
        h1_j[-1,:,:] = self.EOS_embed(torch.tensor(0,device=self.device))

        # Second layer of syntax RNN
        h_j, hidden = self.rnn2(h1_j)
        if self.rnn_type == 'LSTM':
            hidden = hidden[0] # Use only "h_n" if using LSTM

        # Transform final hidden state to get first s_i
        hidden = hidden.view(self.num_layers,2,batch_size,self.x_hidden_size)
        hidden = hidden[-1,:,:,:] # get last layer activities only
        hidden = torch.cat((hidden[0,:,:],hidden[1,:,:]),dim=1)
        s_0 = self.relu(self.fc0(hidden))
        s_0 = s_0.unsqueeze(2)

        return m_i, h_j, s_0

class SynAttnDecoder(nn.Module):
    def __init__(self, m_hidden_size, x_hidden_size, out_vocab_size, rnn_type,
                 num_layers, dropout_p, syn_act, sem_mlp, device):
        super(SynAttnDecoder, self).__init__()
        self.m_hidden_size = m_hidden_size # semantic hidden size
        self.x_hidden_size = x_hidden_size # syntax hidden size
        self.out_vocab_size = out_vocab_size # number of possible actions
        self.rnn_type = rnn_type # RNN type (GRU or LSTM)
        self.num_layers = num_layers # number of layers in decoder rnn
        self.dropout_p = dropout_p # dropout rate
        self.syn_act = syn_act # Syntax-action option
        self.sem_mlp = sem_mlp # Semantic MLP (Nonlinear semantics)
        self.device = device

        # Softmax for producing attention over command sequence
        self.softmax = nn.Softmax(dim=1)

        # Decoder RNN "g" from equations in paper
        rnn_dropout = dropout_p if num_layers > 1 else 0.0 # no RNN dropout if num_layers=1
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(2*x_hidden_size, 2*x_hidden_size, num_layers, dropout=rnn_dropout)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(2*x_hidden_size, 2*x_hidden_size, num_layers, dropout=rnn_dropout)

        # Nonlinear semantics with MLP
        if sem_mlp:
            self.sem_nn = nn.Linear(m_hidden_size,m_hidden_size)
            self.relu = nn.ReLU()

        # Output layer for producing actions from semantic embeddings
        if syn_act:
            # Output will produced from concatenation of syntax and semantics
            self.out = nn.Linear(2*x_hidden_size + m_hidden_size, out_vocab_size)
        else:
            self.out = nn.Linear(m_hidden_size, out_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self,s_i,hidden,m_i, h_j):

        # Attention over commands is produced from s_i
        h_j = h_j.permute(1,0,2)
        e_ij = torch.bmm(h_j,s_i) # equations (4,5) from paper
        alpha_ij = self.softmax(e_ij) # equation (4) from paper

        # RNN input from syntactic encodings only
        h_j = h_j.permute(0,2,1)
        c_i = torch.bmm(h_j,alpha_ij) # equation (6) from paper

        # Actions computed from weighted average of semantic embeddings
        m_i = m_i.permute(1,2,0)
        m_i = torch.bmm(m_i,alpha_ij) # equation (3) from paper

        # Nonlinear semantics: equation (1) in supplementary materials
        if self.sem_mlp:
            m_i = m_i.permute(0,2,1)
            m_i = self.relu(m_i)
            m_i = self.sem_nn(m_i)
            m_i = self.relu(m_i)
            m_i = m_i.permute(0,2,1)

        # Syntax-action: see section 3.4 of paper
        if self.syn_act:
            m_i = torch.cat((m_i,c_i),dim=1)
        a_i = m_i.squeeze(2)
        a_i = self.out(a_i)
        a_i = self.log_softmax(a_i)

        # RNN produces new control signal
        c_i = c_i.permute(2,0,1) # RNN expects (seq_len,batch,input_size)
        s_next, hidden = self.rnn(c_i,hidden) # equation (6) from paper
        s_next = s_next.permute(1,2,0)

        return a_i, s_next, hidden, alpha_ij

    def initHidden(self,batch_size):
        zeros_tensor = torch.zeros(self.num_layers, batch_size,
                                   2*self.x_hidden_size, device=self.device)
        if self.rnn_type == 'GRU':
            return zeros_tensor
        elif self.rnn_type == 'LSTM':
            return (zeros_tensor,zeros_tensor)

class Seq2SeqSynAttn(nn.Module):
    def __init__(self, in_vocab_size, m_hidden_size, x_hidden_size,
                 out_vocab_size, rnn_type, num_layers, dropout_p,
                 seq_sem, syn_act, sem_mlp, max_len, device):
        super(Seq2SeqSynAttn, self).__init__()
        self.in_vocab_size = in_vocab_size # number of commands
        self.m_hidden_size = m_hidden_size # semantic hidden size
        self.x_hidden_size = x_hidden_size # syntax hidden size
        self.out_vocab_size = out_vocab_size # number of actions
        self.rnn_type = rnn_type # (GRU or LSTM)
        self.num_layers = num_layers # number of layers in RNNs
        self.dropout_p = dropout_p # dropout rate
        self.seq_sem = seq_sem # Sequential semantics option
        self.syn_act = syn_act # Syntax-Action option
        self.sem_mlp = sem_mlp # Nonlinear semantics (semantics with MLP)
        self.max_len = max_len # Max length (only for prediction)
        self.device = device

        self.encoder = SynAttnEncoder(in_vocab_size, m_hidden_size,
                                      x_hidden_size, rnn_type, num_layers,
                                      dropout_p, seq_sem, device)
        self.decoder = SynAttnDecoder(m_hidden_size, x_hidden_size,
                                      out_vocab_size, rnn_type, num_layers,
                                      dropout_p, syn_act, sem_mlp, device)

    def forward(self, instructions, true_actions):
        batch_size = len(instructions)
        # Pad sequences of true actions
        seq_lens = [a.shape[0] for a in true_actions]
        padded_true_actions = pad_sequence(true_actions,padding_value=-100)
        if self.max_len is None:
            max_len = padded_true_actions.shape[0]
        else:
            max_len = self.max_len

        # Encoder
        m_i,h_j,s_i = self.encoder(instructions)

        # Decoder
        hidden = self.decoder.initHidden(batch_size)
        actions = torch.zeros(max_len,batch_size,self.out_vocab_size,
                              device=self.device)
        for i in range(1,max_len):
            a_i, s_i, hidden, alpha_ij = self.decoder(s_i, hidden, m_i, h_j)
            actions[i,:,:] = a_i

        # Prepare decoder outputs and actions for NLLLoss (ignore -100)
        actions = actions[1:,:,:] # Remove <SOS> token
        actions = actions.permute(1,2,0) # (batch,out_vocab_size,seq_len)

        padded_true_actions = padded_true_actions[1:,:] # Remove <SOS> token
        padded_true_actions = padded_true_actions.permute(1,0) # (batch,seq_len)

        return actions, padded_true_actions
