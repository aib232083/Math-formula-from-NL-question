import os
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from param import args

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_units=256, num_layers=1, dropout_rate=0.5):
        super(LSTMAttnDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(2*hidden_units + embed_dim, hidden_units, num_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_units, vocab_size)
        # self.attention = AttentionNetwork(hidden_units)
        
    def forward(self, x, h_c):
        x = self.dropout(self.embedding(x))
        x = x.unsqueeze(1)
#         print(f' emb shape {x.shape}')
        # context, attn_weights = self.attention(h_c[0], encoder_out)
        # x = torch.cat([x, context], dim=2)
        lstm_out, (ht,ct) = self.lstm(x, h_c)
        output = self.fc(lstm_out)
        return output, (ht,ct)
    

class AttentionNetwork(nn.Module):
    def __init__(self, hidden_units):
        super(AttentionNetwork, self).__init__()
        self.attn_combine = nn.Linear(hidden_units * 3, hidden_units)
        self.score_layer = nn.Linear(hidden_units, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs):
        decoder_hidden = decoder_hidden.repeat(encoder_outputs.shape[1], 1 , 1)
        decoder_hidden = decoder_hidden.transpose(0,1)
#         print(decoder_hidden.shape,encoder_outputs.shape)
       
        combined = torch.cat((decoder_hidden, encoder_outputs), dim=2)
        
        energy = torch.tanh(self.attn_combine(combined))
        scores = self.score_layer(energy).squeeze(2)
        
        attention_weights = F.softmax(scores, dim=1).unsqueeze(1)
        
        context = torch.bmm(attention_weights, encoder_outputs)
        
        return context, attention_weights
    
    
class LSTMAttnDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_units=256, num_layers=1, dropout_rate=0.5):
        super(LSTMAttnDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(2*hidden_units + embed_dim, hidden_units, num_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_units, vocab_size)
        self.attention = AttentionNetwork(hidden_units)
        
    def forward(self, x, h_c, encoder_out):
        x = self.dropout(self.embedding(x))
        x = x.unsqueeze(1)
#         print(f' emb shape {x.shape}')
        context, attn_weights = self.attention(h_c[0], encoder_out)
        x = torch.cat([x, context], dim=2)
        lstm_out, (ht,ct) = self.lstm(x, h_c)
        output = self.fc(lstm_out)
        return output, (ht,ct)
    

class AttentionNetworkBERT(nn.Module):
    def __init__(self, hidden_units):
        super(AttentionNetworkBERT, self).__init__()
        self.attn_combine = nn.Linear(hidden_units * 4, hidden_units)
        self.score_layer = nn.Linear(hidden_units, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs):
        # Repeat decoder hidden state to match the encoder output shape
#         print(decoder_hidden)
        decoder_hidden = decoder_hidden.repeat(encoder_outputs.shape[1], 1 , 1)
        decoder_hidden = decoder_hidden.transpose(0,1)
#         print(decoder_hidden.shape,encoder_outputs.shape)
        # Concatenate the decoder hidden state and encoder outputs
        combined = torch.cat((decoder_hidden, encoder_outputs), dim=2)
        
        # Compute the attention energies
        energy = torch.tanh(self.attn_combine(combined))
        scores = self.score_layer(energy).squeeze(2)
        
        # Apply softmax to normalize the scores to probabilities
        attention_weights = F.softmax(scores, dim=1).unsqueeze(1)
        
        # Compute the context vectors
        context = torch.bmm(attention_weights, encoder_outputs)
        
        return context, attention_weights
    

class LSTMAttnDecoderBERT(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_units=256, num_layers=1, dropout_rate=0.5):
        super(LSTMAttnDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(768 + embed_dim, hidden_units, num_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_units, vocab_size)
        self.attention = AttentionNetworkBERT(hidden_units)
        
    def forward(self, x, h_c, encoder_out):
        x = self.dropout(self.embedding(x))
        x = x.unsqueeze(1)
#         print(f' emb shape {x.shape}')
#         print(h_c[0].shape)
#         h_c[0] = torch.tensor(h_c[0])
        context, attn_weights = self.attention(h_c[0], encoder_out)
        x = torch.cat([x, context], dim=2)
        lstm_out, (ht,ct) = self.lstm(x, h_c)
        output = self.fc(lstm_out)
        return output, (ht,ct)
