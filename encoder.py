import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel
from torchtext.vocab import GloVe
from param import args

SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]

class GloveEmbeddings():
    def __init__(self, embed_dim, word2idx):
        self.embed_dim = embed_dim
        self.word2idx = word2idx
        self.vocab_size = len(word2idx)
        # Ensure special tokens are defined globally or within this context
        self.special_tokens = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.embedding_matrix = torch.zeros((self.vocab_size, self.embed_dim))
    def get_embedding_matrix(self):
        # Initialize GloVe embeddings
        glove = GloVe(name='6B', dim=self.embed_dim)
        

        # Special tokens handling
        self._initialize_special_tokens(self.embedding_matrix)

        # Populate the embedding matrix
        for word, idx in self.word2idx.items():
            if word in self.special_tokens:
                continue  # Skip special tokens
            self.embedding_matrix[idx] = self._get_word_embedding(word, glove)

        return self.embedding_matrix

    def _initialize_special_tokens(self, embedding_matrix):
        # Pad token already set to zeros by default
        for token in self.special_tokens:
            if token == '<pad>':
                continue  # Skip pad token, already zeros
            self.embedding_matrix[self.special_tokens[token]] = torch.randn(self.embed_dim)

    def _get_word_embedding(self, word, glove):
        # Retrieve a word's embedding if available, else use <unk> token's embedding
        return torch.tensor(glove.vectors[glove.stoi[word]]) if word in glove.stoi else self.embedding_matrix[self.special_tokens['<unk>']]
        


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_units=256, num_layers=1, p = 0.5, bidirectional=True, embed_matrix=None):
        super(LSTMEncoder, self).__init__()
#         self.input_size = input_size
        self.embed_dim = embed_dim
#         self.hidden_units = hidden_units
#         self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.bidirectional = bidirectional
        self.embed_matrix = None
        if self.embed_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(self.embed_matrix, padding_idx=0)
        else:
            self.embedding = nn.Embedding(input_size, self.embed_dim, padding_idx=0)
        self.LSTM = nn.LSTM(embed_dim, hidden_units, num_layers = num_layers, dropout=p, batch_first=True, bidirectional=True)
#         if bidirectional:
        self.hidden = nn.Linear(2*hidden_units, hidden_units)
        self.cell = nn.Linear(2*hidden_units, hidden_units)
            
    def forward(self, x):
#         print("ENCODER INPUT SHAPE", x.shape)
        x = self.dropout(self.embedding(x))
#         print("ENCODER EMBEDDING SHAPE", x.shape)
        
        encoder_out, (ht, ct) = self.LSTM(x)        
#         print("ENCODER OUTPUT SHAPE: encoder_out, ht, ct", encoder_out.shape, ht.shape, ct.shape)
#         if self.bidirectional:
            # concatenate the forward and backward LSTM hidden states
        ht = self.hidden(torch.cat((ht[0:1], ht[1:2]), dim=2))
        ct = self.cell(torch.cat((ct[0:1], ct[1:2]), dim=2))
        return encoder_out, (ht, ct)
    
class BertEncoder(nn.Module):
    def __init__(self,  model_type ,hidden_units=256):
        super(BertEncoder, self).__init__()
        self.bert_tune_layers = -1
        self.bert_model = BertModel.from_pretrained('bert-base-cased')
        self.model_type = model_type
        if self.model_type == 'frozen':
            for param in self.bert_model.parameters():
                        param.requires_grad = False
        elif self.model_type == 'tuned':
            for param in self.bert_model.parameters():
                        param.requires_grad = False
            for param in self.bert_model.encoder.layer[-self.bert_tune_layers].parameters():
                        param.requires_grad = True

    def forward(self, x, attn_mask):
        outputs = self.bert_model(input_ids = x, attention_mask = attn_mask)
        encodings = outputs.last_hidden_state
#         print(f'encoderout shape {encodings.shape}')
        return encodings
