
import os
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import json 
from time import time
from tqdm import tqdm
from datetime import datetime
from torchtext.vocab import GloVe
from dataset import *
from encoder import LSTMEncoder, BertEncoder,GloveEmbeddings
import random
from decoder import LSTMDecoder, LSTMAttnDecoder, LSTMAttnDecoderBERT
from param import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Seq2Seq_encoder_decoder(nn.Module):
    def __init__(self,encoder_w2i,encoder_ipsize,decoder_ipsize):
        super(Seq2Seq_encoder_decoder, self).__init__()
        self.decoder_ipsize = decoder_ipsize
        glove = GloveEmbeddings(256, encoder_w2i)
        embedding_matrix = glove.get_embedding_matrix()
            
        self.encoder = LSTMEncoder(input_size = encoder_ipsize, embed_dim = 300, p = 0.3,embed_matrix = embedding_matrix)
        self.decoder = LSTMDecoder(vocab_size = decoder_ipsize, embed_dim = 256, dropout_rate = 0.3)

        
    def forward(self, source, target, tf_ratio=0.6):
        batch_size, target_len = target.size()
        target_vocab_size = self.decoder_ipsize

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(device)
        _, (hidden, cell) = self.encoder(source)
        # hidden = torch.zeros(1, batch_size, 256).to(device)
        # cell = torch.zeros(1, batch_size, 256).to(device)
        # First input to the decoder is the <sos> tokens
        inpu = target[:, 0]
#         print(inpu)
        for t in range(1, target_len):
#             print(inpu)
            output, (hidden,cell) = self.decoder(inpu, (hidden, cell))
#             print(output.shape)
            outputs[:,t,:] = output.squeeze(1)
#             print(f'op shape {output.shape}')
            top1 = output.argmax(1) 
#             print(f'top1 shape {top1.shape}')
            if random.random() < tf_ratio:
                inpu = target[:, t]  
#                 print(inpu.shape)
            else:
                top1
            
        return outputs
    
class Seq2Seq_attn_encoder_decoder(nn.Module):
    def __init__(self,encoder_w2i,encoder_ipsize,decoder_ipsize):
        super(Seq2Seq_attn_encoder_decoder, self).__init__()
        self.decoder_ipsize = decoder_ipsize
        glove = GloveEmbeddings(256, encoder_w2i)
        embedding_matrix = glove.get_embedding_matrix()
            
        self.encoder = LSTMEncoder(input_size = encoder_ipsize, embed_dim = 300, p = 0.3,embed_matrix = embedding_matrix)
        self.decoder = LSTMAttnDecoder(vocab_size = decoder_ipsize, embed_dim = 256, dropout_rate = 0.3)

        
    def forward(self, source, target, tf_ratio=0.6):
        batch_size, target_len = target.size()
        target_vocab_size = self.decoder_ipsize

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(device)
        encoder_out, (hidden, cell) = self.encoder(source)
        # hidden = torch.zeros(1, batch_size, 256).to(device)
        # cell = torch.zeros(1, batch_size, 256).to(device)
        # First input to the decoder is the <sos> tokens
        inpu = target[:, 0]
#         print(inpu)
        for t in range(1, target_len):
#             print(inpu)
            output, (hidden,cell) = self.decoder(inpu, (hidden, cell),encoder_out)
#             print(output.shape)
            outputs[:,t,:] = output.squeeze(1)
#             print(f'op shape {output.shape}')
            top1 = output.argmax(1) 
#             print(f'top1 shape {top1.shape}')
            if random.random() < tf_ratio:
                inpu = target[:, t]  
#                 print(inpu.shape)
            else:
                top1
            
        return outputs

class Bert2SeqAttn(nn.Module):
    def __init__(self,encoder_w2i,encoder_ipsize,decoder_ipsize):
        super(Bert2SeqAttn, self).__init__()
        self.decoder_ipsize = decoder_ipsize
        glove = GloveEmbeddings(256, encoder_w2i)
#         embedding_matrix = glove.get_embedding_matrix()
            
        self.encoder = BertEncoder(args.bert_model_type)
        self.decoder = LSTMAttnDecoder(vocab_size = decoder_ipsize, embed_dim = 256, dropout_rate = 0.3)

        
    def forward(self, source, attn_mask, target, tf_ratio=0.6):
        batch_size, target_len = target.size()
        target_vocab_size = self.decoder_ipsize

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(device)
        encoder_outputs = self.encoder(source, attn_mask)
        hidden = torch.zeros(1, batch_size, 256).to(device)
        cell = torch.zeros(1, batch_size, 256).to(device)
        # First input to the decoder is the <sos> tokens
        inpu = target[:, 0]
#         print(inpu)
        for t in range(1, target_len):
#             print(inpu)
            output, (hidden,cell) = self.decoder(inpu, (hidden, cell),encoder_outputs)
#             print(output.shape)
            outputs[:,t,:] = output.squeeze(1)
#             print(f'op shape {output.shape}')
            top1 = output.argmax(1) 
#             print(f'top1 shape {top1.shape}')
            if random.random() < tf_ratio:
                inpu = target[:, t]  
#                 print(inpu.shape)
            else:
                top1
            
        return outputs