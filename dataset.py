from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertTokenizer
import pandas as pd
import os
import pickle
from vocab import *
import torch
from param import args

class Seq2Seq_dataset(Dataset):
    def __init__(self, file_path, data_prefix="train"):
        self.file_path = file_path
        self.data = self.load_data(data_prefix)
        self.encoder_vocab, self.en_word2idx, self.en_idx2word = self.load_vocabulary("encoder")
        self.decoder_vocab, self.de_word2idx, self.de_idx2word = self.load_vocabulary("decoder")
        print(f"Encoder Vocab Size = {len(self.en_word2idx)}, Decoder Vocab Size = {len(self.de_word2idx)}")

    def load_data(self, prefix):
        data_path = os.path.join(self.file_path, f"{prefix}_data.xlsx")
        return pd.read_excel(data_path)
    
    def load_vocabulary(self, role):
        vocab_file = os.path.join(self.file_path, f"{role}.vocab")
        with open(vocab_file, "r") as file:
            vocab = file.readlines()
        
        word2idx_file = os.path.join(self.file_path, f"{role}_word2idx.pickle")
        idx2word_file = os.path.join(self.file_path, f"{role}_idx2word.pickle")
        with open(word2idx_file, "rb") as file:
            word2idx = pickle.load(file)
        with open(idx2word_file, "rb") as file:
            idx2word = pickle.load(file)
        
        return vocab, word2idx, idx2word
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question_tokens = ["<sos>"] + tokenize_question(row['question']) + ["<eos>"]
        formula_tokens = ["<sos>"] + ans_tokenize(row) + ["<eos>"]

        question_indices = [self.en_word2idx.get(token, self.en_word2idx["<unk>"]) for token in question_tokens]
        formula_indices = [self.de_word2idx.get(token, self.de_word2idx["<unk>"]) for token in formula_tokens]

        sample = {
            'id': row['id'],
            'question': question_indices,
            'formula': formula_indices,
            'answer': row['answer']
        }
        return sample
    
def collate(batch):
    
    max_len_ques = max([len(sample['question']) for sample in batch])
    max_len_formula = max([len(sample['formula']) for sample in batch])
    
    ques_lens = torch.zeros(len(batch), dtype=torch.long)
    padded_ques = torch.zeros((len(batch), max_len_ques), dtype=torch.long)
    # ques_attn_mask = torch.zeros((len(batch), max_len_ques), dtype=torch.long)
    
    formula_lens = torch.zeros(len(batch), dtype=torch.long)
    padded_formula = torch.zeros((len(batch), max_len_formula), dtype=torch.long)


    for idx in range(len(batch)):
        
        formula = batch[idx]['formula']
        question = batch[idx]['question']
        
        ques_len = len(question)
        formula_len = len(formula)
        ques_lens[idx] = ques_len
        formula_lens[idx] = formula_len
        # indexes[idx] = batch[idx]['index']
        
        padded_ques[idx, :ques_len] = torch.LongTensor(question)
        # ques_attn_mask[idx, :ques_len] = torch.ones((1, ques_len), dtype=torch.long)

        padded_formula[idx, :formula_len] = torch.LongTensor(formula)
        

        
    return {'question': padded_ques, 'formula': padded_formula, 'answer' : batch[idx]['answer'],'ques_lens': formula_lens, 'query_lens': formula_lens}

    
class BERT_dataset(Dataset):
    def __init__(self, file_path, data_prefix="train"):
        self.file_path = file_path
        self.data = self.load_data(data_prefix)
        self.encoder_vocab, self.en_word2idx, self.en_idx2word = self.load_vocabulary("encoder")
        self.decoder_vocab, self.de_word2idx, self.de_idx2word = self.load_vocabulary("decoder")
        print(f"Encoder Vocab Size = {len(self.en_word2idx)}, Decoder Vocab Size = {len(self.de_word2idx)}")
        self.en_tokenizer =  BertTokenizer.from_pretrained("bert-base-cased")
    def load_data(self, prefix):
        data_path = os.path.join(self.file_path, f"{prefix}_data.xlsx")
        return pd.read_excel(data_path)
    
    def load_vocabulary(self, role):
        vocab_file = os.path.join(self.file_path, f"{role}.vocab")
        with open(vocab_file, "r") as file:
            vocab = file.readlines()
        
        word2idx_file = os.path.join(self.file_path, f"{role}_word2idx.pickle")
        idx2word_file = os.path.join(self.file_path, f"{role}_idx2word.pickle")
        with open(word2idx_file, "rb") as file:
            word2idx = pickle.load(file)
        with open(idx2word_file, "rb") as file:
            idx2word = pickle.load(file)
        
        return vocab, word2idx, idx2word
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question_tokens = self.en_tokenizer.encode(row['question'])
        formula_tokens = ["<sos>"] + ans_tokenize(row) + ["<eos>"]

        question_indices = [self.en_word2idx.get(token, self.en_word2idx["<unk>"]) for token in question_tokens]
        formula_indices = [self.de_word2idx.get(token, self.de_word2idx["<unk>"]) for token in formula_tokens]

        sample = {
            'id': row['id'],
            'question': question_indices,
            'formula': formula_indices,
            'answer': row['answer']
        }
        return sample
    
def collate_bert(batch):
    
    max_len_ques = max([len(sample['question']) for sample in batch])
    max_len_formula = max([len(sample['formula']) for sample in batch])
    
    ques_lens = torch.zeros(len(batch), dtype=torch.long)
    padded_ques = torch.zeros((len(batch), max_len_ques), dtype=torch.long)
    ques_attn_mask = torch.zeros((len(batch), max_len_ques), dtype=torch.long)
    
    formula_lens = torch.zeros(len(batch), dtype=torch.long)
    padded_formula = torch.zeros((len(batch), max_len_formula), dtype=torch.long)


    for idx in range(len(batch)):
        
        formula = batch[idx]['formula']
        question = batch[idx]['question']
        
        ques_len = len(question)
        formula_len = len(formula)
        ques_lens[idx] = ques_len
        formula_lens[idx] = formula_len
        # indexes[idx] = batch[idx]['index']
        
        padded_ques[idx, :ques_len] = torch.LongTensor(question)
        ques_attn_mask[idx, :ques_len] = torch.ones((1, ques_len), dtype=torch.long)

        padded_formula[idx, :formula_len] = torch.LongTensor(formula)
        

        
    return {'question': padded_ques,  'ques_attn_mask': ques_attn_mask,'formula': padded_formula, 'answer' : batch[idx]['answer'],'ques_lens': formula_lens, 'query_lens': formula_lens}

