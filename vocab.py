import json
from collections import Counter, defaultdict
import pickle
import pandas as pd
import os
import numpy as np
import torch
import spacy
from param import args

nlp = spacy.load('en_core_web_sm')
def tokenize_question(question):


    tokens = [token.text.lower() for token in nlp(question)]
    
    return tokens

def generate_question_vocab(file_path):
    with open(file_path, "r") as f:
        val_ls = json.load(f)


    ques_vocab = Counter()
    max_ques = -1

    for idx, dp in enumerate(val_ls):
        
        question = dp["Problem"]
        ques_tokens = tokenize_question(question)

        max_ques = max(max_ques, len(ques_tokens))
        ques_vocab.update(ques_tokens)

    return ques_vocab, max_ques

def ans_tokenize(ans):
    cc = 0
    if '|' in ans['linear_formula'][-1]:
        list_fromulas = ans['linear_formula'].split('|')[:-1]
#         print(1)
    elif '|' in ans['linear_formula']:
        list_fromulas = ans['linear_formula'].split('|')
#         print(list_fromulas)
    else:
        list_fromulas = [ans['linear_formula']]
        cc = 3
        
    tok_lis = []
#     print(list_fromulas)
    l = len(list_fromulas)
    for c,i in enumerate(list_fromulas):
        i = i.split('(')
        tok_lis.append(i[0].strip())
        tok_lis.append('(')
        if cc!= 3:
#             print(i)
            i[1] = i[1].split(',')
#             print(i[1])
            i[1][-1] = i[1][-1].replace(')','')
#             print(i[1])
            if len(i[1]) > 1:
                tok_lis.append(i[1][0].strip())
                tok_lis.append(',')
                tok_lis.append(i[1][1].strip())
            else:
                tok_lis.append(i[1][0].strip())
        else:
#             print(i)
            i[1] = i[1].split(',')
            i[1][-1] = i[1][-1].replace(')','')
#             print(i[1])
            if len(i[1]) > 1:
                tok_lis.append(i[1][0].strip())
                tok_lis.append(',')
                tok_lis.append(i[1][1].strip())
            else:
                tok_lis.append(i[1][0].strip())
        if l-1 != c:
            tok_lis.extend([')','|'])
        else:
            tok_lis.append(')')
#     print(f'tok_lis is {tok_lis}')
    return tok_lis
    
def ans_vocab(file_path):
    with open(file_path, "r") as f:
        ans_ls = json.load(f)
    fromula_vocab = Counter()
    max_formula = -1
    for ans in ans_ls:
#         print(ans)
        tokens = ans_tokenize(ans)
#         print(tokens)
        max_formula = max(max_formula, len(tokens))
        fromula_vocab.update(tokens)
    return fromula_vocab, max_formula
        
#     print(i)

SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]

def word_to_index(vocab):
    word2idx = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    idx2word = {i: token for i, token in enumerate(SPECIAL_TOKENS)}
    
    # Start indexing for non-special tokens after the special ones
    start_index = len(SPECIAL_TOKENS)
    
    # Update dictionaries with vocabulary words
    for i, (word, _) in enumerate(vocab.items(), start=start_index):
        if word not in word2idx:  # Check might be redundant if vocab is well-defined
            word2idx[word] = i
            idx2word[i] = word
    
    # Assert to ensure the dictionaries are inverses of each other
    assert len(word2idx) == len(idx2word), "word2idx and idx2word dictionaries mismatch in length"
    
    return word2idx, idx2word

def generate_encoder_vocab(train_path, output_path):

    encoder_vocab = Counter()
    

    ques_vocab, max_ques = generate_question_vocab(train_path)
 
    encoder_vocab.update(ques_vocab)
    

    encoder_vocab = dict(encoder_vocab.most_common())

    with open(os.path.join(output_path, "encoder.vocab"), "w") as out:
        for k, count in encoder_vocab.items():
            try:
                out.write(f"{k}\n")
            except:
                out.write(f"{str(k)}\n")
    


    en_word2idx, en_idx2word = word_to_index(encoder_vocab)
    
    
    with open(os.path.join(output_path, "encoder_word2idx.pickle"), 'wb') as out:
        pickle.dump(en_word2idx, out)
    
    with open(os.path.join(output_path, "encoder_idx2word.pickle"), 'wb') as out:
        pickle.dump(en_idx2word, out)
    

    return en_word2idx,en_idx2word, encoder_vocab

def generate_decoder_vocab(train_path, output_path):
    decoder_vocab = Counter()
    formula_vocab, max_formula = ans_vocab(train_path)
    decoder_vocab.update(formula_vocab)
    
    
    decoder_vocab = dict(decoder_vocab.most_common())

    with open(os.path.join(output_path, "decoder.vocab"), "w") as out:
        for k, count in decoder_vocab.items():
            try:
                out.write(f"{k}\n")
            except:
                out.write(f"{str(k)}\n")
                
    de_word2idx, de_idx2word = word_to_index(decoder_vocab)
      
    with open(os.path.join(output_path, "decoder_word2idx.pickle"), 'wb') as out:
        pickle.dump(de_word2idx, out)
    
    with open(os.path.join(output_path, "decoder_idx2word.pickle"), 'wb') as out:
        pickle.dump(de_idx2word, out)
    return de_word2idx,de_idx2word, decoder_vocab
    
def process_data(file_path, output_path, prefix = 'train'):
    
    data_points = []    
    with open(file_path, "r") as f:
        val_ls = json.load(f)

    for idx, dp in enumerate(val_ls):
        question = dp["Problem"]
        ques_tokens = " ".join(tokenize_question(question))
#         formula = dp["linear_formula"]
        formula_tokens = " ".join(ans_tokenize(dp))
#         if idx%20 == 0:
#             print(formula_tokens)
        data_points.append([ques_tokens, formula_tokens])
    
    df = pd.DataFrame(data_points, columns=["question", "linear_formula"])
    file_name = os.path.join(output_path, "{}_data.xlsx".format(prefix))
    df.to_excel(file_name, index=False)
    return



if __name__ == "__main__":

    generate_encoder_vocab("/home/mikshu/Documents/DL_A1_part2_git/data/train.json", "/home/mikshu/Documents/DL_A1_part2_git/preprocessed_data")
    print("processing training data...")
    generate_decoder_vocab("/home/mikshu/Documents/DL_A1_part2_git/data/train.json", "/home/mikshu/Documents/DL_A1_part2_git/preprocessed_data")
    process_data("/home/mikshu/Documents/DL_A1_part2_git/data/train.json", "/home/mikshu/Documents/DL_A1_part2_git/preprocessed_data", "train")
    print("processing val data...")
    process_data("/home/mikshu/Documents/DL_A1_part2_git/data/dev.json", "/home/mikshu/Documents/DL_A1_part2_git/preprocessed_data", "dev")