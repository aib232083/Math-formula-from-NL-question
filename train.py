import os
import subprocess
import json
import pickle
import shutil
from time import time
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn as nn
from param import args

from encoder_decoder import Seq2Seq_encoder_decoder, Seq2Seq_attn_encoder_decoder, Bert2SeqAttn
from dataset import Seq2Seq_dataset, collate, BERT_dataset, collate_bert
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    if args.model_type == "lstm_lstm" or args.model_type == "lstm_lstm_attn":
        train_dataset = Seq2Seq_dataset(args.preprocessed_dir, "train")
        train_loader = DataLoader(train_dataset, batch_size = 64, shuffle=True, 
                                    collate_fn=collate)

        val_dataset = Seq2Seq_dataset(args.preprocessed_dir, "dev")
        val_loader = DataLoader(val_dataset, batch_size = 64, shuffle=False, 
                                    collate_fn=collate)
        
    else:
        train_dataset = BERT_dataset(args.preprocessed_dir, "train")
        train_loader = DataLoader(train_dataset, batch_size = 64, shuffle=True, 
                                    collate_fn=collate)

        val_dataset = BERT_dataset(args.preprocessed_dir, "dev")
        val_loader = DataLoader(val_dataset, batch_size = 64, shuffle=False, 
                                    collate_fn=collate)       
    
    encoder_ipsize = len(train_dataset.en_word2idx)
    decoder_ipsize = len(train_dataset.de_word2idx)
    encoder_w2i = train_dataset.en_word2idx
    
    if args.model_type == "lstm_lstm":
        model = Seq2Seq_encoder_decoder(encoder_w2i,encoder_ipsize,decoder_ipsize).to(device)
    elif args.model_type == "lstm_lstm_attn":
        model = Seq2Seq_attn_encoder_decoder(encoder_w2i,encoder_ipsize,decoder_ipsize).to(device)
    else:
        model = Bert2SeqAttn(encoder_w2i,encoder_ipsize,decoder_ipsize).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
             
               
    scheduler =  optim.lr_scheduler.CosineAnnealingLR(optimizer, 2, verbose=False)

    loss_tracker = defaultdict(list)
    time_tracker = defaultdict(list)

    t0 = time()
    
    
    min_val_loss = np.inf
    for epoch in range(50):
        model.train()
        print("\n\n\n|===================================== Epoch: {} =====================================|".format(epoch))
        epoch_loss = []
        s = 0
        cnt = 0
        for c, data in enumerate(train_loader):
            l = []
#             print(f'data is {data}')
        #     print(data['question'].shape, data['query'].shape, data['ques_lens'].shape, data['query_lens'].shape)
            optimizer.zero_grad()
            question = data['question'].to(device)
#             print(question)
            query = data['formula'].to(device)
            
#             print(query)
            
            if args.model_type == "lstm_lstm" or  args.model_type == "lstm_lstm_attn":
                # ques_attn_mask = data['ques_attn_mask'].to(device)
                output = model(question, query)
            else:
                ques_attn_mask = data['ques_attn_mask'].to(device)
                output = model(question, query)

            for i in range(query.shape[0]):
                lt = []
                for j in range(output.shape[1]):
                    lt.append(torch.argmax(output[i][j]).item())
                l.append(lt)
            l = torch.tensor(l)
            
            for i in range(query.shape[0]):
#                 print(query[i],l[i].to(device))
                s += torch.equal(query[i],l[i].to(device))
                cnt +=1


            if c%200 == 0:
                print(query[1],l[1].to(device))
            output = output.reshape(-1, output.shape[2])
            query = query.reshape(-1)
            
#             print(f'shape l is {l.shape}')
#             print(f' query shape is {query.shape}')
#             print(query)
#             print(query.shape,output.shape)
            loss = criterion(output, query)
            loss.backward()
            epoch_loss.append(loss.item())
#             print(l,query)

#             cnt += 1
            optimizer.step()
#             print(f'exact match is {s}')
        print(f'train acc is {s/cnt}')
        scheduler.step()
        t1 = time()
        s = 0
        cnt = 0
        model.eval()
        val_loss = []
        if epoch % 1 == 0:
            
            print("Evaluating model on val data.")            
            prefix = "train_eval"

            
            for c, data in enumerate(val_loader):
#                 print(f'data is {data}')
                l = []
                question = data['question'].to(device)
                query = data['formula'].to(device)
                if args.model_type == "lstm_lstm" or  args.model_type == "lstm_lstm_attn":
                    # ques_attn_mask = data['ques_attn_mask'].to(device)
                    output = model(question, query)
                else:
                    ques_attn_mask = data['ques_attn_mask'].to(device)
                    output = model(question, query)
                for i in range(query.shape[0]):
                    lt = []
                    for j in range(output.shape[1]):
                        lt.append(torch.argmax(output[i][j]).item())
                    l.append(lt)
                l = torch.tensor(l)
            
                for i in range(query.shape[0]):
#                 print(query[i],l[i].to(device))
                    s += torch.equal(query[i],l[i].to(device))
                    cnt +=1
            
#                 l = l[:,1:].to(device)
#                 l = torch.cat([l,b],dim=1)
                if c%30 == 0:
                    print(query[1],l[1].to(device))
                output = output.reshape(-1, output.shape[2])
                query = query.reshape(-1)
#                 print(query.shape,output.shape)
                loss = criterion(output, query)
                val_loss.append(loss.item())

            print(f'val acc {s/cnt}')
        avg_val_loss = np.mean(val_loss)
        avg_epoch_loss = np.mean(epoch_loss)  
        
        loss_tracker['train'].append(avg_epoch_loss)
        loss_tracker['val'].append(avg_val_loss)

        print("Epoch: {}, Total Time Elapsed: {}Mins, Train Loss: {}, Val Loss: {}".format(epoch, round((t1-t0)/60,2), avg_epoch_loss, avg_val_loss))
        if loss_tracker["val"][-1] < min_val_loss:
            min_val_loss = loss_tracker["val"][-1]
            best_val_model = model
            torch.save({
            'epoch': epoch,
            'model_state_dict': best_val_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': loss_tracker['train'][-1],
            'val_loss' : loss_tracker['val'][-1],
            }, os.path.join(args.model_op_path,f'{args.model_type}.pth'))
            
            item = {'train_loss' : loss_tracker['train'],
                    'val_loss' : loss_tracker['val']}
            df = pd.DataFrame(item)
            df.to_csv(os.path.join(args.stats_path,'stats.csv'),index=False)
            
if __name__ == "__main__":
    with open(os.path.join(args.preprocessed_dir, "encoder_idx2word.pickle"), "rb") as file:
        encoder_idx2word = pickle.load(file)

    with open(os.path.join(args.preprocessed_dir, "decoder_idx2word.pickle"), "rb") as file:
        decoder_idx2word = pickle.load(file)

    train(args)