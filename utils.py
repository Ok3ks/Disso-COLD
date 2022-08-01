##############################################################
#
# RoBERT.py
# This file contains the implementation of the RoBERT model
# An LSTM is applied to a segmented document. The resulting
# embedding is used for document-level classification
#
##############################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import transformers
# get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
import time
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


#Credits to xx for this part of the code
class RoBERT_Model(nn.Module):
    """ Make an LSTM model over a fine tuned bert model. Parameters
    __________
    bertFineTuned: BertModel
        A bert fine tuned instance
    """

    #added output_class to function

    def __init__(self, bertFineTuned, num_classes = 2):
        super(RoBERT_Model, self).__init__()
        self.bertFineTuned = bertFineTuned()
        self.lstm = nn.LSTM(768, 100, num_layers=1, bidirectional=False)
        self.out = nn.Softmax(dim= num_classes)
    

    def forward(self, ids, mask, token_type_ids, lengt):
        """ Define how to performed each call
        Parameters
        __________
        ids: array
            -
        mask: array
            - 
        token_type_ids: array
            -
        lengt: int
            -
        Returns:
        _______
        -
        """
        _, pooled_out = self.bertFineTuned(ids, attention_mask=mask, token_type_ids=token_type_ids)
        chunks_emb = pooled_out.split_with_sizes(lengt)

        seq_lengths = torch.LongTensor([x for x in map(len, chunks_emb)])

        batch_emb_pad = nn.utils.rnn.pad_sequence(chunks_emb, padding_value=-91, batch_first=True)

        batch_emb = batch_emb_pad.transpose(0, 1)  # (B,L,D) -> (L,B,D)
        
        lstm_input = nn.utils.rnn.pack_padded_sequence(
            batch_emb, seq_lengths.cpu().numpy(), batch_first=False, enforce_sorted=False)

        packed_output, (h_t, h_c) = self.lstm(lstm_input,)  # (h_t, h_c))
#         output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, padding_value=-91)

        h_t = F.relu(h_t) #
        return self.out(h_t)


class ToBERT_Model(nn.Module):
    """ Make a transformer over a fine tuned bert model. Parameters
    __________
    bertFineTuned: BertModel
        A bert fine tuned instance
    """

    def __init__(self, bertFineTuned, chunks, num_classes = 2):

        super(ToBERT_Model, self).__init__()
        self.bertFineTuned = bertFineTuned()
        self.transformer = nn.Transformer(d_model = 512, nhead= 2, dropout= 0.1)
        self.out = nn.Softmax(dim= num_classes)
        self.chunks = chunks 
        

    def forward(self, ids, mask, token_type_ids, lengt):
        """ Define how to performed each call
        Parameters
        __________
        ids: array
            -
        mask: array
            - 
        token_type_ids: array
            -
        lengt: int
            -
        Returns:
        _______
        -
        """
        self.inp = []; pos_encoding = []

        for i,j in enumerate(self.chunks): 

            ids, mask, token_type_ids = i
            x = self.bertFineTuned(ids, attention_mask=mask, token_type_ids=token_type_ids)
            self.inp.append(x); pos_encoding.append(j)

        temp_out = self.transformer(x)[:,-1,:]
        return self.out(temp_out)

#My own code

class AssessData():
    def __init__(self, listofstrings: list):
        #Processes list of strings, 
        self._content = {i: string for i,string in enumerate(listofstrings)}
        self._threshold = {'longformer': 4096, 'BERT': 512, "Short" : 300 }
  

    def _string_length(self, text):
        """Processes the number of words in one text string """
        if isinstance(text, list):
            print("This is a list, input string")
        else:
            return len(word_tokenize(text))

    def _get_string_length(self):
        """Processes the number of words in a list of text strings"""
        self._all_length = {i: self._string_length(item) for i,item in self._content.items()}
        return self._all_length
            
    def _create_distribution(self): 
        "Creates a distribution based on the number of words in text strings"
        
        self._get_string_length()
        self._distribution = {"Too Long" : [],"Long": [], "BERT": [], "Short":[]}


        for index,length in self._all_length.items():
            if length > self._threshold['longformer']:
                self._distribution["Too Long"].append(index)
            elif length >= self._threshold['BERT'] and length < self._threshold['longformer']:
                self._distribution["Long"].append(index)
            elif length < self._threshold['BERT'] and length >= self._threshold['Short']:
                self._distribution["BERT"].append(index)
            else:
                self._distribution["Short"].append(index)
        
        return self._distribution


    def _extract(self, key = "Longformers"):
        """Extracts text strings suitable for particular classification technique"""

        if key.lower() in self._distribution.keys():
            indexes = self._distribution[key]
            return [self._content[i] for i in indexes]
        else:
            if self._distribution is None: 
                self._create_distribution()
                self._extract(key = key)
            else:
                print("Select from given distribution {}".format(self._distribution.keys()))

    def _visualise(self):
        """Displays a plot of count of number of words versus the categories created"""
        
        values = {k:len(v) for k, v in self._distribution.items()}
        print(values)
        plt.bar(values.keys(),values.values())
        plt.xlabel("Category")
        plt.ylabel("Count")

        plt.show()

    def _chunk(self, words_per_segment, overlap = 50):
        
        """Chunking for input into BERT, length of input 
        must be less than 512 with overlap of 50 tokens"""

        self._words_per_segment = words_per_segment
        self._chunks = {}
        self._overlap = overlap

        if self._get_string_length:
            for index,length in self._all_length.items():
                temp = self._content.get(index)

                if length > self._words_per_segment:

                    segments = length + self._overlap//self._words_per_segment

                    pointer = self._words_per_segment
                    temp_chunked_content = temp[:pointer]; count = 0

                    while count <= segments:
                        temp_chunked_content.append(temp[pointer - overlap : count*pointer + overlap])
                        count +=1

                else: 
                    temp_chunked_content = [temp[:]]

                self._chunks[index] = temp_chunked_content
        else:
            self._get_string_length()
            self._chunk()

        return self._chunks
        

class PrepareCorpus():

    def __init__(self, path):
        self._path = path


    def _corpus(self):
        """BY date"""
        self._corpus = {}; temp = []; 

        for topic in os.listdir(self._path):
            subfolder = self._path + '/' + topic
            current = []
            for doc in os.listdir(subfolder):
                file = subfolder + '/' + doc
                with open(file, 'r', encoding='utf-8', errors= 'ignore') as t:
                    temp = " ".join(t.readlines())
                    current.append(temp)
            self._corpus[topic] = current
        return self._corpus




#Check string length of all entries
#Filter based on certain string length(100, 300, >512, >1024)
#Chunk strings in a number of segments 
#Chunk strings with a maximum token length 
#Visualise Dataset(token length(100, 300, >512), elements per class )

