# we need a split to not overfit and memorize the data for example 
# maybe in the training data there was one word very often that indicated spam
# if we only learn that this word was an identifier and a new spam with a completely new word comes in, than this model would fail, because it only learned this keyword, and probably not a generalited pattern, this is also true for a handful of words. while this maybe true, we want to learn general patterns, to identify spam

# val is for model architecture, when to stop training, and when model perf. no longer improves probably, or I tested enough hyperparameters, so to determine stop training point, tune hyperparameters etc.

import pandas as pd
import polars as pl
import time
import torch
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from torch import nn
from transformers import BartTokenizer, BartModel #tokenizing the input then pushing it into the model to generate embeddings

#new data with custom messages
custom_messages =  [
    "We have a new product release, do you want to buy it?",
    "Winner! Great deal, call us to get a free product!",
    "Hey Tomorrow is my birthday, do you come to the party?",
]

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartModel.from_pretrained("facebook/bart-base")

def convert_to_embeddings(messages):
    embeddings_list = []
    
    for message in tqdm(custom_messages):
        out_trunc = tokenizer(
            message, 
            padding=True, 
            max_length=500, 
            truncation=True, 
            return_tensors="pt"
        )
        with torch.no_grad():
            bart_model.eval()
            pred = bart_model(**out_trunc) 
            embeddings = pred.last_hidden_state.mean(dim=1)
            print(embeddings.shape)
            #and whe can see how the first dim changes as we now look at the individual message and not all messages at once
            #however later we want to combine these embeddings to have 768 values and not these individual vectors, we want to have a matrix/tensor that we can work with that contains all messages at once, we did this here for educational pruposes, so we want to stack vectors, so we want to have the dim 3, 768 back as we had in file _j_
            embeddings = embeddings.reshape(-1)
            print(embeddings.shape) #they are now a simple vector
            embeddings_list.append(embeddings)
            
            return torch.stack(embeddings_list) #now it is 2 dim and again a matrix

print(X.shape)