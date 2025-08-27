# we need a split to not overfit and memorize the data for example 
# maybe in the training data there was one word very often that indicated spam
# if we only learn that this word was an identifier and a new spam with a completely new word comes in, than this model would fail, because it only learned this keyword, and probably not a generalited pattern, this is also true for a handful of words. while this maybe true, we want to learn general patterns, to identify spam

# val is for model architecture, when to stop training, and when model perf. no longer improves probably, or I tested enough hyperparameters, so to determine stop training point, tune hyperparameters etc.

import pandas as pd
import polars as pl
import time
import torch

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
out = tokenizer(custom_messages)
print(out) #this translates the meaning of the text to tokens
# a little bit like https://tiktokenizer.vercel.app/ but without start stop tokens?
# the amount of tokens per message are now different which would be very tricky to train since the input differs between messages

out_pad = tokenizer(custom_messages, padding=True)
#padding true puts filler values into the vectors to make them the same length as the input needs to be for training
#the attention mask can be used for multiplication to later ignore specific tokens of shorter terms
print(out_pad)

out_trunc = tokenizer(custom_messages, padding=True, max_length=500, truncation=True, return_tensors="pt")
#max length limits the message and truncation cuts them off, and return tensors brings them in the correct shape
print(out_trunc)

bart_model = BartModel.from_pretrained("facebook/bart-base") # fetches weights etc., 500-600MB
with torch.no_grad():
    bart_model.eval()
    pred = bart_model(**out_trunc) #could also be written like bart_model(input_ids=out_trunc["input_ids"], attention_mask=out_trunc["attention_mask"]), so we are using the dict keys directly as parameters and pass the dict, this shrunks the code extremly
    print(pred)
    pred_written = bart_model(input_ids=out_trunc["input_ids"], attention_mask=out_trunc["attention_mask"])
    print(pred_written)

    print("########")
    print(pred_written.last_hidden_state.shape) # 3 dimensional tensor , 3 messages basically, LLM took 16 steps, 768 output values , in each step 768 values

    embeddings = pred.last_hidden_state.mean(dim=1)
    print(embeddings.shape)
