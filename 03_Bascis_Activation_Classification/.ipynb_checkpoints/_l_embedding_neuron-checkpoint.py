import pandas as pd
import polars as pl
import time
import torch
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from torch import nn
from transformers import BartTokenizer, BartModel #tokenizing the input then pushing it into the model to generate embeddings

custom_messages =  [
    "We have a new product release, do you want to buy it?",
    "Winner! Great deal, call us to get a free product!",
    "Hey Tomorrow is my birthday, do you come to the party?",
]

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartModel.from_pretrained("facebook/bart-base")

def convert_to_embeddings(messages):
    embeddings_list = []    
    for message in tqdm(messages):
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
            embeddings = embeddings.reshape(-1)
            embeddings_list.append(embeddings)

    return torch.stack(embeddings_list) #now it is 2 dim and again a matrix

def evaluate_model(X, y):
    model.eval()
    with torch.no_grad():
        y_pred = nn.functional.sigmoid(model(X)) > 0.5    
        print("accuracy:", (y_pred == y).type(torch.float32).mean())
        print("sensitivity:", (y_pred[y==1] == y[y==1]).type(torch.float32).mean())
        print("specificity:", (y_pred[y==0] == y[y==0]).type(torch.float32).mean())
        print("precision:", (y_pred[y_pred==1] == y[y_pred==1]).type(torch.float32).mean())

df = pd.read_csv("./data/SMSSpamCollection", sep="\t", names=["type", "message"])
df = pl.DataFrame(df)
df = (
    df
    .with_columns(
        spam=pl.when(pl.col("type").eq("spam")).then(pl.lit(1)).otherwise(0)
    )
    .select("message", "spam")
    .with_row_index(name="index")
)

df_train = df.sample(fraction=0.8, seed=0)
indezes = list(df_train["index"])
df_val = df.filter(~pl.col("index").is_in(indezes))

X_train = convert_to_embeddings(df_train["message"].to_list())
X_val = convert_to_embeddings(df_val["message"].to_list())

print(X_train.shape)
print(X_val.shape)
y_train = torch.tensor(df_train["spam"], dtype=torch.float32).reshape(-1, 1)
y_val = torch.tensor(df_val["spam"], dtype=torch.float32).reshape(-1, 1)
print(y_train.shape)
print(y_val.shape)


model = nn.Linear(768, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

t_start = time.time()
for i in range(0, 15000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(loss)
t_end = time.time()

print(t_end-t_start, "seconds")

evaluate_model(X_train, y_train)
evaluate_model(X_val, y_val)

X_custom = convert_to_embeddings(custom_messages)

model.eval()
with torch.no_grad():
    pred = nn.functional.sigmoid(model(X_custom))
    print(pred)