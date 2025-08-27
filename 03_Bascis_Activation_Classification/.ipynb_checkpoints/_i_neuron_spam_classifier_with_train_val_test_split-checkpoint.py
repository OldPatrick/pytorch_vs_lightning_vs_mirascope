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
print(df_val.shape)
print(df_train.shape)

cv = CountVectorizer(max_features=3000) 
# reduce features, if transofrmation to dense and then to tensor is too slow
messages_train = cv.fit_transform(df_train["message"]) #here learning
messages_val = cv.transform(df_val["message"]) #here no longer learning!!! we learn on train, here we just transform !!!
# if a word is not in train and be ignored in val, that is how it qwould work also in test and in reality! Good!
# both steps together
# here it is essential that the count vectorizer only learns and see vocabulary from the training data

# X = messages does not work, sparse matricess can not be converted directly into a tensor
X_train = torch.tensor(messages_train.todense(), dtype=torch.float32)
y_train = torch.tensor(df_train["spam"], dtype=torch.float32)

print(X_train.shape)
print(y_train.shape)
#with reshaping this would be needed to do for every column, and this is quite expensive, so we need to reshape it, to bring it into this correct order
y_train = torch.tensor(df_train["spam"], dtype=torch.float32).reshape(-1, 1)

print(X_train.shape)
print(y_train.shape)

X_val = torch.tensor(messages_val.todense(), dtype=torch.float32)
y_val = torch.tensor(df_val["spam"], dtype=torch.float32).reshape(-1, 1)

print(X_val.shape)
print(y_val.shape)

model = nn.Linear(3000, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()
# loss contains internally a sigmoid, so we need not to apply it here to our Neuron directly, later maybe we need it in a FCN or so, but we will need it for the predictions
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


def evaluate_model(X, y):

    model.eval()
    with torch.no_grad():
        y_pred = nn.functional.sigmoid(model(X))
        print(y_pred)
        print(y_pred.min())
        print(y_pred.max())
        y_pred = nn.functional.sigmoid(model(X)) > 0.5
    
        print("accuracy:", (y_pred == y).type(torch.float32).mean())
        print("sensitivity:", (y_pred[y==1] == y[y==1]).type(torch.float32).mean())
        print("specificity:", (y_pred[y==0] == y[y==0]).type(torch.float32).mean())
              # where the prediction is 1 and the reality is also 1 so at what percentage did I correctly identified spam where it was really spam 
        print("precision:", (y_pred[y_pred==1] == y[y_pred==1]).type(torch.float32).mean())
    
        # specificity for example if legal says all business emails need to reach us
evaluate_model(X_train, y_train)
evaluate_model(X_val, y_val)

# if val too far away, limit training time early stop, change learning rate #to smaller or higher or less complexity with less features, but it can get worse, we could also give more complexity for now, we want to optimize val set but not the train
# with more features e.g. 3000 instead of 1000 the precisiosn for example got better


#new data with custom messages
custom_messages =  [
    "We have a new product release, do you want to buy it?",
    "Winner! Great deal, call us to get a free product!",
    "Hey Tomorrow is my birthday, do you come to the party?",
]


custom_msg_transformed = cv.transform(custom_messages)
X_custom = torch.tensor(custom_msg_transformed.todense(), dtype=torch.float32)

model.eval()
with torch.no_grad():
    pred = nn.functional.sigmoid(model(X_custom))
    print (pred)