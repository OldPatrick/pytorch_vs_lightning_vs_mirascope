import pandas as pd
import polars as pl
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
)
cv = CountVectorizer(max_features=1000) # reduce features if transofrmation to dense and then to tensor is too slow
messages = cv.fit_transform(df["message"]) #both steps together

#X = messages
# sparse matricess can not be converted directly into a tensor
X = torch.tensor(messages.todense(), dtype=torch.float32)
y = torch.tensor(df["spam"], dtype=torch.float32)

print(X.shape)
print(y.shape)
#with reshaping this would be needed to do for every column, and this is quite expensive, so we need to reshape it, to bring it into this correct order
y = torch.tensor(df["spam"], dtype=torch.float32).reshape(-1, 1)

print(X.shape)
print(y.shape)

model = nn.Linear(1000, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for i in range(0, 10000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(loss)
        
model.eval()
with torch.no_grad():
    y_pred = model(X)
    print(y_pred)
    print(y_pred.min())
    print(y_pred.max())

# But the predictions cant be used like this -0.0140 or something like this. There may be 
# values even larger than 1, the problem is they need to be crunshed into values of 0 and 1
# it even feels wrong to call this probability