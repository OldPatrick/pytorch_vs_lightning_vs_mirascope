# in file d we did not mapped the output to a range between 0 and 1 indicating probas (which they still probably arent?), because it can change depending of the starting point of the NN, and since it underlies so many changes, probability feels like an ultimate truth, when it can change so rapidly, however for this I use the sigmoid activation function

#1 / 1+e^-x

# with MSE Loss the error gets too small and gradients become small as well
# The error gets so small close to 0, but very small, and we wont make much updates
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
loss_fn = torch.nn.BCEWithLogitsLoss()
# loss contains internally a sigmoid, so we need not to apply it here to our Neuron directly, later maybe we need it in a FCN or so, but we will need it for the predictions
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

t_start = time.time()
for i in range(0, 15000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(loss)
t_end = time.time()
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
    
print(t_end-t_start, "seconds")
# But the predictions cant be used like this -0.0140 or something like this. There may be 
# values even larger than 1, the problem is they need to be crunshed into values of 0 and 1
# it even feels wrong to call this probability



