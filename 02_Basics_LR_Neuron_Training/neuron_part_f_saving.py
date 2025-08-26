import polars as pl
import torch
import os
from polars import selectors as cs
from torch import nn
import altair as alt
alt.data_transformers.enable("vegafusion")

if not os.path.isdir("./model"):
    os.mkdir("./model")

df = pl.read_csv("used_cars.csv", separator=",")

df = (
    df
    .with_columns(
        pl.col("price")
        .str.replace(r'\$', "")
        .str.replace(r',', '')
        .str.replace(r',', '')
        .cast(pl.Int64)   
    )
    .with_columns(
        pl.col('milage')
        .str.replace(r' mi.', '').str.replace(r',', '')
        .str.replace(r',', '')
        .cast(pl.Int64)  
    ) 
    .with_columns(
        age=pl.col("model_year").max()-pl.col("model_year")
    )

    .select("milage", "price", "age")
)

X = torch.column_stack([
        torch.tensor(df["age"], dtype=torch.float32),
        torch.tensor(df["milage"], dtype=torch.float32)
    ]
)
#allows multiple columns to be stacked in a matrix compatible format

y = torch.tensor(df["price"], dtype=torch.float32).reshape((-1, 1)) #-1 pick amount of rows automatically
y_mean = y.mean()
y_std = y.std()
y = ((y-y_mean)/y_std)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X-X_mean) / X_std

torch.save(X_mean, "./model/X_mean.pt")
torch.save(X_std, "./model/X_std.pt")
torch.save(y_mean, "./model/y_mean.pt")
torch.save(y_std, "./model/y_std.pt")

model = nn.Linear(2, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001) 
losses =  []

for i in range(10000):
    optimizer.zero_grad() 
    outputs = model(X)
    loss = loss_fn(outputs, y)
    losses.append(loss.item())
    loss.backward() 
    optimizer.step() 

    if i % 100 == 0: # every 100 iterations show update
        #print(model.bias)
        #print(model.weight)
        print(loss)

losses_data = pl.DataFrame(losses)
losses_data = losses_data.with_row_index(name="my_index")
losses_data.plot.line(y="column_0", x="my_index")

torch.save(model.state_dict(), "./model/model.pt")