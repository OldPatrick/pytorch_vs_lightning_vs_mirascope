import polars as pl
import torch
from polars import selectors as cs
from torch import nn
import altair as alt
alt.data_transformers.enable("vegafusion")

df = pl.read_csv("used_cars.csv", separator=",")
df2 = pl.read_csv("used_cars.csv", separator=",")
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
    .with_columns(
        accident_new=pl.when(pl.col("accident").eq("None reported")).then(pl.lit(0)).otherwise(1)
    )
    .select("milage", "price", "age", "accident_new")
)

X = torch.column_stack([
        torch.tensor(df["age"], dtype=torch.float32),
        torch.tensor(df["milage"], dtype=torch.float32),
        torch.tensor(df["accident_new"], dtype=torch.float32)
    ]
)

y = torch.tensor(df["price"], dtype=torch.float32).reshape((-1, 1)) #-1 pick amount of rows automatically
y_mean = y.mean()
y_std = y.std()

y = ((y-y_mean)/y_std)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X-X_mean) / X_std

model = nn.Linear(3, 1)
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
        print(loss)
prediction = model(X)
print(prediction)
losses_data = pl.DataFrame(losses)
losses_data = losses_data.with_row_index(name="my_index")
losses_data.plot.line(y="column_0", x="my_index")

my_X = torch.tensor([
    [5, 10000, 0], 
    [2, 10000, 0],
    [2, 30000, 1] 
], dtype=torch.float32)
# remember, the accident varaiable helps improving but we do not know how many accidents, which aread the impact was and how severe, as well as if the accident was fixed.
# all of this information would be beneficial but is not available.

prediction2 = model((my_X-X_mean)/X_std)
print(prediction2)

print("#####")

print((prediction2*y_std)+y_mean)

