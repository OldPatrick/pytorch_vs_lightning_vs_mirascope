import polars as pl
import torch
from polars import selectors as cs
from torch import nn

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
print(X.shape)

model = nn.Linear(2, 1)
#two inputs and one output
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
prediction = model(X)
print(prediction)
#we can not use the columns/series from the data frame directly for the model
#We need ofc to convert it to tensors (sometimes ML models also need arrays, or a conversion from polars to pandas)