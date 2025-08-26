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
#allows multiple columns to be stacked in a matrix compatible format

y = torch.tensor(df["price"], dtype=torch.float32).reshape((-1, 1)) #-1 pick amount of rows automatically
print(y) #could also use 4009 the max length rows

model = nn.Linear(2, 1)
#two inputs and one output
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00000000001) #other bigger lrs will show nan as the learning rate, but this is already too small
# if I make he learning rate quite small like 0.00000001 maybe something will show up, but the learning steps are o small nothing will be learned
# at the beginning there ma be some learning but

#remember low learning rate results in small learning steps, only slow or minimal changes in the loss or even stalled learning process

#remember high learning rate or too high causes the model to make extreme steps and adjustments to weights
#leading to outgrowing numbers the bounds, the model will not converge
for i in range(40000):
    optimizer.zero_grad() 
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward() 
    optimizer.step() 

    if i % 100 == 0: # every 100 iterations show update
        print(model.bias)
        print(model.weight)

prediction = model(X)
print(prediction)
#we can not use the columns/series from the data frame directly for the model
#We need ofc to convert it to tensors (sometimes ML models also need arrays, or a conversion from polars to pandas)

#printing my own prediction 
prediction2 = model(torch.tensor([
    [5, 10000], 
], dtype=torch.float32))
print(prediction2)

#printing my own prediction 
prediction3 = model(torch.tensor([
    [5, 10000],
    [10, 6000] 
], dtype=torch.float32))
print(prediction3)

#older and less miles you get less money,
#but if I put age 5 and miles 20000, I get that it will be more valuable

prediction4 = model(torch.tensor([
    [5, 30000], 
], dtype=torch.float32))
print(prediction4)
# so we didnt really learn the underlying structure of the general relationship, our model is worthless or there i some meaningful stuff but
# maybe there is some special car in there that distortes this relationship we have thought like luxury cars? but than only age and not milage should contribute to higher prices