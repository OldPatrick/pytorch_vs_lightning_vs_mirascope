import polars as pl
import torch
from polars import selectors as cs
from torch import nn

#import sys

# a sudden price change from 5000 to 80000 dollars would need a big step in the learning procedure and it doesnt mean that the
# data is wrong as an old car may have that price and a new one also the biggr one. but how do we account fo these big changes 
# how can the model do these big steps without again creating big numbers and crushing?
# Gradient explosion with big numbers or Gradient diminishing with very small values and no learning    
# so large gradients (Ableitung) cause drastic weight updates and too large weights

# So we need to normalize data to stabilize learning, ofc this can be due to outliers that we need to normalize but not necessarily
#put predictions into smaller range

# z score 0 of the data would be mean and std = 1

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

print(y) #could also use 4009 the max length rows

#sys.exit() # to stop the python program here

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

print("#####")
print((prediction2*y_std)+y_mean)
#predition outputs are now also normalized
#so we would need to turn it back
#so multipy by std and add the mean

