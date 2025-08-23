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

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X-X_mean) / X_std

model = nn.Linear(2, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001) 
# we had to use a very low learning rate to get some learning, beause values where to large,
# now with standardized input and output we should be able to decrease the learning rate again
# bringing all input features to the same scale allows us to take a uniform learning rate otherwise, 
# would need depending on the scale a smaller or bigger because of the steps we are doing
# so uniforming lr through normalizing
for i in range(40000):
    optimizer.zero_grad() 
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward() 
    optimizer.step() 

    if i % 100 == 0: # every 100 iterations show update
        #print(model.bias)
        #print(model.weight)
        print(loss)
prediction = model(X)
print(prediction)
#we can not use the columns/series from the data frame directly for the model
#We need ofc to convert it to tensors (sometimes ML models also need arrays, or a conversion from polars to pandas)

#printing my own prediction 

#but X data also needs to be scaled first:

my_X = torch.tensor([
    [5, 10000], 
    [2, 10000],
    [2, 30000] 
], dtype=torch.float32)
     

# miles have more influence than age, and ofc this is across all brands, but this is a basic prediction
# but ofc the relationship is linear and since a car losses in the first year more than in next years, this may be not a correct grasp of the underlying relationship
prediction2 = model((my_X-X_mean)/X_std)
print(prediction2)

print("#####")

print((prediction2*y_std)+y_mean)
#predition outputs are now also normalized
#so we would need to turn it back
#so multipy by std and add the mean

#yet the price is completely shit as nothing is really learned still and the normalization only helps in not exploding or vanishing gradients
