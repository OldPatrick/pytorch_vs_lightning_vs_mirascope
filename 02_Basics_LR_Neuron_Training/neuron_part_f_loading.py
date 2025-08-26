import polars as pl
import torch
from polars import selectors as cs
from torch import nn
import altair as alt
alt.data_transformers.enable("vegafusion")


X_mean = torch.load("./model/X_mean.pt", weights_only=True)
X_std = torch.load("./model/X_std.pt", weights_only=True)
y_mean = torch.load("./model/y_mean.pt", weights_only=True)
y_std = torch.load("./model/y_std.pt", weights_only=True)

my_X = torch.tensor([
    [5, 10000], 
    [2, 10000],
    [2, 30000] 
], dtype=torch.float32)
     
model = nn.Linear(2, 1)
model.load_state_dict(torch.load("./model/model.pt"))

model.eval()
# Not keeping tracks of over stuff
with torch.no_grad():
    prediction2 = model((my_X-X_mean)/X_std)
    print(prediction2)

    print("#####")

    print((prediction2*y_std)+y_mean)
