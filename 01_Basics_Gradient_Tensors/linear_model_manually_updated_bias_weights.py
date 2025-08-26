import torch
from torch import nn


w1 = torch.tensor(1.8)
X1 = torch.tensor(
    [
        [10.0], 
        [38.0], 
        [100.0], 
        [150.0]
    ],
    dtype=torch.float32
)

model = nn.Linear(1, 1)

model.bias = nn.Parameter(  #takes a vector
    torch.tensor([32.0], dtype=torch.float32)
)
# only floating point numbers accept gradients, and also converting an int tensor 10.000 times would be inefficient

model.weight = nn.Parameter(
    torch.tensor([[1.8]], dtype=torch.float32) #takes a matrix
)

print(model)
y_pred = model(X1)

#created a neuron
print("Usage of Linear(1,1):", y_pred)
print("But if we look at the weight and the bias of the model, we see it is not what we defined above, they were init randomly:")
print("bias:", model.bias)
print("weights:", model.weight)

