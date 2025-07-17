import torch
from torch import nn
#https://docs.pytorch.org/docs/stable/nn.html

#This basically tells you what sort of layers you want to use in your model
# write floating numbers overall otherwise you will have mixed dtype which is not liked by tensors?
#float 32 32bits with accuracy
# changing the dtype of a tensor goes like below, but be aware that operations between differing dtypes in matrix multiplication also happens between float32 and float64

b = torch.tensor(32.0)
w1 = torch.tensor(1.8)

# a column reflects a feature here
X1 = torch.tensor(
    [
        [10.0], 
        [38.0], 
        [100.0], 
        [150.0]
    ], dtype=torch.float32
)

print(X1.dtype)
X1 = X1.type(torch.float64)
print(X1.dtype)
X1 = X1.type(torch.float32)
print(X1.dtype)

model = nn.Linear(1, 1)

print(model)
# shows us that 1 feature gets in, 1 output comes out, and we want to have a bias activated just like in the pred fucntion below:
y_pred_old = 1 * b + X1 * w1 #cpu advantage not from gpu, vectorised extension
y_pred = model(X1)
# if we now use X with its tensor for this linear model, we are already gettinga prediction, and not only 1 but 4, for all inputs

# used model on 4 data points
print("Usage of Linear(1,1):", y_pred)

# but if we look at the weight and the bias of the model, we see it is not what we defined above, they were init randomly:
# But this is how deep learning also works in generally
print("But if we look at the weight and the bias of the model, we see it is not what we defined above, they were init randomly:")
print("bias:", model.bias)
print("weights:", model.weight)

# Looking at the outputs:
#Usage of Linear(1,1): tensor([[ 5.2574],
#        [18.2058],
#        [46.8773],
#        [69.9995]], grad_fn=<AddmmBackward0>)
#But if we look at the weight and the bias of the model, we see it is not what we defined above, they were init randomly:
#bias: Parameter containing:
#tensor([0.6329], requires_grad=True)
#weights: Parameter containing:
#tensor([[0.4624]], requires_grad=True)
# 0.6329 + 0.4624*10.0 = 5.25 (first value, bias and weight estimated)
