import torch

b = torch.tensor(32)
w1 = torch.tensor(1.8)

# calculation will be performed 4 times, without a loop, the big advantage of a tensor
# however a [10] would also be considered a vector but with just one value, the square brackets create the vector

X1 = torch.tensor([10, 38, 100, 150])

y_pred = 1 * b + X1 * w1 #cpu advantage not from gpu, vectorised extension
print("0-dimensional tensor:", y_pred)
