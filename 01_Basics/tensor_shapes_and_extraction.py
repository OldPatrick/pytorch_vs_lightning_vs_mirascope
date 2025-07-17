import torch

b = torch.tensor(32)
w1 = torch.tensor(1.8)

# calculation will be performed 4 times, without a loop, the big advantage of a tensor
# however a [10] would also be considered a vector but with just one value, the square brackets create the vector

X1 = torch.tensor([10, 38, 100, 150])
X2 = torch.tensor([50])
X3 = torch.tensor(25)

y_pred = 1 * b + X1 * w1 #cpu advantage not from gpu, vectorised extension


print("with .shape")
print(X1.shape, "vector with 4 elements")
print(X2.shape, "vector with 1 element")
print(X3.shape, "scalar")
print("")
print("with .size()")
print(X1.size())
print(X2.size())
print(X3.size())
print("")
print("access the first element in a tensor")
print(y_pred[1])
print("access the first element in a tensor getting it out of the box")
print(y_pred[1].item())