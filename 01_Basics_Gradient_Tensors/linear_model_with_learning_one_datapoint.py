import torch
from torch import nn, optim

# key idea of learning is
# - no equation system must be solved
# - parameters are adjusted iteratively
# - good if there is no ideal solution or we have a lot of parameters

# think of it like a volume dial of a DJ, if volume is too high you scale it down,
# if volume is too low you scale it up, every dial is a parameter


# the greater the impact of a parameter the greater we need to adjust it (not in steps)

# The gradient tells us how much a parameter affects the error
# shows the direction and rate change of the error with respect to a paramter
# we calculate the gradient for each parameter we want to optimize

# lr is the step size that controls how much each weight is updated
# doing a small adjustment doing calculations again and checking if it worked or not

# weight new = weight old - learning_rate*gradient
# since the bias is also a parameter just like the weights, we would do this also there


# the process of updating weights is gradient descent, gradient descent is the process and gradient is the derivative


#input temp
#output Fahrenheit
X1 = torch.tensor([[10]], dtype=torch.float32)
y1 = torch.tensor([[50]], dtype=torch.float32)

model = nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
for i in model.parameters():
    print(i)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


#Training Phase 1#Epoch..or better One Datapoint
print("Weight before:", model.weight)
print("Bias before:", model.bias)
optimizer.zero_grad() #This is the key. It sets param.grad back to None. It does not erase the fact that the weights were updated based on X1. The updated weights are already stored in the model. This call just clears the temporary storage (.grad) for the next gradient calculation.
#it is ::not:: erasing the weights

outputs = model(X1)
loss = loss_fn(outputs, y1)
loss.backward() #creating the gradient and pushing errors back, knows how much a parameter would need to go back
optimizer.step() #actually perform now a new step
print("Weight after:",model.weight)
print("Bias after:", model.bias)
print("loss", loss)
#And as described since the weight is more influential the bias will no be updated that much after the epoch, as the weight is more influential

y_pred = model(X1)
print("Prediction:", y_pred)
print ("manually doing MSE on one observation:", loss_fn(torch.tensor([[5]], dtype=torch.float32), torch.tensor([[10]], dtype=torch.float32)))
