import torch
from torch import nn, optim

#key idea of learning is
#- no equation system must be solved
#- parameters are adjusted iteratively
#- good if there is no ideal solution or we have a lot of parameters
    
#think of it like a volume dial of a DJ, if volume is too high you scale it down,
#if volume is too low you scale it up, every dial is a parameter


# the greater the impact of a parameter the greater we need to adjust it (not in steps)

#The gradient tells us how much a parameter affects the error
#shows the direction and rate change of the error with respect to a paramter
#we calculate the gradient for each parameter we want to optimize

# lr is the step size that controls how much each weight is updated
# doing a small adjustment doing calculations again and checking if it worked or not

#weight new = weight old - learning_rate*gradient
#since the bias is also a parameter just like the weights, we would do this also there


# the process of updating weights is gradient descent, gradient descent is the process and gradient is the derivative


#input temp
#output Fahrenheit
X1 = torch.tensor([[10]], dtype=torch.float32)
y1 = torch.tensor([[50]], dtype=torch.float32)

X2 = torch.tensor([[37.78]], dtype=torch.float32)
y2 = torch.tensor([[100.0]], dtype=torch.float32)

model = nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

#01Training First data Point
#optimizer.zero_grad() 
#outputs = model(X1)
#loss = loss_fn(outputs, y1)
#loss.backward() 
#optimizer.step() 
#
#optimizer.zero_grad()
#outputs = model(X2)
#loss = loss_fn(outputs, y2)
#loss.backward() 
#optimizer.step() 
#
#y_pred = model(X1)
#print("Prediction:", y_pred)
#print ("manually doing MSE on one observation:", loss_fn(torch.tensor([[5]], dtype=torch.float32), torch.tensor([[10]], dtype=torch.float32)))

#02Training with loop
#for i in range(10000):
#    optimizer.zero_grad() 
#    outputs = model(X1)
#    loss = loss_fn(outputs, y1)
#    loss.backward() 
#    optimizer.step() 
#    
#    optimizer.zero_grad()
#    outputs = model(X2)
#    loss = loss_fn(outputs, y2)
#    loss.backward() 
#    optimizer.step() 

    # gives us Prediction: tensor([[nan]], grad_fn=<AddmmBackward0>) numerical problem let us reduce the learning rate

#03Training with smaller learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001) 
for i in range(40000):
    optimizer.zero_grad() 
    outputs = model(X1)
    loss = loss_fn(outputs, y1)
    loss.backward() 
    optimizer.step() 
    
    optimizer.zero_grad()
    outputs = model(X2)
    loss = loss_fn(outputs, y2)
    loss.backward() 
    optimizer.step() 

    if i % 100 == 0: # every 100 iterations show update
        print(model.bias)
        print(model.weight)


# and this gives: Prediction: tensor([[41.8954]], grad_fn=<AddmmBackward0>)
# or something else, and you will see how the model updates the bias toward 32 and the weight toward 1.8
# could do it with 50000, then may be it may converge

#after 40000 iterations:
#tensor([31.4899], requires_grad=True)
#Parameter containing:
#tensor([[1.8152]], requires_grad=True)
#Prediction: tensor([[49.6454]], grad_fn=<AddmmBackward0>)

y_pred = model(X1)
print("Prediction:", y_pred)
print ("manually doing MSE on one observation:", loss_fn(torch.tensor([[5]], dtype=torch.float32), torch.tensor([[10]], dtype=torch.float32)))
