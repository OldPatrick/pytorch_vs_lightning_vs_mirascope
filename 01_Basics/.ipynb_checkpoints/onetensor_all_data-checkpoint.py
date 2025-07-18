import torch
from torch import nn, optim

#input temp
#output Fahrenheit
X = torch.tensor(
    [
        [10],
        [37.78]
    ], 
    dtype=torch.float32
)


y = torch.tensor(
    [
        [50],
        [100.0]
    ],
    dtype=torch.float32
)

model = nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001) 

for i in range(40000):
    optimizer.zero_grad() 
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward() 
    optimizer.step() 

    if i % 100 == 0: # every 100 iterations show update
        print(model.bias)
        print(model.weight)

y_pred = model(X)
print("Prediction:", y_pred)
