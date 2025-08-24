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
    #Gradients accumulate by default
    #When you call .backward(), PyTorch adds the computed gradients to the .grad attribute of each parameter. It does not overwrite them.
    #This is useful for some advanced techniques (like gradient accumulation over multiple batches), but in standard training, you want to compute gradients fresh for each batch.
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward() 
    optimizer.step() 

    if i % 100 == 0: # every 100 iterations show update
        print(model.bias)
        print(model.weight)
    
print("-----")
measurements = torch.tensor(
    [
        [37.78]
    ], 
    dtype=torch.float32
)
#turn off features specific for training and bring it into evluation mode:
model.eval()
with torch.no_grad() #tells pytorch we do not need to keep track of the gradients
#its not needed when we are finished with training
    prediction = model(measurements)
    print("Prediction:", prediction)
