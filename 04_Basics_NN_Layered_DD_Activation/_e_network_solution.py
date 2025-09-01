#If training a neuron with 500,000 iterations takes too long, try reducing the number of iterations to 250,000 and increasing the learning rate, for example, to 0.025.

#When working with networks, it is possible to get stuck in local minima for several iterations or more due to the random initialization of weights and biases. 
#If the loss does not decrease for a significant portion of the iterations, rerunning the model might help. This issue depends on factors such as the number of training iterations, 
#the learning rate, activation functions and the optimizer. To resolve it, experiment with these parameters until a working solution is found.

#For this specific dataset, a good configuration to try is 300,000 iterations, a learning rate of 0.01, the Adam optimizer and the ReLU activation function.

#Adding mini-batches to the training process can also significantly reduce the time required compared to the initial setup.

#If you are not satisfied with the accuracy and observe that the loss is still decreasing, you can train the model for more iterations. 
#The exact number will depend on how long you are willing to train and the level of accuracy you aim to achieve.



## 2 inputs, 10 hidden neurons, 1 output


import torch
import polars as pl
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))


def model_training(X, y):
    for _ in range (0, 200000):
        optimizer.zero_grad() #delete gradients to not accumulate so many values
        outputs = hidden_model(X) #hidden_model applied to input
        outputs = nn.functional.sigmoid(outputs)
        outputs = output_model(outputs) # and finally the model on the remaining stuff
        loss = loss_fn(outputs, y) #between prediction and real one
        loss.backward() #backpropagation
        optimizer.step() #(into the right/next direction)

        if _ % 1000 == 0:
            print(loss)
    
df = pl.read_csv("student_exam_data.csv")
print(df.head)
df = (
    df
    .rename({
        "Study Hours": "study_hours",
        "Previous Exam Score" : "old_score",
        "Pass/Fail": "passed"
    })
    .with_row_index(name="index")
)

df_test = df.sample(fraction=0.1, seed=0)
df_remain = df.filter(~pl.col("index").is_in(df_test["index"]))

print(df_test.shape, df_remain.shape)
X = df_remain.drop("passed", "index")
y = df_remain["passed"]

X_train, X_val, y_train, y_val = train_test_split(
X, y, test_size=0.33, random_state=0)

X_train = torch.tensor(X_train.rows(), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_val = torch.tensor(X_val.rows(), dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

X_test = torch.tensor(df_test.drop("index", "passed").rows(), dtype=torch.float32)
y_test = torch.tensor(df_test["passed"], dtype=torch.float32).reshape(-1, 1)

hidden_model = nn.Linear(2, 10)
output_model = nn.Linear(10, 1)
loss_fn = nn.BCEWithLogitsLoss()
print(list(hidden_model.parameters())) # I ned to print the generator object in a list otherwise I need a for loop for showing)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
#problem is optimizer needs to train parameters of the hidden layer and the output layer....
params = list(hidden_model.parameters()) + list(output_model.parameters())
optimizer = torch.optim.SGD(params, lr=0.005)


print(
    df.shape,
    X_train.shape, 
    X_val.shape, 
    X_test.shape,
    "--------",
    y_train.shape, 
    y_val.shape,
    y_test.shape,
)

model_training(X_train, y_train)
model_training(X_val, y_val)

hidden_model.eval()
output_model.eval()
with torch.no_grad():

    outputs_tr = hidden_model(X_train) #hidden_model applied to input
    outputs_tr = nn.functional.sigmoid(outputs_tr)
    outputs_tr = output_model(outputs_tr)

    outputs_val = hidden_model(X_val) #hidden_model applied to input
    outputs_val = nn.functional.sigmoid(outputs_val)
    outputs_val = output_model(outputs_val)

    outputs_te = hidden_model(X_test) #hidden_model applied to input
    outputs_te = nn.functional.sigmoid(outputs_te)
    outputs_te = output_model(outputs_te)

    prediction_tr = nn.functional.sigmoid(outputs_tr) > 0.5  #transform tensors back into a range of 0 and 1
    prediction_v = nn.functional.sigmoid(outputs_val) > 0.5
    prediction_tes = nn.functional.sigmoid(outputs_te) > 0.5

    print(precision_score(y_train.cpu().numpy(), prediction_tr))
    print(precision_score(y_val.cpu().numpy(), prediction_v))
    print(precision_score(y_test.cpu().numpy(), prediction_tes))
   # print(precision_score(y_test, prediction_te)) #man I got better scores in multi class precision of an ankle heel with learned features from digits

