import torch
import polars as pl
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

def model_training(X, y):
    for _ in range (0, 200000):
        optimizer.zero_grad() #delete gradients to not accumulate so many values
        outputs = model(X)
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

model = nn.Linear(2, 1)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

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

model.eval()
with torch.no_grad():
    prediction_tr = model(X_train)
    prediction_v = model(X_val)
    prediction_tes = model(X_test)

    prediction_tr = nn.functional.sigmoid(prediction_tr) > 0.5  #transform tensors back into a range of 0 and 1
    prediction_v = nn.functional.sigmoid(prediction_v) > 0.5
    prediction_tes = nn.functional.sigmoid(prediction_tes) > 0.5

    print(precision_score(y_train.cpu().numpy(), prediction_tr))
    print(precision_score(y_val.cpu().numpy(), prediction_v))
    print(precision_score(y_test.cpu().numpy(), prediction_tes))
   # print(precision_score(y_test, prediction_te)) #man I got better scores in multi class precision of an ankle heel with learned features from digits

