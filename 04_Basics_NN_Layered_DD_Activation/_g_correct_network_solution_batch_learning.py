# why do we use acivation functions? To break linearity
# Rectified Linear Unit

# Sigmoid kann have very small gradients playing back
# we are preserving large gradients, outputting 0 or value itself

# Sparsity: 0 neurons, by chance 5 could output 0 by chance
 #meaning 50 % outputs zero and only half of the network is activated
# Efficiency NN should look at the bigger picture and gets not lost in details
# only a max function and is easier to compute than the exp of a sigmoid

# Adam is like aball running rolling down the hill while SGD is making smaller steps 
# toward the minium and in general steps, when walking down he hill, Adam is rolling, but it may overshoot, so it needs to roll back a little big
# but we can take smaller steps, depending on the parameter how fast its going
# ADAM: Adaptive Moment Estimation
import torch
import polars as pl
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

#print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.device(0))
#print(torch.cuda.get_device_name(0))


def model_training(X_train, y_train, X_val, y_val):
    for _ in range (0, 50000):
        optimizer.zero_grad() #delete gradients to not accumulate so many values
        outputs = model(X_train)
        loss = loss_fn(outputs, y_train) #between prediction and real one
        loss.backward() #backpropagation
        optimizer.step() #(into the right/next direction)

        if _ % 1000 == 0:
            print(loss)

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = loss_fn(val_logits, y_val)
            print(f"Epoch {_:5d} | train_loss={loss.item():.4f} | val_loss={val_loss.item():.4f}")
        
    
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

model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(), #only placeholder
    nn.Linear(10, 1)
)   
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model_training(X_train, y_train, X_val, y_val)

model.eval()
# Evaluierungsmodus Funktionen wie Dropout und Batch-Normalisierung deaktiviert, 
# die während des Trainings variabel sind. Dies führt zu stabilen und vorhersehbaren Ergebnissen bei der Anwendung des Modells

with torch.no_grad():

    outputs_tr = model(X_train)
    outputs_va = model(X_val)
    outputs_te = model(X_test)
   
   #need it still at the end to push it into classification
    prediction_tr = nn.functional.sigmoid(outputs_tr) > 0.5  
    prediction_va = nn.functional.sigmoid(outputs_va) > 0.5
    prediction_te = nn.functional.sigmoid(outputs_te) > 0.5

    print(precision_score(y_train.cpu().numpy(), prediction_tr))
    print(precision_score(y_val.cpu().numpy(), prediction_va))
    print(precision_score(y_test.cpu().numpy(), prediction_te))

    # we are now missing the opportunity to save the model that created for example the lowest val_loss, to use for prediction of test
