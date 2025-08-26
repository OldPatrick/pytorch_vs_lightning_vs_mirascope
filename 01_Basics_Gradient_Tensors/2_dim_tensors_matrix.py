import torch

b = torch.tensor(32)
w1 = torch.tensor(1.8)

# a column reflects a feature here
X1 = torch.tensor(
    [
        [10], 
        [38], 
        [100], 
        [150]
    ]
)

y_pred = 1 * b + X1 * w1 #cpu advantage not from gpu, vectorised extension
print("Matrix with 4 rows and 1 column:", X1)
print("Matrix with 4 rows and 1 column:", X1.size())


print("slicing X1[2, 0]:", X1[2, 0])
print("the first sliced element gives me the row, the second the column")

print("slicing X1[:, 0]:", X1[:, 0])
print("Gives me the vector again")