import torch

b = torch.tensor(32)
# also works if the skalar is not explicitly defined as a tensor, even b and w1 so parameters can be optionally a tensor, there is no must for them to be tensors, but input must be
w1 = torch.tensor(1.8)

X1 = torch.tensor(100)


#Ein Tensor ist in der Tat ein mathematisches Objekt, das als "Box" fungiert und Werte speichert. Er verallgemeinert Skalare, Vektoren und Matrizen auf höhere Dimensionen, was entscheidend für Berechnungen in Bereichen wie maschinellem Lernen und Deep Learning ist.

y_pred = 1 * b + X1 * w1
print("0-dimensional tensor:", y_pred)