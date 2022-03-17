# simple linear regression using torch

import torch
import numpy as np
import pickle

SIZE = 2000 # number of data points
POLY = 4 # cubic, can change

X = torch.unsqueeze(torch.linspace(-np.pi, np.pi, SIZE), axis=1)
Y = torch.sin(X)

model = torch.randn((1, POLY), requires_grad = True)

data = torch.cat([X.pow(i) for i in range(POLY)], axis=1)
print(f"Example data point: {data[5]}")

lr = 1e-6
for epoch in range(1, 2001):
    Y_pred = data@model.T
    loss = (Y_pred-Y).pow(2).sum()
    
    if ((epoch%100)==0):
        print(f"Epoch: {epoch}, loss: {loss.item()}")
    
    loss.backward()
    
    with torch.no_grad():
        model -= lr* model.grad
        model.grad = None
    
pickle.dump(model, open("model.pkl", "wb"))
    


