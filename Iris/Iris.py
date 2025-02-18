from ucimlrepo import fetch_ucirepo 
import torch
import numpy as np
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 

# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 

X = np.array(X)
y = np.array(y)
y = y.reshape(-1)


X = torch.tensor(X,dtype=torch.float32)
for i in range(y.shape[0]):
    if y[i] == 'Iris-setosa':
        y[i] = 0
    elif y[i] == 'Iris-versicolor':
        y[i] = 1
    else:
        y[i] = 2

# 将 y 转换为整数类型
y = y.astype(np.int64)

# 将 y 转换为 PyTorch 张量
y = torch.tensor(y, dtype=torch.long)  # 使用 long 类型表示整数标签

device = 'mps'
class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.fc2 = torch.nn.Linear(8, 3)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = Classifier().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.1)
epochs = 1000

for epoch in range(epochs):
    X = X.to(device)
    y = y.to(device)
    model.train()
    y_pred_logit = model(X)
    loss = loss_fn(y_pred_logit, y)
    y_pred_prob = torch.softmax(y_pred_logit,dim=1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    accuracy = (y == torch.argmax(y_pred_prob,dim=1)).float().mean()
    if epoch%100==0:
        print(f'Epoch {epoch} Loss {loss.item()} Acc {accuracy}')