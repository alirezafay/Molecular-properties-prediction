#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset,TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pandas as pd
from collections import Counter
import sklearn
from sklearn.model_selection import train_test_split
import torch.utils.data as data
from sklearn.model_selection import KFold


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[3]:


data = pd.read_csv('BBBP.csv')


# In[4]:


data


# In[5]:


print((data.iloc[:, 3][2049]))


# In[6]:


length_char = []
for i in range(2050):
    length_char.append(len(data.iloc[:,3][i]))


# In[7]:


unique_elements = len(set(length_char))
max_length = np.max(length_char)
min_length = np.min(length_char)
print(max_length)
print(min_length)
counter = Counter(length_char)
most_common_value = counter.most_common(1)[0][0]
print(most_common_value)


# In[8]:


plt.hist(length_char, unique_elements)
plt.xlabel('Element Values')
plt.ylabel('Number of Occurrences')
plt.title('Histogram of Element Values')
plt.show()


# In[9]:


smiles_combined = ''.join(data['smiles'])
token_counts = Counter(smiles_combined)
total_tokens = sum(token_counts.values())
token_diversity = len(token_counts)
print("Token Diversity:", token_diversity)
print("Token Occurrence Frequency:")
for token, count in token_counts.items():
    frequency = count / total_tokens
    print(f"{token}: {frequency}")


# In[10]:


tokens = []
unique_elements = set(smiles_combined)
for i in unique_elements:
    tokens.append(i)

def one_hot_encode(x):
    code = np.zeros((39,39))
    if x == 'v':
        out = code[0]
    else:
        for j in range(39):
            code[j][j] = 1
        index = tokens.index(x)
        out = code[index]
    return torch.from_numpy(out)


# In[10]:





# In[11]:


dataset = torch.zeros(2050,400,39)

data_list = []

for j in range(2050):
    data_list.append(data.iloc[:,3][j])

for i in range(2050):
    data_list[i] = data_list[i][:400].ljust(400, 'v')

for z in range(2050):
    for i in range(400):
        dataset[z][i] = one_hot_encode(data_list[z][i])


# In[12]:


labels = torch.zeros(2050)
for i in range(2050):
    labels[i] = data.iloc[:,2][i]


# 

# In[13]:


dataset[0][1]


# In[14]:


def training_data(model,num_epochs,train_loader,learn_rate):
    Loss_train=[]
    Loss_val=[]
    model.train()
    optimizer = optim.Adam(model.parameters(),lr=learn_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        correct_pred = 0
        total_samples = 0
        for input, labels in train_loader:
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            total_samples += labels.size(0)
            preds = torch.argmax (output, dim=1)
            correct_pred += (preds == labels).sum().item()
        epoch_train_accuracy =  100 * correct_pred / total_samples
        Loss_val.append(epoch_train_accuracy)
        Loss_train.append(epoch_train_loss)
        print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%")


# In[15]:


def testing_data(model,test_loader):
    total = 0
    correct_predict = 0
    model.eval()
    for inputt, labelz in test_loader:
        output = model(inputt)
        total += labelz.size(0)
        predict = torch.argmax (output, dim=1)
        correct_predict += (predict == labelz).sum().item()
    test_accuracy =  100 * correct_predict / total
    print(f"Test Accuracy: {test_accuracy}%")


# In[15]:





# In[16]:


class FC(nn.Module):
  def __init__(self,in_size,hidden_size,out_size):
    super(FC,self).__init__()
    self.fc1 = nn.Linear(in_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, out_size)
    self.dropout = nn.Dropout(p=0.4)
    self.sigmoid = nn.Sigmoid()

  def forward(self,x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.fc2(out)
    out = self.relu(out)
    out = self.dropout(out)
    out = self.fc3(out)
    out = self.sigmoid(out)
    return out


# In[17]:


x_train, x_test,y_train,y_test = train_test_split(dataset, labels, test_size=0.1, random_state=42)
x_train_flat = x_train.view(1845, -1)
x_test_flat = x_test.view(2050-1845, -1)
y_train = y_train.long()
y_test = y_test.long()
x_train_flat = x_train_flat.to(device)
y_train = y_train.to(device)
x_test_flat = x_test_flat.to(device)
y_test = y_test.to(device)
train_data = TensorDataset(x_train_flat, y_train)
test_data = TensorDataset(x_test_flat, y_test)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, shuffle=False)


# In[17]:





# In[17]:





# In[17]:





# In[18]:


model_FC = FC(400*39, 100, 2)
model_FC = model_FC.to(device)
training_data(model_FC,10,train_loader,0.001)


# In[19]:


testing_data(model_FC,test_loader)


# In[19]:





# In[20]:


torch.save(model_FC.state_dict(), "model_fc.pth")


# In[21]:


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(p=0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


# In[23]:


model_lstm = LSTM(400*39, 100, 2)
model_lstm = model_lstm.to(device)
training_data(model_lstm,20,train_loader,0.001)


# In[24]:


testing_data(model_lstm,test_loader)


# In[25]:


torch.save(model_lstm.state_dict(), "model_lstm.pth")


# In[26]:


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, batch_first=True,bidirectional=True)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(p=0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (h_n, c_n) = self.bilstm(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


# In[27]:


model_bilstm = BiLSTM(400*39, 100, 2)
model_bilstm = model_bilstm.to(device)
training_data(model_bilstm,20,train_loader,0.001)


# In[28]:


testing_data(model_bilstm,test_loader)


# In[29]:


torch.save(model_bilstm.state_dict(), "model_bilstm.pth")


# In[30]:


def train_cross(model,num_epochs,train_loader,learn_rate):
    Loss_train=[]
    Loss_val=[]
    model.train()
    optimizer = optim.Adam(model.parameters(),lr=learn_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        correct_pred = 0
        total_samples = 0
        for input, labels in train_loader:
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            total_samples += labels.size(0)
            preds = torch.argmax (output, dim=1)
            correct_pred += (preds == labels).sum().item()
        epoch_train_accuracy =  100 * correct_pred / total_samples
    return epoch_train_accuracy


# In[31]:


def testing_cross(model,test_loader):
    total = 0
    correct_predict = 0
    model.eval()
    for inputt, labelz in test_loader:
        output = model(inputt)
        total += labelz.size(0)
        predict = torch.argmax (output, dim=1)
        correct_predict += (predict == labelz).sum().item()
    test_accuracy =  100 * correct_predict / total
    return test_accuracy


# In[33]:


k = 10
kf = KFold(n_splits=k)
accuracy_scores_train_bilstm = []
accuracy_scores_valid_bilstm = []
accuracy_scores_train_lstm = []
accuracy_scores_valid_lstm = []
for train_index, val_index in kf.split(dataset):
    train_data, val_data = dataset[train_index], dataset[val_index]
    y_train, y_test = labels[train_index], labels[val_index]
    train_data_flat = train_data.view(train_data.size()[0], -1)
    train_data_flat = train_data_flat.to(device)
    val_data_flat = val_data.view(val_data.size()[0], -1)
    val_data_flat = val_data_flat.to(device)
    y_train = y_train.long()
    y_train = y_train.to(device)
    y_test = y_test.long()
    y_test = y_test.to(device)
    train_data = TensorDataset(train_data_flat, y_train)
    val_data = TensorDataset(val_data_flat, y_test)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(val_data, shuffle=False)
    model_bilstm_cross = BiLSTM(400*39, 100, 2)
    model_bilstm_cross = model_bilstm_cross.to(device)
    accuracy_train_bilstm = train_cross(model_bilstm_cross,25,train_loader,0.001)
    accuracy_valid_bilstm = testing_cross(model_bilstm_cross,test_loader)
    accuracy_scores_train_bilstm.append(accuracy_train_bilstm)
    accuracy_scores_valid_bilstm.append(accuracy_valid_bilstm)

    model_lstm_cross = LSTM(400*39, 100, 2)
    model_lstm_cross = model_lstm_cross.to(device)
    accuracy_train_lstm = train_cross(model_lstm_cross,25,train_loader,0.001)
    accuracy_valid_lstm = testing_cross(model_lstm_cross,test_loader)
    accuracy_scores_train_lstm.append(accuracy_train_lstm)
    accuracy_scores_valid_lstm.append(accuracy_valid_lstm)
avg_accuracy_train_bilstm = sum(accuracy_scores_train_bilstm) / k
avg_accuracy_valid_bilstm = sum(accuracy_scores_valid_bilstm) / k
avg_accuracy_train_lstm = sum(accuracy_scores_train_lstm) / k
avg_accuracy_valid_lstm = sum(accuracy_scores_valid_lstm) / k

print(f"Average Accuracy of BiLSTM Training: {avg_accuracy_train_bilstm}")
print(f"Average Accuracy of BiLSTM Validation: {avg_accuracy_valid_bilstm}")
print(f"Average Accuracy of LSTM Training: {avg_accuracy_train_lstm}")
print(f"Average Accuracy of LSTM Validation: {avg_accuracy_valid_lstm}")


