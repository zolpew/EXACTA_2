import torch
from torchvision import datasets, transforms
import re
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import EXACTALIB as exc
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
#preprocessing gambar
transform = transforms.Compose([transforms.Resize(48),                                
                                transforms.ToTensor()])


#load dataset
dataset = datasets.ImageFolder('C:/Users/numan/Work/Skin_Cancer detection/archive (3)', transform = transform)


#pisah menjadi train dan test
train_set, val_set = random_split(dataset, [9015, 1000])
train_loader = DataLoader(train_set, shuffle=True)
test_loader = DataLoader(val_set, shuffle=True)
train_idx = train_set.indices

net = exc.Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


show_img = next(iter(train_loader))[0][0]
plt.imshow( transforms.ToPILImage()(show_img))

obj = exc.get_target()


epochs = 2
for epoch in range(epochs):
    running_loss = 0.0
    for i,data in enumerate(train_loader):
        idex = obj.get(train_idx[i])
        target = torch.zeros([1,7])
        target[0][idex-1] = 1
        img = data[0]
        
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        #forward + backward + optimize
        outputs = net(img)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
        
 
        
PATH = './CNN_1.pth'
torch.save(net.state_dict(), PATH)

    

dataiter = iter(test_loader)
images, labels = next(dataiter)

net = exc.Net()
net.load_state_dict(torch.load(PATH))
outputs = net(images)
predicted = torch.max(outputs, 1)

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
       
