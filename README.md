# PyTorch 使用 vgg16 訓練 Stanford Dog Datasets

## 先前準備
* 確認有安裝相關套件，e.g. PyTorch、numpy、pandas、...
* Ubuntu 20.04 (Focal Fossa) 配 RTX 系列 GPU 安裝 PyTorch 可參考: [連結](https://hackmd.io/@TienYi/PyTorch-TensorFlow-install)

## 1. import 需要的 packages
```python=
import os, torch, torchvision, random
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from torch import optim
from torchsummary import summary
```
## 2. 確認一下使用 cuda 或是 cpu
```python=+
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
```

## 3. Dataloader 實作
```python=+
# 實作一個可以讀取 stanford dog (mini) 的 Pytorch dataset
class DogDataset(Dataset):
    
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames    # 資料集的所有檔名
        self.labels = labels          # 影像的標籤
        self.transform = transform    # 影像的轉換方式
 
    def __len__(self):
        return len(self.filenames)    # return DataSet 長度
 
    def __getitem__(self, idx):       # idx: Inedx of filenames
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image) # Transform image
        label = np.array(self.labels[idx])
        return image, label           # return 模型訓練所需的資訊
```

## 4. 定義 Normalize 以及 Transform 的參數
```python=+
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

# Transformer
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
 
test_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
```

## 5. 將每一個類別以 8 : 2 的比例分割成 Training data 和 Testing data 傳至 dataloader
```python=+
def split_Train_Val_Data(data_dir):
    dataset = ImageFolder(data_dir) 
    # 建立 20 類的 list
    character = [[] for i in range(len(dataset.classes))]
    # print(character)
    
    # 將每一類的檔名依序存入相對應的 list
    for x, y in dataset.samples:
        character[y].append(x)
      
    train_inputs, test_inputs = [], []
    train_labels, test_labels = [], []
    
    for i, data in enumerate(character): # 讀取每個類別中所有的檔名 (i: label, data: filename)
        np.random.seed(42)
        np.random.shuffle(data)
            
        # -------------------------------------------
        # 將每一類都以 8:2 的比例分成訓練資料和測試資料
        # -------------------------------------------
        num_sample_train = int(len(data) * 0.8)
        num_sample_test = len(data) - num_sample_train
        # print(str(i) + ': ' + str(len(data)) + ' | ' + str(num_sample_train) + ' | ' + str(num_sample_test))
        
        for x in data[:num_sample_train] : # 前 80% 資料存進 training list
            train_inputs.append(x)
            train_labels.append(i)
            
        for x in data[num_sample_train:] : # 後 20% 資料存進 testing list
            test_inputs.append(x)
            test_labels.append(i)

    train_dataloader = DataLoader(DogDataset(train_inputs, train_labels, train_transformer),
                                  batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(DogDataset(test_inputs, test_labels, test_transformer),
                                  batch_size = batch_size, shuffle = False)
    return train_dataloader, test_dataloader
```

## 6. 建立 CNN Model
```python=+
# 參數設定
batch_size = 32                                  # Batch Size
lr = 1e-3                                        # Learning Rate
epochs = 50                                      # epoch 次數

data_dir = 'stanford_dog'                        # 資料夾名稱
```

## 7. 利用 Pytorch 內建的 CNN model 來進行訓練且 torchsummary 來印出模型的架構資訊
```python=+
train_dataloader, test_dataloader = split_Train_Val_Data(data_dir)
C = models.vgg16(pretrained=True).to(device)     # 使用內建的 model 
optimizer_C = optim.SGD(C.parameters(), lr = lr) # 選擇你想用的 optimizer
summary(C, (3, 244, 244))                        # 利用 torchsummary 的 summary package 印出模型資訊，input size: (3 * 224 * 224)
# Loss function
criterion = nn.CrossEntropyLoss()                # 選擇想用的 loss function
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 244, 244]           1,792
              ReLU-2         [-1, 64, 244, 244]               0
            Conv2d-3         [-1, 64, 244, 244]          36,928
              ReLU-4         [-1, 64, 244, 244]               0
         MaxPool2d-5         [-1, 64, 122, 122]               0
            Conv2d-6        [-1, 128, 122, 122]          73,856
              ReLU-7        [-1, 128, 122, 122]               0
            Conv2d-8        [-1, 128, 122, 122]         147,584
              ReLU-9        [-1, 128, 122, 122]               0
        MaxPool2d-10          [-1, 128, 61, 61]               0
           Conv2d-11          [-1, 256, 61, 61]         295,168
             ReLU-12          [-1, 256, 61, 61]               0
           Conv2d-13          [-1, 256, 61, 61]         590,080
             ReLU-14          [-1, 256, 61, 61]               0
           Conv2d-15          [-1, 256, 61, 61]         590,080
             ReLU-16          [-1, 256, 61, 61]               0
        MaxPool2d-17          [-1, 256, 30, 30]               0
           Conv2d-18          [-1, 512, 30, 30]       1,180,160
             ReLU-19          [-1, 512, 30, 30]               0
           Conv2d-20          [-1, 512, 30, 30]       2,359,808
             ReLU-21          [-1, 512, 30, 30]               0
           Conv2d-22          [-1, 512, 30, 30]       2,359,808
             ReLU-23          [-1, 512, 30, 30]               0
        MaxPool2d-24          [-1, 512, 15, 15]               0
           Conv2d-25          [-1, 512, 15, 15]       2,359,808
             ReLU-26          [-1, 512, 15, 15]               0
           Conv2d-27          [-1, 512, 15, 15]       2,359,808
             ReLU-28          [-1, 512, 15, 15]               0
           Conv2d-29          [-1, 512, 15, 15]       2,359,808
             ReLU-30          [-1, 512, 15, 15]               0
        MaxPool2d-31            [-1, 512, 7, 7]               0
AdaptiveAvgPool2d-32            [-1, 512, 7, 7]               0
           Linear-33                 [-1, 4096]     102,764,544
             ReLU-34                 [-1, 4096]               0
          Dropout-35                 [-1, 4096]               0
           Linear-36                 [-1, 4096]      16,781,312
             ReLU-37                 [-1, 4096]               0
          Dropout-38                 [-1, 4096]               0
           Linear-39                 [-1, 1000]       4,097,000
================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.68
Forward/backward pass size (MB): 258.51
Params size (MB): 527.79
Estimated Total Size (MB): 786.98
----------------------------------------------------------------
```

## 8. 儲存訓練資訊的 List
```python=+
loss_epoch_C = []
train_acc, test_acc = [], []
best_acc, best_auc = 0.0, 0.0
```

## 9. 實作模型訓練和測試模型效能
```python=+
if __name__ == '__main__':    
    for epoch in range(epochs):
        start_time = time.time()
        iter = 0
        correct_train, total_train = 0, 0
        correct_test, total_test = 0, 0
        train_loss_C = 0.0

        C.train() # 設定 train 或 eval
        print('epoch: ' + str(epoch + 1) + ' / ' + str(epochs))  
        
        # ---------------------------
        # Training Stage
        # ---------------------------
        for i, (x, label) in enumerate(train_dataloader) :
            x, label = x.to(device), label.to(device)
            optimizer_C.zero_grad()                         # 清空梯度
            train_output = C(x)                             # 將訓練資料輸入至模型進行訓練 (Forward propagation)
            train_loss = criterion(train_output, label)     # 計算 loss
            train_loss.backward()                           # 將 loss 反向傳播
            optimizer_C.step()                              # 更新權重
            
            # 計算訓練資料的準確度 (correct_train / total_train)
            _, predicted = torch.max(train_output.data, 1)  # 取出預測的 maximum
            total_train += label.size(0)
            correct_train += (predicted == label).sum()
            train_loss_C += train_loss.item()
            iter += 1
                    
        print('Training epoch: %d / loss_C: %.3f | acc: %.3f' % \
              (epoch + 1, train_loss_C / iter, correct_train / total_train))
        
        # --------------------------
        # Testing Stage
        # --------------------------
        C.eval() # 設定 train 或 eval
        for i, (x, label) in enumerate(test_dataloader) :
            with torch.no_grad():                           # 測試階段不需要求梯度
                x, label = x.to(device), label.to(device)
                test_output = C(x)                          # 將測試資料輸入至模型進行測試
                test_loss = criterion(test_output, label)   # 計算 loss
                
                # 計算測試資料的準確度 (correct_test / total_test)
                _, predicted = torch.max(test_output.data, 1)
                total_test += label.size(0)
                correct_test += (predicted == label).sum()
        
        print('Testing acc: %.3f' % (correct_test / total_test))
                                     
        train_acc.append(100 * (correct_train / total_train).cpu()) # training accuracy
        test_acc.append(100 * (correct_test / total_test).cpu())    # testing accuracy
        loss_epoch_C.append((train_loss_C / iter))            # loss 

        end_time = time.time()
        print('Cost %.3f(secs)' % (end_time - start_time))
```

```
epoch: 1 / 50
Training epoch: 1 / loss_C: 5.587 | acc: 0.014
Testing acc: 0.020
Cost 122.854(secs)
epoch: 2 / 50
Training epoch: 2 / loss_C: 4.714 | acc: 0.040
Testing acc: 0.041
Cost 122.754(secs)

show more (open the raw output data in a text editor) ...

epoch: 50 / 50
Training epoch: 50 / loss_C: 1.147 | acc: 0.678
Testing acc: 0.691
Cost 121.660(secs)
```

## 10. 將每一個 epoch 的 Loss 以及 Training / Testing accuracy 紀錄下來並繪製成圖
```python=+
plt.figure()
plt.plot(list(range(epochs)), loss_epoch_C) # plot your loss
plt.title('Training Loss')
plt.ylabel('loss'), plt.xlabel('epoch')
plt.legend(['loss_C'], loc = 'upper left')
plt.savefig('loss.png')
plt.show()

plt.figure()
plt.plot(list(range(epochs)), train_acc)    # plot your training accuracy
plt.plot(list(range(epochs)), test_acc)     # plot your testing accuracy
plt.title('Training acc')
plt.ylabel('acc (%)'), plt.xlabel('epoch')
plt.legend(['training acc', 'testing acc'], loc = 'upper left')
plt.savefig('acc.png')
plt.show()
```
![](https://imgur.com/lkI1jgS.png)
![](https://i.imgur.com/TLTWy5R.png)
