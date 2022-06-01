import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import  tqdm
import timm
from sklearn.metrics import classification_report
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 30
num_classes = 4
batch_size = 6
learning_rate = 0.001
img_size = {"s": [224, 224],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480],
                "_rw_m":[320,416]
            }
num_model = "_rw_m"
#data,data0-1
train_dataset = torchvision.datasets.ImageFolder(root='./data/train',
                                    transform=transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
                                            )

test_dataset = torchvision.datasets.ImageFolder(root='./data/val',
                                   transform=transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                   transforms.CenterCrop(img_size[num_model][1]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
for X, y in train_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
model = timm.create_model('efficientnetv2_rw_m', pretrained=True,num_classes=4)
#model.load_state_dict(torch.load('./weights/efficientnetv2_rw_m/distinguish1-2-model--9.pth'))
model.to(device)

#config = resolve_data_config({}, model=model)
#transform = create_transform(**config)
#model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    train_loader = tqdm(train_loader,file=sys.stdout)
    sample_num = 0
    for i, (images, labels) in enumerate(train_loader):
        sample_num += images.shape[0]
        images = images.to(device)
        labels = labels.to(device)
        print(labels)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)#
        print(predicted)
        accu_num += torch.eq(predicted, labels).sum()
        accu_loss += loss.detach()
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (i + 1),
                                                                               accu_num.item() / sample_num)
        torch.save(model.state_dict(), "./weights/efficientnetv2_rw_m/base-model--{}.pth".format(epoch))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
val_list = []
pred_list = []
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        for t in labels:
            val_list.append(t.data.item())
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
       # print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        for p in predicted:
            pred_list.append(p.data.item())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the  test images: {} %'.format(100 * correct / total))

print(classification_report(val_list, pred_list,target_names=train_dataset.class_to_idx))
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
