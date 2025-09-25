import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size1=128, hidden_size2=64, output_size=10, dropout_rate=0.2):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        self.dropout = nn.Dropout(dropout_rate)

        self.relu = nn.ReLU()

    #定义前向传播
    def forward(self, x):
        x = x.view(-1, 784)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        return x

def get_data_loaders(batch_size=64, test_batch_size=1000):

    #先进行数据预处理：transform是数据预处理管道，用于将原始数据转换成神经网络可以处理的格式
    transform = transforms.Compose([ #Compose：将多个数据预处理操作组合在一起
        transforms.ToTensor(), #将PIL图像或numpy数组转换为PyTorch张量
        transforms.Normalize((0.1307,), (0.3081,)) #标准化：将图像的像素值缩放到0-1之间
    ])
    
    #加载训练数据集
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform #每次加载图片时自动执行 ToTensor() + Normalize()
    )

    #加载测试数据集
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform 
    )

    #创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

def train_model(model, device, train_loader, optimizer, criterion, epoch):
    model.train() #设置为训练模式：# 启用Dropout和BatchNorm的训练行为
                    # 进行前向传播、反向传播、参数更新
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        #将数据移动到设备上（CPU或GPU）
        data, target = data.to(device), target.to(device)

        #清零梯度
        optimizer.zero_grad()

        #前向传播
        output  = model(data)#在PyTorch中，nn.Module类定义了__call__()方法，当你写model(data)时，Python实际上调用的是model.__call__(data)。
        #直接调用forward()只执行前向传播，调用model()执行完整的处理流程（前向传播、反向传播、参数更新）

        #计算损失
        loss = criterion(output, target)

        #反向传播
        loss.backward()

        #更新参数:新权重 = 旧权重 - 学习率 × 梯度
        optimizer.step()

        #统计
        total_loss += loss.item() #loss.item()：将PyTorch张量转为Python数值
        pred = output.argmax(dim=1, keepdim=True) #argmax(dim=1)：找到每行最大值的索引，即预测的数字；keepdim=True：保持维度不变
        #eq(target.view_as(pred))：将预测的数字与真实数字进行比较，返回一个布尔值
        #target.view_as(pred) 是PyTorch中的张量形状变换方法，将target从[64]变为[64, 1]和pred一样的格式
        #sum().item()：将布尔值转换为数字，即预测正确的数量
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        #打印训练进度：Train Epoch: 1 [6400/60000 (11%)]    Loss: 0.456123
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

def test_model(model, device, test_loader, criterion):
    model.eval() # 关闭Dropout，固定BatchNorm统计量
                    # 只进行前向传播，获得稳定预测结果
    test_loss = 0
    correct = 0

    with torch.no_grad(): # 不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            #累加损失
            test_loss += criterion(output, target).item()

            #获取预测结果
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)

        print(f'\nTest set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

        return test_loss, accuracy

def visualize_predictions(model, device, test_loader, num_samples=8):
    model.eval()

    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    images = images[:num_samples]
    labels = labels[:num_samples]

    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(num_samples):
        img = images[i].cpu().squeeze()
        true_label = labels[i].item()
        pred_label = predictions[i].cpu().item()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.show()

def main():
    torch.manual_seed(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    batch_size = 64
    test_batch_size = 1000
    epochs = 10
    learning_rate = 0.01

    train_loader, test_loader = get_data_loaders(batch_size, test_batch_size)

    model = MLP().to(device)
    print(f"Model architecture:\n{model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []    

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_model(model, device, train_loader, optimizer, criterion, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        test_loss, test_acc = test_model(model, device, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time:.2f} seconds')

    plt.figure(figsize=(12, 4))
        
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Test Accuracy')
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    visualize_predictions(model, device, test_loader, num_samples=8)

if __name__ == '__main__':
    main()