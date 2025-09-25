# -*- coding: utf-8 -*-
"""
基于 PyTorch 从零实现的多层感知机 (MLP) 
用于 MNIST 手写数字分类任务
"""

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
    """
    多层感知机模型
    """
    def __init__(self, input_size=784, hidden_size1=128, hidden_size2=64, output_size=10, dropout_rate=0.2):
        super(MLP, self).__init__()
        
        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
        # Dropout层用于防止过拟合
        self.dropout = nn.Dropout(dropout_rate)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        前向传播
        """
        # 将输入数据展平 (batch_size, 28, 28) -> (batch_size, 784)
        x = x.view(-1, 784)
        
        # 第一层：线性变换 + ReLU激活 + Dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 第二层：线性变换 + ReLU激活 + Dropout
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # 输出层：线性变换（不需要激活函数，交叉熵损失会处理）
        x = self.fc3(x)
        
        return x


def get_data_loaders(batch_size=64, test_batch_size=1000):
    """
    获取MNIST数据加载器
    """
    # 数据预处理：转换为Tensor并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 加载训练数据集
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # 加载测试数据集
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=test_batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader


def train_model(model, device, train_loader, optimizer, criterion, epoch):
    """
    训练模型一个epoch
    """
    model.train()  # 设置为训练模式
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据移动到设备上（CPU或GPU）
        data, target = data.to(device), target.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # 打印训练进度
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def test_model(model, device, test_loader, criterion):
    """
    测试模型
    """
    model.eval()  # 设置为评估模式
    test_loss = 0
    correct = 0
    
    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 累加损失
            test_loss += criterion(output, target).item()
            
            # 获取预测结果
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy


def visualize_predictions(model, device, test_loader, num_samples=8):
    """
    可视化模型预测结果
    """
    model.eval()
    
    # 获取一批测试数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # 选择前num_samples个样本
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # 进行预测
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
    
    # 绘制图像和预测结果
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
    """
    主函数
    """
    # 设置随机种子
    torch.manual_seed(1)
    
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 超参数设置
    batch_size = 64
    test_batch_size = 1000
    epochs = 10
    learning_rate = 0.01
    
    # 获取数据加载器
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(batch_size, test_batch_size)
    
    # 创建模型
    model = MLP().to(device)
    print(f"Model architecture:\n{model}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # 记录训练历史
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    print("\nStarting training...")
    start_time = time.time()
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        # 训练
        train_loss, train_acc = train_model(model, device, train_loader, optimizer, criterion, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 测试
        test_loss, test_acc = test_model(model, device, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time:.2f} seconds')
    
    # 绘制训练曲线
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
    
    # 可视化预测结果
    print("Generating prediction visualizations...")
    visualize_predictions(model, device, test_loader)
    
    # 保存模型
    torch.save(model.state_dict(), 'pytorch_mnist_model.pth')
    print("Model saved as 'pytorch_mnist_model.pth'")
    
    # 打印最终结果
    final_test_loss, final_test_acc = test_model(model, device, test_loader, criterion)
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")


if __name__ == '__main__':
    main()
