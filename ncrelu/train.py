import torch
from torch import nn
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import MyNet


if __name__ == "__main__":
    data_train = MNIST("./data",
                       train=True,
                       download=True,
                       transform=transforms.Compose([transforms.Resize((32, 32)),
                                                     transforms.ToTensor()]))

    data_test = MNIST("./data",
                      train=False,
                      download=True,
                      transform=transforms.Compose([transforms.Resize((32, 32)),
                                                    transforms.ToTensor()]))

    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_train, batch_size=256, shuffle=False, num_workers=8)

    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    Net = MyNet().to(device)
    Net.train()
    lr = 0.01
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(Net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    train_loss = 0
    correct = 0
    total = 0

    epoches = 50

    loss_list = []

    #进行迭代训练
    for epoch in range(epoches):
        running_loss = 0.0
        for step, data in enumerate(data_train_loader, start = 0):
            inputs, labels = data                                   # 读取一个batch的数据
            inputs, labels = inputs.to(device), labels.to(device)   # 数据拷贝到GPU显存中
            optimizer.zero_grad()                                   # 梯度清零, 初始化
            outputs = Net(inputs)                                   # 前向传播
            loss = criterion(outputs, labels)                       # 计算误差
            loss.backward()                                         # 反向传播
            optimizer.step()                                        # 权重更新
            running_loss += loss.item()                             # 误差累计
            # print("loss is: {}".format(loss.item()))

            # 每50个batch打印一次损失值
            if step % 50 == 49:
                print("epoch:{} batch_idx:{} loss:{}".format(epoch+1, step+1, running_loss/50))
                loss_list.append(running_loss/50)
                running_loss = 0.0 # 误差清零

    print('Finished Training')

    save_path = 'MyNet.pth'                          
    torch.save(Net.state_dict(), save_path)          # 保存模型
    
    # 打印损失值变化曲线
    import matplotlib.pyplot as plt
    plt.plot(loss_list)
    plt.title('traning loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
