import torch
import torch.nn as nn

from custom_defined_op import AlphaNCReLUModuleFunction


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6 * 2, 16, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc3 = nn.Linear(16 * 6 * 6, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = AlphaNCReLUModuleFunction.apply(x)
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


if __name__ == "__main__":
    model = MyNet()  # 初始化实例
    ret = model(torch.randn(1, 1, 32, 32))  # 输入一张图片, 测试输出结果
    print(ret.shape)
