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
    model_path = "MyNet.pth"
    Net = MyNet()
    criterion = nn.CrossEntropyLoss().to(device)
    print(torch.load(model_path))
    Net.load_state_dict(torch.load(model_path))
    Net.to(device)
    print(Net)
    Net.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = Net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(data_test_loader), 'Loss: %.3f | Acc: %.3f%%(%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    inputs = inputs[0:1, :, :, :]
    torch.onnx.export(Net, inputs, "MyNet.onnx", opset_version=11, input_names=['input'], output_names=['output'], enable_onnx_checker=False)
