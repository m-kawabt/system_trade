import torch.nn as nn


class MyModel0(nn.Module):
    def __init__(self):
        self.layers = []
        self.layers += [nn.Linear(200*4, 1024, bias=True)]
        self.layers += [nn.ReLU(inplace=True)]
        self.layers += [nn.Linear(1024, 1024, bias=True)]
        self.layers += [nn.ReLU(inplace=True)]
        self.layers += [nn.Linear(1024, 1024, bias=True)]
        self.layers += [nn.ReLU(inplace=True)]
        self.layers += [nn.Linear(1024, 1024, bias=True)]
        self.layers += [nn.ReLU(inplace=True)]
        self.layers += [nn.Linear(1024, 3, bias=True)]
        self.softmax1 = [nn.Softmax()]
        self.softmax2 = [nn.Softmax()]

        self.modulelist = nn.ModuleList(self.layers)


    def forward(self, x):
        for l in self.modulelist:
            x = l(x)