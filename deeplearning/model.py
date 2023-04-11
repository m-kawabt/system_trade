import torch.nn as nn
import torch


class MyModel0(nn.Module):
    def __init__(self):
        super(MyModel0, self).__init__()
        layers = []
        layers += [nn.Linear(180*4, 1024, bias=True)]
        layers += [nn.ReLU(inplace=True)]
        # layers += [nn.Linear(1024, 1024, bias=True)]
        # layers += [nn.ReLU(inplace=True)]
        # layers += [nn.Linear(1024, 1024, bias=True)]
        # layers += [nn.ReLU(inplace=True)]
        # layers += [nn.Linear(1024, 1024, bias=True)]
        # layers += [nn.ReLU(inplace=True)]
        layers += [nn.Linear(1024, 4, bias=True)]
        self.softmax = nn.Softmax(dim=-1)

        self.modulelist = nn.ModuleList(layers)


    def forward(self, x):
        for l in self.modulelist:
            x = l(x)
        batch_size = x.size(0)
        x = torch.reshape(x, [batch_size, 2, 2])
        x = self.softmax(x)

        return x

if __name__ == '__main__':
    model = MyModel0()
    print(model.modules)