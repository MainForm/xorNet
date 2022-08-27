import torch.nn as nn
import torch.nn.functional as F

class xorNet(nn.Module):
    def __init__(self):
        super(xorNet,self).__init__()

        self.layer1 = nn.Linear(2,4)
        self.layer2 = nn.Linear(4,1)
        

    def forward(self, x):
        x = self.layer1(x)
        x = F.sigmoid(x)
        x = self.layer2(x)
        
        return x