import torch

from torch.utils.data import Dataset

from random import randint

class xorDataSet(Dataset):
    def __init__(self ,size=100000):

        answer = [
            #input      answer
            ([0.,0.],     0.),
            ([0.,1.],     1.),
            ([1.,0.],     1.),
            ([1.,1.],     0.)
        ]

        self.data = []

        for _ in range(size):
            self.data.append(answer[randint(0,len(answer) - 1)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor([self.data[idx][1]])