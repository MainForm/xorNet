import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import xorDataSet

from model import xorNet

# CUDA사용 가능 여부 확인
if torch.cuda.is_available() == True:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

BATCH_SIZE = 32
EPOCH_SIZE = 10

if __name__ == '__main__':
    dataset = xorDataSet()

    dataloader = DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)

    model = xorNet().to(DEVICE)
    #optimizer = torch.optim.SGD(model.parameters(),lr = 0.1,momentum=0.5)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.02)
    criterion = nn.MSELoss()


    for Epoch_idx in range(EPOCH_SIZE):
        print(f'{Epoch_idx} : ')

        model.train()
        for idx, (input,answer) in enumerate(dataloader):
            input = input.to(DEVICE)
            answer = answer.to(DEVICE)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output,answer)
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f'{idx * len(input)} / {len(dataloader.dataset)} losss : {loss.item()}')