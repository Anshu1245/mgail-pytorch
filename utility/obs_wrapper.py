import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class obs_wrap(nn.Module):
    
    def __init__(self):
        super(obs_wrap, self).__init__()

        #cnn pass, stack, fc    
        self.conv1 = nn.Conv2d(3, 10, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 15, 6)
        self.conv3 = nn.Conv2d(15, 20, 5)
        self.fc1 = nn.Linear(20*4*4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(128, 64)
        
    def forward(self, obs, from_env = 1):
        if from_env:
            pov = np.transpose(obs['pov']) #(TODO: transpose DOESNOT solve the 64x64x3 to 3x64x64 issue!)
            pov = torch.from_numpy(pov)
            pov = pov.unsqueeze(0)
        vec = torch.from_numpy(obs['vector']).float()
        pov = self.pool(F.relu(self.conv1(pov.float())))
        pov = self.pool(F.relu(self.conv2(pov)))
        pov = self.pool(F.relu(self.conv3(pov)))
        pov = pov.view(-1, 20*4*4)
        pov = F.relu(self.fc1(pov))
        pov = self.fc2(pov)
        pov = torch.squeeze(pov)
        s = torch.cat((pov, vec), 0)
        return self.fc3(s)

        
