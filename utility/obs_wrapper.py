import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ObsWrap(nn.Module):
    
    def __init__(self):
        super(ObsWrap, self).__init__()

        #cnn pass, stack, fc    
        self.conv1 = nn.Conv2d(3, 10, 7)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 15, 6)
        self.conv3 = nn.Conv2d(15, 20, 5)
        self.fc1 = nn.Linear(20*4*4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(128, 64)
        
    def forward(self, obs):
        pov = torch.from_numpy(np.array([])).float()
        vec = torch.from_numpy(np.array([])).float()
        for ob in range(len(obs)):
            pov_ob = np.transpose(obs[ob]['pov']) #(TODO: transpose DOESNOT solve the 64x64x3 to 3x64x64 issue!)
            pov_ob = torch.from_numpy(pov_ob).float()
            pov_ob = pov_ob.unsqueeze(0)
            vec_ob = torch.from_numpy(obs[ob]['vector']).float()
            vec_ob = vec_ob.unsqueeze(0)
            pov = torch.cat((pov, pov_ob))
            vec = torch.cat((vec, vec_ob))
        pov = self.pool(F.relu(self.conv1(pov)))
        pov = self.pool(F.relu(self.conv2(pov)))
        pov = self.pool(F.relu(self.conv3(pov)))
        pov = pov.view(-1, 20*4*4)
        pov = F.relu(self.fc1(pov))
        pov = self.fc2(pov)
        # pov = torch.squeeze(pov)
        s = torch.cat((pov, vec), 1)
        return self.fc3(s)
    
    def whatever(self, a):
        print("you're screwed")
        
