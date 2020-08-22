import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Discriminator(nn.Module):

    def __init__(self, inp, out, size, lr, drop, wdecay, is_training):
        super(Discriminator, self).__init__()
        self.inp = inp
        self.out = out
        self.hidden0 = size[0]
        self.hidden1 = size[1]
        self.drop = drop
        self.lr = lr
        self.wdecay = wdecay
        self.is_training = is_training

        # architecture
        self.fc1 = nn.Linear(self.inp, self.hidden0)
        self.fc2 = nn.Linear(self.hidden0, self.hidden1)
        self.fc3 = nn.Linear(self.hidden1, self.out)


    def forward(self, s, a):
        x = torch.cat((s, a), 0)
        x = F.relu(self.fc1(x))
        x = F.dropout(F.relu(self.fc2(x)), p=self.drop, training = self.is_training)
        x = self.fc3(x)  # softmax? 
        return x

    def train(self, loss, d_params):
        opt = optim.Adam(d_params, lr = self.lr, weight_decay = self.wdecay)
        opt.zero_grad()
        loss.backward()
        opt.step()



class Policy(nn.Module):
    
    def __init__(self, inp, out, size, lr, drop, n_accum_steps, wdecay, is_training):
        super(Policy, self).__init__()
        self.inp = inp
        self.out = out        
        self.hidden0 = size[0]
        self.hidden1 = size[1]
        self.drop = drop
        self.lr = lr
        self.wdecay = wdecay
        self.n_accum_steps =  n_accum_steps
        self.is_training = is_training

        # architecture
        self.fc1 = nn.Linear(self.inp, self.hidden0)
        self.fc2 = nn.Linear(self.hidden0, self.hidden1)
        self.fc3 = nn.Linear(self.hidden1, self.out)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = F.dropout(F.relu(self.fc2(x)), p=self.drop, training = self.is_training)
        x = self.fc3(x)   
        return x

    def train(self, loss, p_params):
        opt = optim.Adam(p_params, lr = self.lr, weight_decay = self.wdecay)
        opt.zero_grad()
        loss.backward()
        opt.step()


class ForwardModel(nn.Module):

    def __init__(self, s_size, a_size, encod_size, lr):
        super(ForwardModel, self).__init__()
        self.s_size = s_size
        self.a_size = a_size







        


