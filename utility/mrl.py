import gym 
import minerl
from obs_wrapper import ObsWrap
import torch 
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np

wrap = ObsWrap()
env = gym.make("MineRLTreechopVectorObf-v0")
obs = [env.reset()]
tot = 0
done  = False
target = torch.randn(2, 64)
i = 2
    
while not done:
    a = env.action_space.sample()
    s, reward, done, info = env.step(a)
    
    
    if i == 30:
        done = True
    i+=1

env1 = deepcopy(env)

done = 0
while not done:
    print('resumed')
    a = env1.action_space.sample()
    s, reward, done, info = env1.step(a)

'''
opt = optim.SGD(wrap.parameters(), lr = 0.001)
opt.zero_grad()
lossfn = nn.MSELoss()
obs.append(s)
if i%1 == 0:
    obs = wrap(obs)
    loss = lossfn(obs, target)
    loss.backward()
    opt.step()
    print(loss)
    obs = []
i += 1
'''