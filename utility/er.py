import random
import torch
import numpy as np
from collections import namedtuple

class BufferClass:
    def __init__(self, buffer_size):
        self.buffer_sz = buffer_size
        self.buffer = []
        self.pos = 0
        self.experience = namedtuple('Transition', ['states', 'actions', 'rewards', 'next_states', 'dones'])

    def __len__(self):
        return len(self.buffer)
    
    def append(self, *args):
        e = args
        # if replay buffer not full, append new experience
        if len(self.buffer) < self.buffer_sz:
            self.buffer.append(e)
        else:
            self.buffer[self.pos] = e
        self.pos += 1
        self.pos = int(self.pos % self.buffer_sz)

    def sample(self, batch_size):
        raise NotImplementedError

    
class ReplayBuffer(BufferClass):
    def __init__(self, buffer_size):
        super().__init__(buffer_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.buffer), size=batch_size)
        
        states = {'pov':[], 'vector':[]}
        next_states = {'pov':[], 'vector':[]}
        actions, rewards, dones = [], [], []

        for i in idx:
            exp = self.buffer[i]
            for key, value in exp[0].items():
                states[key].append(value)
            for key, value in exp[3].items():
                next_states[key].append(value)
            actions.append(exp[1])
            rewards.append(exp[2])
            dones.append(int(exp[4]))

        for key, value in states.items():
            states[key] = np.array(value)
        for key, value in next_states.items():
            next_states[key] = np.array(value)
        actions = np.array(actions)
        rewards, dones = np.array(rewards).reshape(-1,1), np.array(dones).reshape(-1,1)

        return self.experience(states, actions, rewards, next_states, dones)


class DemonstrationBuffer(BufferClass):
    def __init__(self, envs: list, trim: bool, trim_reward: list, shuffle: bool=True):
        # create demonstration dataset
        n_samples = self.collate_data(envs, trim, trim_reward)
        # buffer size
        super().__init__(buffer_size=n_samples)
        delattr(self, 'buffer')

        if shuffle:
            idx = np.random.permutation(np.arange(n_samples))
            for key, value in self.stself.statesates.items():
                self.states[key] = value[idx]
            for key, value in self.next_states.items():
                self.next_states[key] = value[idx]
            self.actions = self.actions[idx]
            self.rewards = self.rewards[idx]
            self.dones = self.dones[idx]
        
    def sample(self, batch_size):
        idx = np.random.randint(0, self.buffer_sz, size=batch_size)

        states = {}
        for key, value in self.states.items():
            states[key] = value[idx]
        next_states = {}
        for key, value in self.next_states.items():
            next_states[key] = value[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]

        return self.experience(states, actions, rewards, next_states, dones)

    def append(self, *args):
        raise TypeError('cannot append samples to demonstration buffer.')

    def collate_data(self, envs, trim: bool, trim_reward: int):
        if not isinstance(envs, list):
            envs = [envs]
        # if an integer is provided as trim_reward, then use this for all envs
        if not isinstance(trim_reward, list):
            trim_reward = [trim_reward for i in range(len(envs))]
        
        self.states = {'pov': [], 'vector': []}
        self.actions = []
        self.rewards = []
        self.next_states = {'pov': [], 'vector': []}
        self.dones = []
        n_samples = 0

        import minerl
        for n_demo, env in enumerate(envs):
            env_data = minerl.data.make(environment=env, data_dir='/home/anshuman/MineRLTreechop/')
            trajectories = env_data.get_trajectory_names()
            for traj in trajectories:
                try:
                    traj_reward = 0
                    # sample trajectory only if its un-corrupted
                    for i, sample in enumerate(env_data.load_data(traj, include_metadata=True)):
                        if i == 0:
                            meta_data = sample[5]
                            # if trimming trajectories, skip those that dont meet required reward
                            if trim and meta_data['total_reward'] < trim_reward[n_demo]:
                                break

                        self.states['pov'].append(sample[0]['pov'])
                        self.states['vector'].append(sample[0]['vector'])
                        self.actions.append(sample[1]['vector'])
                        self.rewards.append(sample[2])
                        self.next_states['pov'].append(sample[3]['pov'])
                        self.next_states['vector'].append(sample[3]['vector'])
                        self.dones.append(sample[4])
                        n_samples += 1

                        traj_reward += sample[2]

                        # if trimming break when required reward is met
                        if trim and traj_reward >= trim_reward[n_demo]:
                            # makr the end of trimmed trajectory
                            self.dones[-1] = True
                            break

                except TypeError:
                    # sometimes trajectory file is corrupted, if so skip it
                    pass
        
        self.states['pov'], self.states['vector'] = np.array(self.states['pov']), np.array(self.states['vector'])
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards).reshape(-1,1)
        self.next_states['pov'], self.next_states['vector'] = np.array(self.next_states['pov']), np.array(self.next_states['vector'])
        self.dones = np.array(self.dones).reshape(-1,1)

        self.actions_std = np.std(self.actions)

        return n_samples