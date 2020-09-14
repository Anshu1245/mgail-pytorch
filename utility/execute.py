from utility.brains import Discriminator, Policy, ForwardModel
# from utility.mgail import MGAIL 
from utility.obs_wrapper import ObsWrap
from utility.er import *
import torch
import torch.nn as nn
import sys
import numpy as np

class Execute:
    def __init__(self, env, config):
        
        self.env = env
        self.config = config
        self.training = config.is_training
        # self.alg = MGAIL(self.env, self.config)
        self.wrap = ObsWrap()
        self.d = Discriminator(inp = self.config.state_size + self.config.action_size,
                               out = 2,
                               size = self.config.d_size,
                               lr = self.config.d_lr,
                               drop = self.config.drop,
                               wdecay = self.config.weight_decay)
        self.dloss = nn.CrossEntropyLoss()
        self.p = Policy(inp = self.config.state_size,
                        out = self.config.action_size,
                        size = self.config.p_size,
                        lr = self.config.p_lr,
                        drop = self.config.drop,
                        n_accum_steps = self.config.policy_accum_steps,
                        wdecay = self.config.weight_decay)
        self.fm = ForwardModel(s_size=self.config.state_size,
                               a_size=self.config.action_size,
                               encod_size=self.config.fm_size,
                               lr=self.config.fm_lr)
        self.fmloss = nn.MSELoss()
        self.er_expert = DemonstrationBuffer(envs=['MineRLTreechopVectorObf-v0'], trim=False, trim_reward=None, shuffle=False)
        self.er_agent = ReplayBuffer(buffer_size = self.config.er_agent_size)
        self.itr = 0
        self.discriminator_policy_switch = 0
        self.mode = 'Prep'
        self.sigma = self.er_expert.actions_std / self.config.noise_intensity




    def train_fm(self):             
        count = 0
        print("training fm")
        initial_gru_state = torch.ones((1, self.fm.encod_size))
        exp_batch = self.er_agent.sample(32)
        states = self.convert_to_tensors(exp_batch.states)
        actions = self.convert_to_tensors(exp_batch.actions)
        states_ = self.convert_to_tensors(exp_batch.next_states)
        if not count:
            print(states['pov'].shape)
            count = 1
        s = self.wrap(states)
        s_ = self.wrap(states_)
        actions = self.normalize(actions, self.er_expert.actions_mean, self.er_expert.actions_std)
        preds, _ = self.fm([s, actions, initial_gru_state])
        loss = self.fmloss(preds, s_)
        self.fm.train_(loss, self.fm.parameters())


    
    def train_d(self):
        print("training d")
        self.d.train()
        exp_batch = self.er_agent.sample(32)
        state_a_ = self.convert_to_tensors(exp_batch.states)
        action_a = self.convert_to_tensors(exp_batch.actions)
        action_a = self.normalize(action_a, self.er_expert.actions_mean, self.er_expert.actions_std)
        state_a_ = self.wrap(state_a_)
        exp_batch = self.er_expert.sample(32)
        state_e_ = self.convert_to_tensors(exp_batch.states)
        action_e = self.convert_to_tensors(exp_batch.actions)
        action_e = self.normalize(action_e, self.er_expert.actions_mean, self.er_expert.actions_std)
        state_e_ = self.wrap(state_e_)
        s = torch.cat((state_a_, state_e_))
        a = torch.cat((action_a, action_e))
        labels_a = torch.zeroes(state_a_.shape[0], 1)
        labels_e = torch.ones(state_e_.shape[0], 1)
        labels = torch.cat((labels_a, labels_e))
        labels = torch.cat((labels, 1-labels), 1)
        preds = self.d(s, a)
        loss = self.dloss(preds, labels)
        self.d.train_(loss, self.d.parameters())




    def train_p(self, obs, done):
        self.p.train()
        total_cost = 0
        t = 0
        initial_gru_state = torch.ones((1, self.fm.encod_size))


        # Adversarial Learning
        if done:
            state = self.env.reset()
        else:
            state = obs
        
        pov = state['pov']
        state = self.convert_to_tensors({'pov':np.expand_dims(pov, 0), 'vector':np.expand_dims(state['vector'], 0)})
        state = self.wrap(state).squeeze(0)
        
        # Accumulate the (noisy) adversarial gradient
        for i in range(self.config.policy_accum_steps):
            print("accumulating gradients")
            # accumulate AL gradient
            
            mu = self.p(state)
            eta = self.sigma * torch.randn(size=mu.shape, dtype=torch.float)
            action = mu + eta

            d = self.d(state, action)
            label = torch.tensor([[0, 1]]).float()
            cost = self.dloss(d, label)
            total_cost += torch.pow(self.config.gamma, t)*cost

            # get next state
            action_detached = action
            action_detached = action_detached.detach().numpy()
            action_detached = {'vector':action_detached}
            action_detached = self.denormalize(action_detached, self.er_expert.actions_mean, self.er_expert.actions_std)
            state_e, _, _, _ = self.env.step(action_detached)
            pov = state_e['pov']
            state_e = self.convert_to_tensors({'pov':np.expand_dims(pov, 0), 'vector':np.expand_dims(state_e['vector'], 0)})
            state_e = self.wrap(state_e).squeeze(0)
            state_a, _ = self.fm([state, action, initial_gru_state])

            nu = state_e - state_a
            state =  state_a + nu
            

            t += 1


        self.p.train_(total_cost, self.p.parameters())
        print("training policy")
        return state

            
        
    
    
    def collect_experience(self, obs, record=1, vis=0, n_steps=None, noise_flag=True, start_at_zero=True):

        with torch.no_grad():
            self.p.eval()
            # environment initialization point
            if start_at_zero:
                observation0 = self.env.reset()
            else:
                observation0 = obs

            '''
            else:
                qposs, qvels = algo.er_expert.sample()[5:]
                observation = self.env.reset(qpos=qposs[0], qvel=qvels[0])
            '''
            drop = self.config.drop
            t = 0
            R = 0
            done = 0
            if n_steps is None:
                n_steps = self.config.n_steps_test

            print("collecting experience")
            while not done:
                if not noise_flag:
                    drop = 0.
                
                pov = observation0['pov']
                state = self.convert_to_tensors({'pov':np.expand_dims(pov, 0), 'vector':np.expand_dims(observation0['vector'], 0)})
                state = self.wrap(state).squeeze(0)
                mu = self.p(state)
                mu = self.denormalize(mu, self.er_expert.actions_mean, self.er_expert.actions_std)
                eta = torch.normal(mean=0.0, std=self.sigma, size=mu.shape)
                a = torch.squeeze(mu + noise_flag * eta)

                a = a.numpy()
                action_add = a
                a = {'vector':a}
                observation, reward, done, _ = self.env.step(a)
                status = done
                done = done or t > n_steps
                t += 1
                R += reward
                print(reward)

                if record:
                    action = action_add
                    self.er_agent.append(observation0, action, reward, observation, done)
                    observation0 = observation
            return observation, status

    def train_step(self):
        # phase_1 - Adversarial training
        # forward_model: learning from agent data
        # discriminator: learning in an interleaved mode with policy
        # policy: learning in adversarial mode

        # Fill Experience Buffer
        print("%dth iteration"%(self.itr+1))
        if self.itr == 0:
            
            done = True
            obs, done = self.collect_experience(None, start_at_zero=done)
            print('collecting initial experience')

        # Adversarial Learning
        else:
            self.train_fm()

            self.mode = 'Prep'
            if self.itr < self.config.prep_time:
                self.train_d()
            else:
                self.mode = 'AL'

                if self.discriminator_policy_switch:
                    self.train_d()
                else:
                    obs = self.train_p(obs, done)

                if self.itr % self.config.collect_experience_interval == 0:
                    obs, done = self.collect_experience(start_at_zero=done, n_steps=self.config.n_steps_train, obs=obs)

                # switch discriminator-policy
                if self.itr % self.config.discr_policy_itrvl == 0:
                    self.discriminator_policy_switch = not self.discriminator_policy_switch

        
    def normalize(self, x, mean, stddev):
        return (x-mean)/stddev

    def denormalize(self, x, mean, stddev):
        return x*stddev + mean    


    def convert_to_tensors(self, *args):
        if len(args) == 1:
            # dict was passed, return dict
            if isinstance(args[0], dict):
                tensors = {}
                for key, value in args[0].items():
                    tensors[key] = self.convert_to_tensors(value)

            # tuple or list was passed, return tuple
            elif isinstance(args[0], tuple) or isinstance(args[0], list):
                tensors = list([self.convert_to_tensors(arr) for arr in args[0]])

            # single array was passed
            elif isinstance(args[0], np.ndarray):
                tensors = torch.from_numpy(args[0])

            elif isinstance(args[0], torch.Tensor):
                tensors = args[0]

            else:
                raise TypeError('{} object cannot be converted to tensor'.format(type(args[0])))
            
        else:
            tensors = list([self.convert_to_tensors(arg) for arg in args])
        
        return tensors


    def save_model(self, dir_name=None):
        PATH = './model.pth'
        torch.save(self.p.state_dict(), PATH)
        







