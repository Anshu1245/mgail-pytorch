from utility.brains import Discriminator, Policy, ForwardModel
# from utility.mgail import MGAIL 
from utility.obs_wrapper import ObsWrap
from utility.er import *
import torch
import torch.nn as nn
import sys

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
                               wdecay = self.config.weight_decay,
                               is_training = self.training)
        self.dloss = nn.CrossEntropyLoss()
        self.p = Policy(inp = self.config.state_size,
                        out = self.config.action_size,
                        size = self.config.p_size,
                        lr = self.config.p_lr,
                        drop = self.config.drop,
                        n_accum_steps = self.config.policy_accum_steps,
                        wdecay = self.config.weight_decay,
                        is_training = self.training)
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
        
        initial_gru_state = torch.ones((1, self.fm.encod_size))
        states_, actions, _, states = self.er_agent.sample(32)[:4]
        s = self.wrap(states)
        s_ = self.wrap(states_)
        preds, _ = self.fm([s, actions, initial_gru_state])
        loss = self.fmloss(preds, s_)
        self.fm.train(loss, self.fm.parameters())


    
    def train_d(self):
        
        state_a_, action_a = self.er_agent.sample(32)[:2]
        state_a_ = self.wrap(state_a_)
        state_e_, action_e = self.er_expert.sample(32)[:2]
        state_e_ = self.wrap(state_e_)
        s = torch.cat((state_a_, state_e_))
        a = torch.cat((action_a, action_e))
        labels_a = torch.zeroes(state_a_.shape[0], 1)
        labels_e = torch.ones(state_e_.shape[0], 1)
        labels = torch.cat((labels_a, labels_e))
        labels = torch.cat((labels, 1-labels), 1)
        preds = self.d(s, a)
        loss = self.dloss(preds, labels)
        self.d.train(loss, self.d.parameters())




    def train_p(self, obs, done):
        
        total_cost = 0
        t = 0
        initial_gru_state = torch.ones((1, self.fm.encod_size))


        # Adversarial Learning
        if done:
            state = self.env.reset()
        else:
            state = obs
        
        # Accumulate the (noisy) adversarial gradient
        for i in range(self.config.policy_accum_steps):
            # accumulate AL gradient
            state = self.wrap([state])
            mu = self.p(state)
            eta = self.sigma * torch.random_normal(shape=torch.shape(mu))
            action = mu + eta

            d = self.d(state, action)
            label = torch.tensor([[0, 1]]).float()
            cost = self.dloss(d, label)
            total_cost += torch.pow(self.config.gamma, t)*cost

            # get next state
            state_e, r, _, _ = self.env.step(action)
            state_a, _ = self.fm([state, action, initial_gru_state], reuse=True)

            nu = state_e - state_a
            state =  state_a + nu
            

            t += 1


        self.p.train(total_cost, self.p.parameters())
        return state

            
        
    
    
    def collect_experience(self, obs, record=1, vis=0, n_steps=None, noise_flag=True, start_at_zero=True):
        
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

        while not done:
            if not noise_flag:
                drop = 0.

            state = self.wrap([observation0])
            mu = self.p(state)
            eta = torch.random_normal(shape=torch.shape(a), stddev=self.sigma)
            a = torch.squeeze(mu + noise_flag * eta)

            observation, reward, done, _ = self.env.step(a)
            status = done
            done = done or t > n_steps
            t += 1
            R += reward
            print(reward)

            if record:
                action = a
                self.er_agent.append(observation0, action, reward, observation, done)
                observation0 = observation
        return observation, status

    def train_step(self):
        # phase_1 - Adversarial training
        # forward_model: learning from agent data
        # discriminator: learning in an interleaved mode with policy
        # policy: learning in adversarial mode

        # Fill Experience Buffer
        if self.itr == 0:
            
            done = True
            obs, done = self.collect_experience(None, start_at_zero=done)
            print('collecting experience')

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

        

    

    def save_model(self, dir_name=None):
        PATH = './model.pth'
        torch.save(self.p.state_dict(), PATH)
        







