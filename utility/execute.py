from brains import Discriminator, Policy, ForwardModel
from mgail import MGAIL 

class Execute:
    def __init__(self, env, config):
        
        self.env = env
        self.config = config
        self.training = config.is_training
        self.alg = MGAIL(self.env, self.config)




    def train_fm(self):             # TODO
        algo = self.alg
        states_, actions, _, states = algo.er_agent.sample()[:4]


    def train_d(self):
        algo = self.alg
        state_a_, action_a = algo.er_agent.sample()[:2]
        state_e_, action_e = algo.er_expert.sample()[:2]
        s = torch.cat((state_a, state_e), 1)
        a = torch.cat((action_a, action_e), 1)




    def collect_exp(self, env, start_at_zero = True):


