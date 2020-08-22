from brains import Discriminator, Policy, ForwardModel


class MGAIL:
    def __init__(self, env, config):
        self.env = env
        self.training = config.is_training
        self.config = config

        # create networks and memory
        self.d = Discriminator(inp = self.config.state_size + self.config.action_size,
                               out = 2,
                               size = self.config.d_size,
                               lr = self.config.d_lr,
                               drop = self.config.drop,
                               wdecay = self.config.weight_decay,
                               is_training = self.config.training)
        
        self.p = Policy(inp = self.config.state_size,
                        out = self.config.action_size,
                        size = self.config.p_size,
                        lr = self.config.p_lr,
                        drop = self.config.drop,
                        n_accum_steps = self.config.policy_accum_steps,
                        wdecay = self.config.weight_decay,
                        is_training = self.config.training)

        self.fm = ForwardModel(s_size=self.config.state_size,
                               a_size=self.config.action_size,
                               encod_size=self.config.fm_size,
                               lr=self.config.fm_lr)
