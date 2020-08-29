import numpy as np 

class Config:
    def __init__(self):
        
        self.trained_model = None
        self.is_training = True
        self.expert_data = 'expert_trajectories/hopper_er.bin'
        self.n_train_iters = 1000000
        self.n_episodes_test = 1
        self.test_interval = 1000
        self.n_steps_test = 1000
        self.vis_flag = True
        self.save_models = True
        self.config_dir = None
        self.continuous_actions = True

        # env spaces
        self.state_size = 64
        self.action_size = 64
        self.action_space = np.asarray([None]*self.action_size)

        # Main parameters to play with:al_loss
        self.er_agent_size = 50000
        self.prep_time = 1000
        self.collect_experience_interval = 15
        self.n_steps_train = 10
        self.discr_policy_itrvl = 100
        self.gamma = 0.99
        self.batch_size = 70
        self.weight_decay = 1e-7
        self.policy_al_w = 1e-2
        self.policy_tr_w = 1e-4
        self.policy_accum_steps = 10
        self.total_trans_err_allowed = 1000
        self.temp = 1.
        self.cost_sensitive_weight = 0.8
        self.noise_intensity = 6.
        self.drop = 0.25

        # Hidden layers sizes
        self.fm_size = 100
        self.d_size = [200, 100]
        self.p_size = [100, 50]

        # Learning rates
        self.fm_lr = 1e-4
        self.d_lr = 1e-3
        self.p_lr = 1e-4
            