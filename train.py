from utility.execute import Execute
from utility.config import Config
import gym 
import minerl

def main():

    env = gym.make('MineRLTreechopVectorObf-v0')
    config = Config()
    exe = Execute(env, config)
    for i in range(1000):
        exe.train_step()
    exe.save_model()

if __name__ == '__main__':
    
    main()
