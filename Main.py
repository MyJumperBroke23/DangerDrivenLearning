from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import time
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Number of runs before danger network is updated
danger_learn_rate = 100;

# Do many runs
for run in range(5000):
    if run % danger_learn_rate == 0:
        print("Update here")
    # Step through run
    done = True
    for step in range(5000):
        if done:
            state = env.reset()
        #time.sleep(0.1);
        state, reward, done, info = env.step(env.action_space.sample())
        #print(env.action_space);
        env.render()
    env.close()
