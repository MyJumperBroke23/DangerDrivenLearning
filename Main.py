from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import time
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from CNN_Danger import Danger
from CNN_Policy import Policy

env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

DISCOUNT = 0.99
LEARNING_RATE = 1e-4
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.01
EPSILON_DECAY = 10000

danger_network = Danger(1, DISCOUNT, 0.3, LEARNING_RATE) #output_dim, discount, clip_factor, learning_rate)
policy_network = Policy(7, DISCOUNT, LEARNING_RATE, INITIAL_EPSILON, FINAL_EPSILON,
                        EPSILON_DECAY) #output_dim, discount, learning_rate, eps_i, eps_f, eps_d))

# Number of runs before danger network is updated
danger_learn_rate = 100;

# Let danger be defined as the possibility of dying within x frames, x being a hyperparameter
# Let the reward for the policy network be defined as alpha * danger - beta * done

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
        action = policy_network.select_action(state.copy())
        prev_state = state
        state, reward, done, info = env.step(action) #env.action_space.sample()
        policy_network.add_mem(prev_state, reward, action, state, done) # state, reward, action, next_state, done)
        policy_network.optimize_model()
        #print(env.action_space);
        #env.render()
        print(step)
    env.close()
