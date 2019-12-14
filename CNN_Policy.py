import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import random
import math


class CNN_Policy(nn.Module):
    def __init__(self, output_dim):
        super(CNN_Policy, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.linear = nn.Linear(25056, output_dim)

    def forward(self, x):
        if (len(x.shape) == 3):
            x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
        x = x.view(x.shape[0], x.shape[3], x.shape[2], x.shape[1])
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        #print(F.relu(self.linear(x)))
        return F.relu(self.linear(x))


def select_action(state, initial_epsilon, final_epsilon, steps_done, epsilon_decay, model):
    sample = random.random()
    eps_threshold = final_epsilon + (initial_epsilon - final_epsilon) * \
                    math.exp(-1. * steps_done / epsilon_decay)
    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.Tensor(state)
            steps_done += 1
            q_calc = model(state)
            node_activated = int(torch.argmax(q_calc))
            return node_activated
    else:
        node_activated = random.randint(0,10)
        steps_done += 1
        return node_activated


class ReplayMemory(object): # Stores [state, reward, action, next_state, done]

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [[],[],[],[],[]]

    def push(self, data):
        """Saves a transition."""
        for idx, point in enumerate(data):
            #print("Col {} appended {}".format(idx, point))
            self.memory[idx].append(point)

    def sample(self, batch_size):
        rows = random.sample(range(0, len(self.memory[0])), batch_size)
        experiences = [[],[],[],[],[]]
        for row in rows:
            for col in range(5):
                experiences[col].append(self.memory[col][row])
        return experiences

    def __len__(self):
        return len(self.memory[0])


class Policy:
    def __init__(self, output_dim, discount, learning_rate, eps_i, eps_f, eps_d):
        self.memory = ReplayMemory(1024)
        self.model = CNN_Policy(output_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.discount = discount
        self.learning_rate = learning_rate
        self.BATCH_SIZE = 8
        self.steps_done = 0
        self.initial_epsilon = eps_i
        self.final_epsilon = eps_f
        self.epsilon_decay = eps_d

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * \
                        math.exp(-1. * self.steps_done / self.epsilon_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                state = torch.Tensor(state)
                self.steps_done += 1
                q_calc = self.model(state)
                node_activated = int(torch.argmax(q_calc))
                return node_activated
        else:
            node_activated = random.randint(0, 6)
            self.steps_done += 1
            return node_activated

    def add_mem(self, state, reward, action, next_state, done):
        self.memory.push((state, reward, action, next_state, done))

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return 0
        experiences = self.memory.sample(self.BATCH_SIZE)
        state_batch = torch.Tensor(experiences[0])
        reward_batch = torch.Tensor(experiences[1])
        action_batch = torch.LongTensor(experiences[2]).unsqueeze(1)
        next_state_batch = torch.Tensor(experiences[3])
        done_batch = experiences[4]

        #print(self.model(state_batch))

        pred_q = self.model(state_batch).gather(1, action_batch)

        next_state_q_vals = torch.zeros(self.BATCH_SIZE)

        for idx, next_state in enumerate(next_state_batch):
            if done_batch[idx] == True:
                next_state_q_vals[idx] = -1
            else:
                # .max in pytorch returns (values, idx), we only want vals
                next_state_q_vals[idx] = self.model(next_state_batch[idx])[0].max(0)[0]
                #print(next_state_q_vals[idx])

        better_pred = (reward_batch + next_state_q_vals).unsqueeze(1)

        loss = F.smooth_l1_loss(pred_q, better_pred)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss



