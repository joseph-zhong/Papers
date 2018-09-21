#!/usr/bin/env python

import torch
nn = torch.nn
optim = torch.optim
functional = torch.nn.functional
from environment import Environment
from replay_memory import ReplayMemory
from model import CrawlerDQN
import random
from torch.autograd import Variable

batch_size = 64
replay_capacity = 10000
num_episodes = 12000
episode_length = 80
width = 9
height = 9
gamma = 0.99
epsilon = 0.1

use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

memory = ReplayMemory(replay_capacity)
model = CrawlerDQN()
if use_cuda:
    model.cuda()

optimizer = optim.RMSprop(model.parameters())

def optimize():
    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    
    state_batch = Variable(
            torch.cat([FloatTensor(transition[0])
            for transition in transitions]))
    action_batch = Variable(
            torch.cat([LongTensor(transition[1])
            for transition in transitions]))
    next_state_batch = Variable(
            torch.cat([FloatTensor(transition[2])
            for transition in transitions]))
    reward_batch = Variable(
            torch.cat([FloatTensor([transition[3]])
            for transition in transitions]))
    
    predicted_q_values = model(state_batch).gather(1, action_batch)
    next_q_values = model(next_state_batch).gather(1, action_batch).max(1)[0]
    actual_q_values = (next_q_values * gamma) + reward_batch
    
    loss = functional.smooth_l1_loss(predicted_q_values, actual_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1,1)
    optimizer.step()
    return loss.data[0]

def train():
    
    for episode_id in xrange(num_episodes):
        print 'episode:', episode_id, '/', num_episodes,
        start = [random.randrange(9), random.randrange(9)]
        goal = [random.randrange(9), random.randrange(9)]
        env = Environment(
                width = width,
                height = height,
                start_position = start,
                goals = [goal],
                use_cuda = use_cuda)
        
        cumulative_reward = 0
        for step in xrange(episode_length):
            state = env.get_tensor()
            state.resize_(1,*state.size())
            action = model.epsilon_greedy_action(state, epsilon)
            env.take_action(action)
            next_state = env.get_tensor()
            next_state.resize_(1,*next_state.size())
            reward = env.smooth_reward()
            cumulative_reward += reward
            memory.push([state, action, next_state, reward])
        
        loss = optimize()
        print cumulative_reward, loss
    
    torch.save(model, 'checkpoint.model')

if __name__ == '__main__':
    train()
