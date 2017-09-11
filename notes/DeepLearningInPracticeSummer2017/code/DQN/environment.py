'''
XXXXXXXXXX
X        X
X  A     X
X        X
X     B  X
X    O   X
X        X
X        X
X  C     X
X        X
XXXXXXXXXX
'''

import torch
import string
import numpy

class Environment(object):
    def __init__(
            self,
            width=9,
            height=9,
            start_position = [0,0],
            goals=[[0,0]],
            start_goal_id=0,
            use_cuda=False):
        
        self.use_cuda = use_cuda
        Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        
        num_channels = 1 + len(goals)
        self.state = Tensor(
            [[[0. for _ in xrange(width)] for _ in xrange(height)]
                for _ in xrange(num_channels)])
        
        self.width = width
        self.height = height
        
        self.player_position = start_position
        self.set_player(start_position)
        
        self.goals = goals
        for goal_id, goal in enumerate(self.goals):
            self.set_goal(goal, goal_id)
        self.set_goal_index(start_goal_id)
    
    def clamp_position(self, position):
        position = list(position[:])
        position[0] = min(self.width-1, max(0, position[0]))
        position[1] = min(self.height-1, max(0, position[1]))
        return position
    
    def set_goal(self, position, goal_id):
        position = self.clamp_position(position)
        prev_position = self.goals[goal_id]
        self.goals[goal_id] = position
        channel = goal_id + 1
        self.move_item(channel, prev_position, position)
    
    def set_goal_index(self, goal_id):
        self.goal_index = goal_id
    
    def set_player(self, position):
        position = self.clamp_position(position)
        prev_position = self.player_position
        self.player_position = position
        channel = 0
        self.move_item(channel, prev_position, position)
    
    def move_item(self, channel, prev_position, position):
        self.state[channel, prev_position[0], prev_position[1]] = 0.
        self.state[channel, position[0],position[1]] = 1.
    
    def take_action(self, action):
        
        if action[0,0] == 0:
            self.set_player(numpy.array([-1, 0]) + self.player_position)
        elif action[0,0] == 1:
            self.set_player(numpy.array([ 0,-1]) + self.player_position)
        elif action[0,0] == 2:
            self.set_player(numpy.array([ 1, 0]) + self.player_position)
        elif action[0,0] == 3:
            self.set_player(numpy.array([ 0, 1]) + self.player_position)
        elif action[0,0] == 4:
            pass
    
    def binary_reward(self):
        if (self.player_position[0] == self.goals[self.goal_index][0] and
            self.player_position[1] == self.goals[self.goal_index][1]):
            return 1.0
        else:
            return 0.0
    
    def smooth_reward(self):
        dx = self.goals[self.goal_index][0] - self.player_position[0]
        dy = self.goals[self.goal_index][1] - self.player_position[1]
        d = (dx*dx + dy*dy)**0.5
        max_d = (self.width*self.width + self.height*self.height)**0.5
        return 1. - (float(d)/max_d)
    
    def draw(self):
        block = [[' ' for _ in xrange(self.width)] for _ in xrange(self.height)]
        block[self.player_position[1]][self.player_position[0]] = 'O'
        for i, goal in enumerate(self.goals):
            block[goal[1]][goal[0]] = string.uppercase[i]
        print 'X' * (self.width + 2)
        for row in block:
            print 'X' + ''.join(row) + 'X'
        print 'X' * (self.width + 2)
        
    def get_tensor(self):
        return self.state.clone()
    
    def get_tensor_single_batch(self):
        t = self.get_tensor()
        t.resize_(1, *t.size())
        return t
