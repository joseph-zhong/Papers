#!/usr/bin/env python
import torch
from environment import Environment
from torch.autograd import Variable
import random

use_cuda = torch.cuda.is_available()

width = 9
height = 9

e = Environment(
        width = width,
        height = height,
        start_position = [0,0], #[random.randrange(9), random.randrange(9)],
        goals = [[8,8]], #[[random.randrange(9), random.randrange(9)]],
        use_cuda = use_cuda)
m = torch.load('checkpoint_dr.model')

run = True

while True:
    e.draw()
    t = e.get_tensor_single_batch()
    q = m(Variable(t))
    print 'reward', e.smooth_reward()
    print 'left  ', q.data.cpu().numpy()[0,0]
    print 'up    ', q.data.cpu().numpy()[0,1]
    print 'right ', q.data.cpu().numpy()[0,2]
    print 'down  ', q.data.cpu().numpy()[0,3]
    print 'stay  ', q.data.cpu().numpy()[0,4]
    if run:
        a = m.greedy_action(t)
        e.take_action(a)
    cmd = raw_input()
    args = cmd.split()
    if len(args):
        if args[0] == 'O':
            e.set_player([int(args[1]), int(args[2])])
    
        if args[0] == 'G':
            e.set_goal([int(args[1]), int(args[2])], 0)
        
        if args[0] in 'qQ':
            break
        
        if args[0] in 'wW':
            e.take_action(torch.LongTensor([[1]]))
        
        if args[0] in 'aA':
            e.take_action(torch.LongTensor([[0]]))
        
        if args[0] in 'sS':
            e.take_action(torch.LongTensor([[3]]))
        
        if args[0] in 'dD':
            e.take_action(torch.LongTensor([[2]]))
        
        if args[0] == 'run':
            run = int(args[1])
