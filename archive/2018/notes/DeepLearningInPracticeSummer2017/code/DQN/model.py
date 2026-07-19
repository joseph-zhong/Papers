import torch
import random
nn = torch.nn
from torch.autograd import Variable

conv1_hidden=16
conv2_hidden=32
conv3_hidden=32
fc1_hidden=64

class CrawlerDQN(nn.Module):
    
    def __init__(self, num_actions=5):
        super(CrawlerDQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, conv1_hidden, kernel_size=3, stride=1),
            nn.BatchNorm2d(conv1_hidden),
            nn.ReLU(),
            nn.Conv2d(conv1_hidden, conv2_hidden, kernel_size=3, stride=1),
            nn.BatchNorm2d(conv2_hidden),
            nn.ReLU(),
            nn.Conv2d(conv2_hidden, conv3_hidden, kernel_size=3, stride=1),
            nn.BatchNorm2d(conv3_hidden),
            nn.ReLU())
        
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * conv3_hidden, fc1_hidden),
            nn.ReLU(),
            nn.Linear(fc1_hidden, num_actions))
        
        self.num_actions = num_actions
        
        for param in self.parameters():
            param.data.normal_(0.0, 0.01)
    
    def forward(self, x):
        conv = self.conv(x)
        conv = conv.view(-1,3*3*conv3_hidden)
        return self.fc(conv)
    
    def epsilon_greedy_action(self, x, epsilon=0.5):
        LongTensor = (
                torch.cuda.LongTensor if next(self.parameters()).is_cuda
                else torch.LongTensor)
        r = random.random()
        if r > epsilon:
            return self.greedy_action(x)
        else:
            return LongTensor([[random.randrange(self.num_actions)]])
    
    def greedy_action(self, x):
        FloatTensor = (
                torch.cuda.FloatTensor if next(self.parameters()).is_cuda
                else torch.FloatTensor)
        #y = self(Variable(x, volatile=True).type(FloatTensor))
        x = Variable(x).type(FloatTensor)
        y = self(x)
        return y.data.max(1)[1].view(1,1)

