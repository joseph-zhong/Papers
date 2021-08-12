from __future__ import print_function
import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

NCLASSES = 10

def NORM(x):
    return 1.0*x/NCLASSES - 0.5
def UNORM(x):
    return (x + 0.5)*NCLASSES

def gen_data(sequence_length=50, batch_size=1):
    data = np.concatenate((np.random.choice(2, (batch_size, sequence_length, 1), True, (0.9, 0.1)), 
                           NORM(np.random.choice(NCLASSES, (batch_size, sequence_length, 1)))), axis=2)
    data[:, 0, 0] = 1
    gt = np.zeros((batch_size, sequence_length), dtype=data.dtype)
    for b in range(batch_size):
        for t in range(sequence_length):
            gt[b, t] = data[b, t, 1] if data[b, t, 0] else gt[b, t-1]
    return data, gt


class SimpleLatch(nn.Module):
    def __init__(self):
        super(SimpleLatch, self).__init__()
        self.forget = nn.Linear(2, 1)
        self.input = nn.Linear(2, 1)
        self.block = nn.Linear(2, 1)
        self.output = nn.Linear(2, 1)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input):
        outputs = []
        c_t = Variable(torch.zeros(input.size(0), 1).double(), requires_grad=True)

        for input_t in input.chunk(input.size(1), dim=1):
            input_t = input_t.squeeze(1)
            fg = self.sig(self.forget(input_t))
            ig = self.sig(self.input(input_t))
            bg = self.tanh(self.block(input_t))
            og = self.sig(self.output(input_t))
            c_t = c_t*fg
            bg = bg*ig
            c_t = bg + c_t
            oo = og*self.tanh(c_t)
            outputs += [oo.unsqueeze(1)]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


def train_em_up(latch):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(latch.parameters(), lr=1e-2)
    for i in range(1000):
        data, gt = gen_data(batch_size=256, sequence_length=50)
        data = Variable(torch.from_numpy(data).double(), requires_grad=True)
        gt = Variable(torch.from_numpy(gt).double(), requires_grad=False)
        out = latch(data)
        loss = criterion(out, gt)
        print('ITER[%d] loss: %f' % (i, loss.data.numpy()[0]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
def test_em(latch):
    data, gt = gen_data()
    data = Variable(torch.from_numpy(data).double(), requires_grad=False)
    gt = Variable(torch.from_numpy(gt).double(), requires_grad=False)
    out = latch(data)
    for t in range(data.size(1)):
        print("[%3d] %d %d -> %5.2f (%d)" % (t, int(data.data.numpy()[0, t, 0]), int(UNORM(data.data.numpy()[0, t, 1])), UNORM(out.data.numpy()[0, t]), int(UNORM(gt.data.numpy()[0, t]))))


if __name__ == '__main__':
    # set ramdom seed to 0
    np.random.seed(1)
    torch.manual_seed(0)
    # build the model
    latch = SimpleLatch()
    latch.double()
    latch.train()
    train_em_up(latch)
    latch.eval()
    test_em(latch)
    
