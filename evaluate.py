import time
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
import data_loader_evaluate_new
from torch.autograd import Variable
from model import _G_xvz, _G_vzx
from itertools import *
import pdb

dd = pdb.set_trace


class FCNetwork(nn.Module):
    def __init__(self):
        super(FCNetwork, self).__init__()
        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Linear(256, 220)
        self.hidden2 = nn.Linear(220, 160)
        self.hidden3 = nn.Linear(160, 128)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return x


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_list", type=str, default="/home/moktari/Moktari/profile_probset_faceNet.txt")
parser.add_argument("-b", "--batch_size", type=int, default=1)
parser.add_argument('--outf', default='./evaluate', help='folder to output images and model_ip checkpoints')
parser.add_argument('--restf', default='./results', help='folder to output images')
parser.add_argument('--modelf', default='./output', help='folder to output images and model_ip checkpoints')
# parser.add_argument('--modelf', default='./weights/latest_weights_Aug_20_6th_meeting', help='folder to output images and model_ip checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

args = parser.parse_args()
print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass
try:
    os.makedirs(args.restf)
except OSError:
    pass

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# need initialize!!
G_vzx = _G_vzx()

train_list = args.data_list
train_loader = torch.utils.data.DataLoader(
    data_loader_evaluate_new.ImageList(train_list, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)


def L1_loss(x, y):
    return torch.mean(torch.sum(torch.abs(x - y), 1))


x = torch.FloatTensor(args.batch_size, 3, 128, 128)
x_bar_bar_out = torch.FloatTensor(2, 3, 128, 128)

v_siz = 0
z_siz = 256 - v_siz
v = torch.FloatTensor(args.batch_size, v_siz)
z = torch.FloatTensor(args.batch_size, z_siz)

if args.cuda:
    G_vzx = torch.nn.DataParallel(G_vzx).cuda()
    x = x.cuda()
    x_bar_bar_out = x_bar_bar_out.cuda()
    v = v.cuda()
    z = z.cuda()

x = Variable(x)
x_bar_bar_out = Variable(x_bar_bar_out)
v = Variable(v)
z = Variable(z)


def load_pretrained_model(net, path, name):
    state_dict = torch.load('%s/%s' % (path, name))
    own_state = net.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print('not load weights %s' % name)
            continue
        own_state[name].copy_(param)
        print('load weights %s' % name)


load_pretrained_model(G_vzx, args.modelf, 'netG_vzx_last.pth')


batch_size = args.batch_size
cudnn.benchmark = True
G_vzx.eval()
# test_net.eval()

for i, (data, dream_tensor, id_name) in enumerate(train_loader):
    id_name = ''.join(id_name)
    img = data
    x.data.resize_(img.size()).copy_(img)
    x_bar_bar_out.data.zero_()
    z_bar = dream_tensor.float().cpu()
    exec('x_bar_bar_%d = G_vzx(z_bar)' % (0))
    for d in range(batch_size):
        x_bar_bar_out.data[0] = x.data[d]
        exec('x_bar_bar_out.data[1] = x_bar_bar_%d.data[0]' % (0))
        vutils.save_image(x_bar_bar_out.data, '%s/probe_%s.png' % (args.outf, id_name), nrow=2, normalize=True,
                          pad_value=255)
        vutils.save_image(x_bar_bar_out.data[1], '%s/%s.png' % (args.restf, id_name), nrow=1, normalize=True,
                          pad_value=255)
