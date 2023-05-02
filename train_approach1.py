import time
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
from matplotlib import cm

import data_loader_new
from torch.autograd import Variable
from model_1 import _G_xvz, _G_vzx, _D_xvs, FeatureExtractor, IP
from model_ip.light_cnn import LightCNN_29Layers_v2
from itertools import *
import pdb
from PIL import Image
from model_ip import common
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, utils

import numpy as np


# import cv2
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


device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")
dd = pdb.set_trace

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data_list", type=str,
                    default="/home/moktari/Moktari/13th_meeting_results/cmupie_profile_trainlist.txt")
parser.add_argument("-ns", "--snapshot", type=int, default=100)
parser.add_argument("-b", "--batch_size", type=int, default=64)  # 16
parser.add_argument("-lr", "--learning_rate", type=float, default=0.00002)  # 0.00002
parser.add_argument("-m", "--momentum", type=float, default=0.)  # 0.5, 0.0
parser.add_argument("-m2", "--momentum2", type=float, default=0.9)  # 0.999
parser.add_argument('--outf', default='./output', help='folder to output images and model_ip checkpoints')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--modelf', default='/home/moktari/Moktari/10th_meeting_results_Img2pose/trainset/CRv3',
                    help='folder to output images and model_ip checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--learn', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--ip_momentum', default=0.9, type=float, metavar='M',
                    help='ip_momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--num_classes', default=249, type=int,
                    metavar='N', help='number of classes (default: 243)')


# Initialize networks
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('LayerNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# need initialize!!
G_vzx = _G_vzx()
D_xvs = _D_xvs()
IP = LightCNN_29Layers_v2(num_classes=args.num_classes)

# G_vzx.apply(weights_init)
# D_xvs.apply(weights_init)
# IP.apply(weights_init)


train_list = args.data_list
train_loader = torch.utils.data.DataLoader(
    data_loader_new.ImageList(train_list, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip()])),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, drop_last=True)


def L1_loss(x, y):
    return torch.mean(torch.sum(torch.abs(x - y), 1))


v_siz = 0
z_siz = 256 - v_siz
x1 = torch.FloatTensor(args.batch_size, 3, 128, 128)

if args.cuda:
    G_vzx = torch.nn.DataParallel(G_vzx.cuda(), device_ids=[0, 1])
    D_xvs = torch.nn.DataParallel(D_xvs.cuda(), device_ids=[0, 1])
    IP = torch.nn.DataParallel(IP.cuda(), device_ids=[0, 1])
    x1 = x1.cuda()
x1 = Variable(x1)
params = []
for name, value in IP.named_parameters():
    if 'bias' in name:
        if 'fc2' in name:
            params += [{'params': value, 'ip_lr': 20 * args.learn, 'weight_decay': 0}]
        else:
            params += [{'params': value, 'ip_lr': 2 * args.learn, 'weight_decay': 0}]
    else:
        if 'fc2' in name:
            params += [{'params': value, 'ip_lr': 10 * args.learn}]
        else:
            params += [{'params': value, 'ip_lr': 1 * args.learn}]


def adjust_learning_rate(optimizer, epoch):
    scale = 0.457305051927326
    step = 10
    ip_lr = args.learn * (scale ** (epoch // step))
    print('learn_rate: {}'.format(ip_lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change ip_lr')
        for param_group in optimizer.param_groups:
            param_group['ip_lr'] = param_group['ip_lr'] * scale


def load_model(net, path, name, grad_off=False):
    state_dict = torch.load('%s/%s' % (path, name))
    own_state = net.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print('not load weights %s' % name)
            continue
        if grad_off:
            param.require_grad = False
        own_state[name].copy_(param)
        print('load weights %s' % name)


load_model(G_vzx, args.outf, 'netG_vzx_last.pth')
load_model(D_xvs, args.outf, 'netD_xvs_last.pth')

lr = args.learning_rate
ourBetas = [args.momentum, args.momentum2]
batch_size = args.batch_size
snapshot = args.nsnapshot
start_time = time.time()

G_vzx_solver = optim.Adam(G_vzx.parameters(), lr=lr, betas=ourBetas)
D_xvs_solver = optim.Adam(D_xvs.parameters(), lr=lr, betas=ourBetas)
IP_optimizer = torch.optim.SGD(params, args.learn, momentum=args.ip_momentum, weight_decay=args.weight_decay)

cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
adversarial_criterion = nn.BCELoss()
adversarial_criterion = adversarial_criterion.cuda()
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
feature_extractor = feature_extractor.cuda()

# feature_extractor_ip = torchvision.models.lightCNN(pretrained=True)
# feature_extractor_ip = feature_extractor_ip.cuda()

criterionL1 = nn.L1Loss().cuda()
Train_Loss = []

for epoch in range(args.epochs):
    T_loss = []
    for i, (view1, data1, dream_tensor) in enumerate(train_loader):
        adjust_learning_rate(IP_optimizer, epoch)
        eps = random.uniform(0, 1)
        tmp = random.uniform(0, 1)
        reconstruct_fake = False
        if tmp < 0.5:
            reconstruct_fake = True

        D_xvs.zero_grad()
        IP.zero_grad()
        G_vzx.zero_grad()

        # get x-->real image v--> view and z-->random vector
        x1.data.resize_(data1.size()).copy_(data1)

        z = dream_tensor.float().cuda()
        x_bar = G_vzx(z)  # random z to generate img x_bar

        x_hat = eps * x1.data + (1 - eps) * x_bar.data  # interpolation of x_bar and x1
        x_hat = Variable(x_hat, requires_grad=True)
        D_x_hat_s = D_xvs(x_hat)

        grads = autograd.grad(outputs=D_x_hat_s,
                              inputs=x_hat,
                              grad_outputs=torch.ones(D_x_hat_s.size()).cuda(),
                              retain_graph=True,
                              create_graph=True,
                              only_inputs=True)[0]
        grad_norm = grads.pow(2).sum().sqrt()
        gp_loss = torch.mean((grad_norm - 1) ** 2)  # gradient with v1

        d_real = D_xvs(x1)
        d_real = torch.sigmoid(d_real)

        d_gen = D_xvs(x_bar)
        d_gen = torch.sigmoid(d_gen)

        adv_loss = adversarial_criterion(d_real, torch.ones(d_real.shape).cuda()) + adversarial_criterion(d_gen,
                                                                                                          torch.zeros(
                                                                                                              d_gen.shape).cuda())
        adv_loss = adv_loss.mean()
        d_xvs_loss = 10. * gp_loss + 0.1 * adv_loss  # x1 real sample, x_bar fake sample
        D_xvs.zero_grad()

        out_sr, feat_sr = IP(x_bar)
        with torch.no_grad():
            out_hr, feat_hr = IP(x1.detach())

        IP_loss = F.mse_loss(feat_sr, feat_hr) + F.mse_loss(out_sr, out_hr)
        IP.zero_grad()

        # "L1 Loss calculation"
        GAN_l1 = L1_loss(x_bar, x1)

        # "Perceptual Loss calculation"
        real_features = feature_extractor(x1)
        fake_features = feature_extractor(x_bar)
        GAN_Vgg = L1_loss(fake_features, real_features)
        GAN_Vgg = GAN_Vgg.mean()

        # "adversarial loss calculation"
        g_cost_d = adversarial_criterion(d_gen, torch.ones(d_gen.shape).cuda())
        g_cost_d = g_cost_d.mean()

        # Total Loss Function
        g_vzx_loss = 1.5 * GAN_l1 + 0.1 * g_cost_d + 0.02 * GAN_Vgg + 0.001 * IP_loss

        G_vzx.zero_grad()
        d_xvs_loss.backward(retain_graph=True)
        g_vzx_loss.backward(retain_graph=True)
        IP_loss.backward(retain_graph=True)

        D_xvs_solver.step()
        G_vzx_solver.step()
        IP_optimizer.step()
        T_loss.append(g_vzx_loss.data.item())
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss_D: %.4f, loss_GL1: %.4f, loss_IP: %.4f, loss_G: %.4f" % (
            epoch, i, len(data1), time.time() - start_time, d_xvs_loss.data.item(), GAN_l1.data.item(),
            IP_loss.data.item(),
            g_vzx_loss.data.item()))
        if i % snapshot == snapshot - 1:
            vutils.save_image(x_bar.data,
                              '%s/x_bar_epoch_%03d_%04d.png' % (args.outf, epoch, i), normalize=True)
            vutils.save_image(x1.data,
                              '%s/x1_epoch_%03d_%04d.png' % (args.outf, epoch, i), normalize=True)
            np.save("FC_L1_Loss_fix_frontal", np.array(Train_Loss))
    if epoch % 5 == 0:
        torch.save(G_vzx.state_dict(), '%s/netG_vzx_epoch_%d_%d.pth' % (args.outf, epoch, i))
        torch.save(D_xvs.state_dict(), '%s/netD_xvs_epoch_%d_%d.pth' % (args.outf, epoch, i))
        torch.save(IP.state_dict(), '%s/netIP_epoch_%d_%d.pth' % (args.outf, epoch, i))
    Train_Loss.append(sum(T_loss) / len(T_loss))
torch.save({
    'G_vzx_optimizer_state_dict': G_vzx.state_dict(),
    'D_xvs_optimizer_state_dict': D_xvs.state_dict(),
    'IP_optimizer_state_dict': IP.state_dict(),
}, args.outf + "/Models/" + str(epoch) + '.pt')
torch.save(G_vzx.state_dict(), '%s/netG_vzx_last.pth' % (args.outf))
torch.save(D_xvs.state_dict(), '%s/netD_xvs_last.pth' % (args.outf))
torch.save(IP.state_dict(), '%s/netIP_last.pth' % (args.outf))
