import os, sys
import numpy as np
from PIL import Image
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pdb
import struct as st
from glob import glob

dd = pdb.set_trace

views = ['050', '051', '140']

pi = 3.1416  # 180 degree
d_60 = pi / 3
d_15 = pi / 12
d_range = pi / 36  # 5 degree

d_45 = d_60 - d_15
d_30 = d_45 - d_15

feat_path = '/home/moktari/Moktari/13th_meeting_results/train_features.bin'
frontal_path = '/home/moktari/Moktari/13th_meeting_results/cmupie_train_frontalset/*/*.png'


def load_feat(feat_file=feat_path):  # THIS FILE IS ADDED IN THE ZIP
    feats = list()
    print('loading feats')
    with open(feat_file, 'rb') as in_f:
        feat_num, feat_dim = st.unpack('ii', in_f.read(8))
        for i in range(feat_num):
            feat = np.array(st.unpack('f' * feat_dim, in_f.read(4 * feat_dim)))
            feats.append(feat)
    print(len(feats))
    return feats


def read_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((128, 128), Image.ANTIALIAS)
    return img


def calc_label_cmu(a):
    labels = {2: 0, 3: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 11: 7, 12: 8, 13: 9, 14: 10, 15: 11, 17: 12, 18: 13, 19: 14,
              20: 15, 21: 16, 22: 17, 23: 18, 24: 19, 25: 20, 26: 21, 27: 22,
              28: 23, 29: 24, 30: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35,
              41: 36, 43: 37, 44: 38, 45: 39, 46: 40, 47: 41, 48: 42,
              50: 43, 51: 44, 52: 45, 53: 46, 54: 47, 55: 48, 56: 49, 57: 50, 58: 51, 59: 52, 60: 53, 61: 54, 62: 55,
              63: 56, 64: 57, 65: 58, 66: 59, 67: 60, 68: 61, 69: 62, 70: 63, 71: 64, 72: 65, 73: 66,
              74: 67, 75: 68, 76: 69, 77: 70, 78: 71, 79: 72, 80: 73, 81: 74, 82: 75, 83: 76, 84: 77, 85: 78, 86: 79,
              87: 80, 88: 81, 89: 82, 90: 83, 91: 84, 92: 85, 93: 86, 94: 87, 95: 88, 96: 89, 97: 90,
              98: 91, 99: 92, 100: 93, 101: 94, 102: 95, 103: 96, 104: 97, 105: 98, 106: 99, 107: 100, 108: 101,
              109: 102, 110: 103, 111: 104, 112: 105, 113: 106, 114: 107, 115: 108, 116: 109,
              117: 110, 118: 111, 119: 112, 120: 113, 121: 114, 122: 115, 123: 116, 124: 117, 125: 118, 126: 119,
              127: 120, 128: 121, 129: 122, 130: 123, 131: 124, 132: 125, 133: 126, 134: 127,
              135: 128, 136: 129, 137: 130, 138: 131, 139: 132, 140: 133, 141: 134, 142: 135, 143: 136, 144: 137,
              145: 138, 146: 139, 147: 140, 148: 141, 149: 142, 150: 143, 151: 144, 152: 145,
              153: 146, 154: 147, 155: 148, 156: 149, 157: 150, 158: 151, 159: 152, 160: 153, 161: 154, 162: 155,
              163: 156, 164: 157, 165: 158, 166: 159, 167: 160, 168: 161, 169: 162, 170: 163,
              171: 164, 172: 165, 173: 166, 174: 167, 175: 168, 176: 169, 177: 170, 178: 171, 179: 172, 180: 173,
              181: 174, 182: 175, 183: 176, 184: 177, 185: 178, 186: 179, 187: 180, 188: 181,
              189: 182, 190: 183, 191: 184, 192: 185, 193: 186, 194: 187, 195: 188, 196: 189, 197: 190, 198: 191,
              199: 192, 200: 193, 201: 194, 202: 195, 203: 196, 204: 197, 205: 198, 206: 199,
              207: 200, 208: 201, 209: 202, 210: 203, 211: 204, 212: 205, 214: 206, 215: 207, 216: 208, 217: 209,
              218: 210, 219: 211, 220: 212, 221: 213, 222: 214, 223: 215, 224: 216, 225: 217,
              226: 218, 227: 219, 228: 220, 229: 221, 230: 222, 231: 223, 232: 224, 233: 225, 234: 226, 235: 227,
              236: 228, 237: 229, 238: 230, 239: 231, 240: 232, 241: 233, 242: 234, 243: 235,
              244: 236, 245: 237, 246: 238, 247: 239, 248: 240, 249: 241, 250: 242}

    return labels.get(a)


def get_multiPIE_img(img_path):
    tmp = random.randint(0, 2)
    view2 = tmp

    view = views[tmp]

    token = img_path.split('/')
    name = token[-1]

    token = name.split('_')
    ID = token[0]
    status = token[2]
    bright = token[4]

    img2_path = '/home/n-lab/moktari/data/CMUPIE/train/' + ID + '/' + ID + '_01_' + status + '_' + view + '_' + bright + '_crop_128.png'
    img2 = read_img(img2_path)
    img2 = img2.resize((128, 128), Image.ANTIALIAS)
    return view2, img2


def get_frontal():
    frontal_list = glob(frontal_path)
    frontal_dict = dict()
    for img in frontal_list:
        fid = img.split('/')[-2]
        req = img.split('_')[-3]
        if req == '08':
            if fid not in frontal_dict.keys():
                frontal_dict[fid] = [img]
            else:
                frontal_dict[fid].append(img)
    return frontal_dict


def get_300w_LP_img(img_path):
    right = img_path.find('_128.jpg')
    for i in range(right - 1, 0, -1):
        if img_path[i] == '_':
            left = i
            break

    view2 = -1
    while (view2 < 0):
        tmp = random.randint(0, 17)
        new_txt = img_path[:left + 1] + str(tmp) + '_128_pose_shape_expression_128.txt'
        new_txt = new_txt.replace("crop_0907", "300w_LP_size_128")

        if os.path.isfile(new_txt):
            param = np.loadtxt(new_txt)
            yaw = param[1]
            if yaw < -d_60 or yaw > d_60:
                view2 = -1
            elif yaw >= -d_60 and yaw < -d_60 + d_range:
                view2 = 0
            elif yaw >= -d_45 - d_range and yaw < -d_45 + d_range:
                view2 = 1
            elif yaw >= -d_30 - d_range and yaw < -d_30 + d_range:
                view2 = 2
            elif yaw >= -d_15 - d_range and yaw < -d_15 + d_range:
                view2 = 3
            elif yaw >= -d_range and yaw < d_range:
                view2 = 4
            elif yaw >= d_15 - d_range and yaw < d_15 + d_range:
                view2 = 5
            elif yaw >= d_30 - d_range and yaw < d_30 + d_range:
                view2 = 6
            elif yaw >= d_45 - d_range and yaw < d_45 + d_range:
                view2 = 7
            elif yaw >= d_60 - d_range and yaw <= d_60:
                view2 = 8

    new_img = img_path[:left + 1] + str(tmp) + '_128.jpg'
    img2 = read_img(new_img)
    img2 = img2.resize((128, 128), Image.ANTIALIAS)

    return view2, img2


class ImageList(data.Dataset):
    def __init__(self, list_file, transform=None, is_train=True, img_shape=[128, 128]):
        img_list = [line.rstrip('\n') for line in open(list_file)]
        print(img_list[0], 'img_list')
        print('total %d images' % len(img_list))
        self.img_list = img_list
        self.feats = load_feat()
        self.transform = transform
        self.is_train = is_train
        self.img_shape = img_shape
        self.transform_img = transforms.Compose([self.transform])
        self.frontal_dict = get_frontal()

    def __getitem__(self, index):
        img1_path = self.img_list[index]
        token = img1_path.split(' ')
        feat_index = int(token[0])
        img1_fpath = token[1]
        z = np.asarray(self.feats[feat_index - 1], dtype=float)
        # fid = img1_fpath.split('/')[-2]###cmupie_trainlist_FSA_Net
        # fid = img1_fpath.split('/')[-2]###cmupie_trainlist_HopeNet_Net
        # fid = img1_fpath.split('/')[-3]###cmupie_trainlist_Img2pose_Net
        fid = img1_fpath.split('/')[-2]  ###cmupie_trainlist_HyperNet
        img1_fpath = random.choice(self.frontal_dict[fid])
        img1 = read_img(img1_fpath)
        view1 = img1_fpath.split('/')[-1]
        view1 = view1.split('_')[3]
        view1 = views.index(view1)

        if self.transform_img is not None:
            img1 = self.transform_img(img1)  # [0,1], c x h x w
            # img2 = self.transform_img(img2)

        return view1, img1, z

    def __len__(self):
        return len(self.img_list)
