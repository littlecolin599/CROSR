import random

from torchvision import transforms
import numpy as np
import os

random.seed(1)


class Hyperparameters:
    def __init__(self):
        self.batch_size = 48
        self.epochs = 200
        self.lr = 0.05
        self.alpha = 0.8

        self.image_size = 128
        self.latent_size = 128 * 3 * 3  # 512
        self.train_rate = 0.8
        self.image_channel = 1

        self.mode = 'train'  # train or test
        self.dataset_dir = '../../datasets/mstar10'  # mstar
        self.save_path = '../save_models/mstar10'

        self.use_gpu = True

        self.dist_type = 'L2'
        self.model = 'cnn'
        self.momentum = 0.9
        self.weight_decay = 0.0005

        self.means = 0.5
        self.stds = 0.5

        self.no_total = 12
        self.no_closed = 10
        self.no_open = 2
        self.kwn, self.unk = GetKwnUnkClasses(self.no_total, self.no_closed, self.no_open, 'sequential')
        self.knw_labels, self.unk_labels = GetAllLabels(self.kwn, self.unk, self.dataset_dir)

        self.trainTransforms = transforms.Compose([
            # transforms.ColorJitter(brightness=0.5, hue=0.3),  # 改变图像亮度， 色调
            transforms.CenterCrop(self.image_size),
            transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.75, 1.0)),  # 随机旋转、水平垂直平移、比例缩放
            transforms.RandomHorizontalFlip(),   # 随机水平翻转
            transforms.RandomVerticalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(self.means, self.stds),  # 均值、方差
        ])

        self.testTransforms = transforms.Compose(
            [
                transforms.CenterCrop(self.image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds)]  # 均值、方差
        )


def GetKwnUnkClasses(no_total, no_closed, no_open, magic_word):
    if magic_word == 'sequential':
        kwn = np.asarray(range(no_closed))
        unk = no_closed + np.asarray(range(np.min((no_open, no_total - no_closed))))

    elif magic_word == 'random':
        rand_id = np.asarray(random.sample(range(no_total - no_closed), no_open))
        kwn = np.sort(np.asarray(random.sample(range(no_total), no_closed)))
        unk = np.asarray(np.where(np.in1d(np.asarray(range(no_total)), kwn) == False))[0, rand_id[0:no_open]]

    elif magic_word == 'manual':
        kwn = np.asarray([9])
        unk = np.asarray([9])

    else:
        print('ERROR: known unknown split type not available')

    return kwn, unk


def GetAllLabels(knw, unk, path):
    labels = []
    knw_labels = []
    unk_labels = []
    for file in os.listdir(path + '/train'):
        labels.append(file)
    for i in range(len(knw)):
        knw_labels.append(labels[knw[i]])
    for i in range(len(unk)):
        unk_labels.append(labels[unk[i]])
    print('knw_labels: ', knw_labels)
    print('unk_labels: ', unk_labels)
    return knw_labels, unk_labels


hyper_para = Hyperparameters()
