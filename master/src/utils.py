import argparse
import copy
import os
import random

import cv2
import pandas as pd
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import *
from master.models.DHR_Net import *
from master.parameters.parameters import *
import matplotlib.pyplot as plt
import scipy.io

# 设置随机种子，防止每次运行结果不同
seed = 222
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


def SetupImageFolders():
    test_path = '../save_folder/results/encoded_images/'
    if os.path.exists(test_path):
        os.system('rd /s /q \"' + '../save_folder/results/encoded_images/\"')
    os.makedirs('../save_folder/results/encoded_images/' + '/kwn/')
    os.makedirs('../save_folder/results/encoded_images/' + '/unw/')

def GetDistance(input1, input2, hyper_para):
    dist = None
    if hyper_para.dist_type == 'L1':
        dist = torch.mean(torch.abs((input1 - input2)))
    elif hyper_para.dist_type == 'L2':
        dist = torch.sqrt(torch.mean((input1 - input2) * (input1 - input2)))
    elif hyper_para.dist_type == 'D':
        D = torch.load('../../temp_folder/D.pth')
        dist = D(input1)
    else:
        print('ERROR: Unidentified distance type')

    return dist


class FilterableImageFolder(ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            valid_classes: List = None
    ):
        self.valid_classes = valid_classes
        super(FilterableImageFolder, self).__init__(root, transform, target_transform, loader, is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        # 增加了这下面这句
        classes = [valid_class for valid_class in classes if valid_class in self.valid_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def get_args():
    parser = argparse.ArgumentParser(description='Train DHR Net')
    parser.add_argument('--lr', default=0.05, type=float, help="learning rate")
    parser.add_argument('--epochs', default=100, type=int, help="Number of training epochs")
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size")
    parser.add_argument('--dataset_dir', default="E/UESTC/ATR/MSTAR数据集/datasets", type=str,
                        help="Number of members in ensemble")
    parser.add_argument('--no_closed', default=10, type=int, help="Number of classes in dataset")
    parser.add_argument('--no_open', default=2, type=int, help="Number of classes out of dataset")
    parser.add_argument('--means', nargs='+', default=0.5, type=float, help="channelwise means for normalization")
    parser.add_argument('--stds', nargs='+', default=0.5, type=float, help="channelwise std for normalization")
    parser.add_argument('--momentum', default=0.9, type=float, help="momentum")
    parser.add_argument('--weight_decay', default=0.0005, type=float, help="weight decay")
    parser.add_argument('--save_path', default="./save_models/mstar10", type=str,
                        help="Path to save the ensemble weights")

    parser.set_defaults(argument=True)

    return parser.parse_args()


def acc_score(true_lab, false_lab):
    score = (true_lab == false_lab).sum()
    return torch.true_divide(score, len(true_lab))


def train_model(net, traindataloader, train_rate, ce_criterion, mse_criterion, num_epochs):
    # optimizer = optim.SGD(net.parameters(), lr=hyper_para.lr, momentum=hyper_para.momentum, weight_decay=hyper_para.weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    optimizer = torch.optim.Adam(net.parameters(), lr=hyper_para.lr)
    batch_num = len(traindataloader)
    train_batch_num = round(batch_num * train_rate)
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = 100
    best_acc = 0.0

    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    ttt = 0
    vvv = 0

    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for step, (b_x, b_y) in enumerate(traindataloader):
            if hyper_para.use_gpu:
                b_x, b_y = b_x.cuda(), b_y.cuda()
            if step < train_batch_num:
                ttt += 1
                net.train()
                # forward + backward + optimize
                logits, reconstruct, _ = net(b_x)
                pre_lab = torch.argmax(logits, 1)
                loss_cc = ce_criterion(logits, b_y)
                loss_rc = mse_criterion(reconstruct, b_x)  # 重建误差
                loss_tl = hyper_para.alpha * loss_rc + (1 - hyper_para.alpha) * loss_cc
                tr_acc = acc_score(b_y.data, pre_lab)
                # zero the parameter gradients
                optimizer.zero_grad()
                loss_tl.backward()
                optimizer.step()
                train_loss_all.append(loss_rc.item())
                train_acc_all.append(tr_acc.item())
            else:
                vvv += 1
                net.eval()
                logits, reconstruct, _ = net(b_x)
                loss_cc = ce_criterion(logits, b_y)
                pre_lab = torch.argmax(logits, 1)
                val_acc_c = acc_score(b_y.data, pre_lab)
                loss_rc = mse_criterion(reconstruct, b_x)  # 重建误差
                loss_tl = hyper_para.alpha * loss_rc + (1 - hyper_para.alpha) * loss_cc
                val_loss_all.append(loss_tl.item())
                val_acc_all.append(val_acc_c.item())

        train_acc = torch.Tensor(train_acc_all[-20:]).mean()
        train_loss = torch.Tensor(train_loss_all[-20:]).mean()
        val_acc = torch.Tensor(val_acc_all[-20:]).mean()
        val_loss = torch.Tensor(val_loss_all[-20:]).mean()
        print('{} Train Loss:{:.4f} Train Acc:{:.4f}'.format(epoch, train_loss, train_acc))
        print('{} Val Loss:{:.4f} Val Acc:{:.4f}'.format(epoch, val_loss, val_acc))
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(net.state_dict())

    net.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(
        data={"epoch": range(len(train_loss_all)),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              })
    val_process = pd.DataFrame(
        data={"epoch": range(len(val_loss_all)),
              "val_loss_all": val_loss_all,
              "val_acc_all": val_acc_all
              })
    return net, train_process, val_process
    '''
    for i, data in enumerate(traindataloader):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits, reconstruct, _ = net(inputs)

        cls_loss = cls_criterion(logits, labels)

        reconst_loss = reconst_criterion(reconstruct, inputs)

        if (torch.isnan(cls_loss) or torch.isnan(reconst_loss)):
            print("Nan at iteration ", iter)
            cls_loss = 0.0
            reconst_loss = 0.0
            logits = 0.0
            reconstruct = 0.0
            continue

        loss = cls_loss + reconst_loss

        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()
        total_cls_loss = total_cls_loss + cls_loss.item()
        total_reconst_loss = total_reconst_loss + reconst_loss.item()

        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        iter = iter + 1

    return [(100 * (correct / total)), (total_cls_loss / iter), (total_reconst_loss / iter), (total_loss / iter)], total_loss
    '''


def test():
    print('#' * 10)
    print('Testing...')
    SetupImageFolders()
    net = DHRNet(hyper_para.no_closed)
    tensor2pil = transforms.ToPILImage()
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    if hyper_para.use_gpu:
        net = net.cuda()
    # 加载模型
    net.load_state_dict(torch.load('../save_folder/models/DHR_' + str(hyper_para.epochs) + '.pth'))

    knwSet = FilterableImageFolder(root=os.path.join(hyper_para.dataset_dir, "train"),
                                     transform=hyper_para.trainTransforms,
                                     valid_classes=hyper_para.knw_labels)
    unkSet = FilterableImageFolder(root=os.path.join(hyper_para.dataset_dir, "test"),
                                    transform=hyper_para.trainTransforms,
                                    valid_classes=hyper_para.unk_labels)

    knwLoader = torch.utils.data.DataLoader(knwSet, batch_size=1, shuffle=True)
    unkLoader = torch.utils.data.DataLoader(unkSet, batch_size=1, shuffle=False)
    correct = 0
    knwTotal = len(knwLoader)
    unkTotal = len(unkLoader)
    kwn_mse = np.zeros((knwTotal,))
    all_test_mse = np.zeros((knwTotal + unkTotal,))
    all_test_label = np.zeros((knwTotal + unkTotal,))
    all_test_score = np.zeros((knwTotal + unkTotal, hyper_para.no_closed))
    pred_label_test_g = np.zeros((knwTotal,))
    unk_unk_mse = np.zeros((unkTotal,))

    vvv = 0
    iii = 0
    i = 0
    k = 0

    for data, target in knwLoader:
        if hyper_para.use_gpu:
            data, target = data.cuda(), target.cuda()
        logits, reconstruct, _ = net(data)
        pre_lab = torch.argmax(logits, 1)
        correct += pre_lab.eq(target.data).cpu().sum()

        temp_img = 0.5 * reconstruct.view(hyper_para.image_channel, hyper_para.image_size, hyper_para.image_size) + 0.5
        output = tensor2pil(temp_img.data.cpu())
        output = testTransform(output)
        output = output.view(1, hyper_para.image_channel, hyper_para.image_size, hyper_para.image_size)

        temp_mse = GetDistance(data.cpu(), output, hyper_para)
        temp_mse = temp_mse.data * torch.ones(1, 1)
        temp_mse = temp_mse.numpy()

        for l in range(hyper_para.image_channel):
            data[:, l, :, :] = data[:, l, :, :] * 0.5 + 0.5
        if vvv < 50 and iii % 50 == 0:
            cv2.imwrite('../save_folder/results/encoded_images/kwn/' + str(vvv) +
                        '_real_' + str(int(target.data.cpu().numpy())) + '.jpg',
                        np.reshape(np.transpose(data.data.cpu().numpy()) * 255,
                                   (hyper_para.image_size, hyper_para.image_size, hyper_para.image_channel))
                        )
            cv2.imwrite('../save_folder/results/encoded_images/kwn/' + str(vvv) +
                        '_zfake_' + str(int(target.data.cpu().numpy())) + '.jpg',
                        np.reshape(np.transpose(temp_img.data.cpu().numpy()) * 255,
                                   (hyper_para.image_size, hyper_para.image_size, hyper_para.image_channel))
                        )
            vvv += 1
        iii += 1

        pred_label_test_g[i] = temp_mse
        kwn_mse[i] = temp_mse
        all_test_mse[k] = temp_mse
        all_test_label[k] = target.data.cpu().numpy()
        all_test_score[k] = logits.data.cpu().numpy()
        k += 1
        i += 1

    vvv = 0
    iii = 0
    i = 0
    # 开集
    for data, target in unkLoader:
        if hyper_para.use_gpu:
            data, target = data.cuda(), target.cuda()
        logits, reconstruct, _ = net(data)
        temp_img = 0.5 * reconstruct.view(hyper_para.image_channel, hyper_para.image_size, hyper_para.image_size) + 0.5
        temp_scores = logits.data.cpu().numpy()

        output = tensor2pil(temp_img.data.cpu())
        output = testTransform(output)
        output = output.view(1, hyper_para.image_channel, hyper_para.image_size, hyper_para.image_size)
        temp_mse = GetDistance(data.cpu(), output, hyper_para)
        temp_mse = temp_mse.data * torch.ones(1, 1)
        temp_mse = temp_mse.numpy()
        mse = temp_mse

        for l in range(hyper_para.image_channel):
            data[:, l, :, :] = data[:, l, :, :] * 0.5 + 0.5

        if vvv < 50 and iii % 50 == 0:
            cv2.imwrite('../save_folder/results/encoded_images/unw/' + str(vvv) +
                        '_real_' + str(int(target.data.cpu().numpy())) + '.jpg',
                        np.reshape(np.transpose(data.data.cpu().numpy()) * 255,
                                   (hyper_para.image_size, hyper_para.image_size, hyper_para.image_channel))
                        )
            cv2.imwrite('../save_folder/results/encoded_images/unw/' + str(vvv) +
                        '_zfake_' + str(int(target.data.cpu().numpy())) + '.jpg',
                        np.reshape(np.transpose(temp_img.data.cpu().numpy()) * 255,
                                   (hyper_para.image_size, hyper_para.image_size, hyper_para.image_channel))
                        )
            vvv += 1
        iii += 1
        unk_unk_mse[i] = mse
        all_test_mse[k] = mse
        all_test_label[k] = -1   # 未知的标签为-1
        all_test_score[k] = temp_scores
        k += 1
        i += 1
    # saving all the files
    scipy.io.savemat('../save_folder/results/mlosr_mse.mat', {'mlosr_mse': all_test_mse})
    scipy.io.savemat('../save_folder/results/mlosr_scores.mat', {'mlosr_scores': all_test_score})
    scipy.io.savemat('../save_folder/results/label.mat', {'label': all_test_label})
    scipy.io.savemat('../save_folder/results/encoded_images/kwn.mat', {'kwn': kwn_mse})
    scipy.io.savemat('../save_folder/results/encoded_images/unk_unk.mat', {'unk_unk': unk_unk_mse})

    '''
    
    net.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reconst_loss = 0.0
    iter = 0
    cls_criterion = nn.CrossEntropyLoss()
    reconst_criterion = nn.MSELoss()

    with torch.no_grad():  # 在确知不使用反向传播的情况下，可以减少内存消耗
        for data in testloader:
            images, labels = data
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            logits, reconstruct, _ = net(images)

            cls_loss = cls_criterion(logits, labels)

            reconst_loss = reconst_criterion(reconstruct, images)

            loss = cls_loss + reconst_loss

            total_loss = total_loss + loss.item()
            total_cls_loss = total_cls_loss + cls_loss.item()
            total_reconst_loss = total_reconst_loss + reconst_loss.item()

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            iter = iter + 1

    return [(100 * (correct / total)), (total_cls_loss / iter), (total_reconst_loss / iter),
            (total_loss / iter)], total_loss
'''


def train():
    trainSet = FilterableImageFolder(root=os.path.join(hyper_para.dataset_dir, "train"),
                                     transform=hyper_para.trainTransforms,
                                     valid_classes=hyper_para.knw_labels)

    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=hyper_para.batch_size,
                                              shuffle=True, pin_memory=True, drop_last=True)

    net = DHRNet(hyper_para.no_closed)
    if hyper_para.use_gpu:
        net = torch.nn.DataParallel(net.cuda())
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.L1Loss()

    net, train_process, val_process = train_model(
        net, trainLoader, hyper_para.train_rate, ce_criterion, mse_criterion, num_epochs=hyper_para.epochs
    )
    plt.figure(figsize=(12, 4))
    plt.subplot(2, 2, 1), plt.plot(train_process.epoch, train_process.train_loss_all, label="Train loss")
    plt.subplot(2, 2, 2), plt.plot(train_process.epoch, train_process.train_acc_all, label="Train acc")
    plt.subplot(2, 2, 3), plt.plot(val_process.epoch, val_process.val_loss_all, label="Val loss")
    plt.subplot(2, 2, 4), plt.plot(val_process.epoch, val_process.val_acc_all, label="Val acc")
    plt.show()
    torch.save(net.state_dict(), '../save_folder/models/DHR_' + str(hyper_para.epochs) + '.pth')  # 保存模型



