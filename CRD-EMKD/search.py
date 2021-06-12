"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time
import random

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn


from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader, get_dataloader_sample
from dataset.tinyimagenet import get_tiny_imagenet_dataloader, get_tiny_dataloader_sample

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # whether test
    parser.add_argument('--test', type=int, default=0, help='if testing, set --test 1')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_10_2', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2',
                                 'densenet100', 'densenet40',
                                 'resnext101', 'resnext29',
                                 'myresnet8', 'myresnet14', 'myresnet20', 'myresnet32', 'myresnet44', 'myresnet56', 'myresnet110',
                                 'mywrn_16_2',
                                 'myResNet18', 'myResNet34', 'myResNet50', 'myResNet101', 'myResNet152', 
                                 'myresnext101', 'myresnext29',])
    parser.add_argument('--model_t', type=str, default='resnet110',
                    choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                'resnet8x4', 'resnet32x4', 'wrn_10_2', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 
                                'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 
                                'MobileNetV2', 'ShuffleV1', 'ShuffleV2',
                                'densenet100', 'densenet40',
                                'resnext101', 'resnext29',
                                'myresnet8', 'myresnet14', 'myresnet20', 'myresnet32', 'myresnet44', 'myresnet56', 'myresnet110',
                                'mywrn_16_2',
                                'myResNet18', 'myResNet34', 'myResNet50', 'myResNet101', 'myResNet152', 
                                'myresnext101', 'myresnext29',])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # block depth
    parser.add_argument('--block_depth', type=str, default='3,3,3', help='the depth of each block')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    # use MC
    parser.add_argument('--mc', type=int, default=0, help='if using MC, set --mc 1')
    parser.add_argument('--mc_weight', type=float, default=0.001, help='weight for milestone checking')

    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    # mc hyper-parameter:
    parser.add_argument('--B1_weight', type=float, default=1.0)
    parser.add_argument('--B2_weight', type=float, default=1.0)
    parser.add_argument('--B3_weight', type=float, default=1.0)
    parser.add_argument('--B4_weight', type=float, default=1.0)

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    
    if opt.model_s[0:2] == 'my':
        iterations = opt.block_depth.split(',')
        opt.block_depth = list([])
        for it in iterations:
            opt.block_depth.append(int(it))
        opt = block_weight(opt, opt.block_depth)

    #opt.model_t = get_teacher_name(opt.path_t)
    print('teacher: ', opt.model_t)
    print('student:{}, arch:{}'.format(opt.model_s, opt.block_depth))

    # opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill, opt.gamma, opt.alpha, opt.beta, opt.trial)
    if opt.mc:
        opt.model_name = 'S:{}_{}_T:{}_{}_{}_mc_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.block_depth, opt.model_t, opt.dataset, opt.distill, opt.gamma, opt.alpha, opt.beta, opt.trial)
    else:
        opt.model_name = 'S:{}_{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.block_depth, opt.model_t, opt.dataset, opt.distill, opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

def remove_module_in_state_dict(filepath):
	state_dict = torch.load(filepath)['state_dict']
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v
	return new_state_dict

def load_teacher(opt, n_cls):
    print('==> loading teacher model')
    model_path = opt.path_t
    model_t = opt.model_t #get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)

    model.load_state_dict(torch.load(model_path)['model'])
    #model.load_state_dict(remove_module_in_state_dict(model_path))

    print('==> done')
    return model

def block_weight(opt, depth):
    block_depth = { 'resnet110':18, 'resnet56':9, 'resnet44':7, 'resnet32':5, 'resnet20':3, 'resnet14':2, 'resnet8':1,
                'myresnet110':18, 'myresnet56':9, 'myresnet44':7, 'myresnet32':5, 'myresnet20':3, 'myresnet14':2, 'myresnet8':1,
                'wrn_40_2':6, 'wrn_16_2':2, 'wrn_10_2':1, 'wrn_40_1':6, 'wrn_16_1':2, 'wrn_10_1':1,
                'mywrn_40_2':6, 'mywrn_16_2':2, 'mywrn_10_2':1, 'mywrn_40_1':6, 'mywrn_16_1':2, 'mywrn_10_1':1,
                'resnext101':11, 'resnext29':3,
                'myresnext101':11, 'myresnext29':3, 
                'myResNet18': 2, }
    original_depth = block_depth[opt.model_s]
    opt.B1_weight = original_depth / depth[0]
    opt.B2_weight = original_depth / depth[1]
    opt.B3_weight = original_depth / depth[2]
    if opt.model_s[0:8] == 'myResNet':
        opt.B4_weight = original_depth / depth[3]
    return opt
    
def train_search(opt):
    best_acc = 0

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 100
    elif opt.dataset == 'cifar10':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar10_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar10_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 10
    elif opt.dataset == 'imagenet':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data, n_cls = get_dataloader_sample(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_sample=False,
                                                                        k=opt.nce_k)
        else:
            train_loader, val_loader, n_data = get_imagenet_dataloader(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
        n_cls = 1000
    elif opt.dataset == 'tiny_imagenet':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data, n_cls = get_tiny_dataloader_sample(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_sample=True,
                                                                        k=opt.nce_k)
        else:
            train_loader, val_loader = get_tiny_imagenet_dataloader(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        )
        n_cls = 200
    else:
        raise NotImplementedError(opt.dataset)

    # model
    if opt.test == 0:
        model_t = load_teacher(opt, n_cls)      # for training
    elif opt.test == 1:
        print('==> loading teacher model')  # for testing
        if opt.model_s[0:2] == 'my':
            model_t = model_dict[opt.model_s](num_classes=n_cls, block_depth=opt.block_depth)
        else:
            model_t = model_dict[opt.model_s](num_classes=n_cls)
        model_t.load_state_dict(torch.load(opt.path_t)['model'])
        print('==> done')

    if opt.model_s[0:8] == 'myresnet' or opt.model_s[0:5] == 'mywrn' or opt.model_s[0:9] == 'myresnext' or opt.model_s[0:8]=='myResNet':
        model_s = model_dict[opt.model_s](num_classes=n_cls, block_depth=opt.block_depth)
    else:    
        model_s = model_dict[opt.model_s](num_classes=n_cls)

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    print('student Total params: %.6fM' % (sum(p.numel() for p in model_s.parameters())/1000000.0))
    
    if opt.test == 1:
        return     # for testing

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            if opt.mc:
                save_file = os.path.join(opt.save_folder, '{B1}_{B2}_{B3}_ckpt_epoch_{epoch}.pth'.format(B1=opt.B1_weight, B2=opt.B2_weight, B3=opt.B3_weight, epoch=epoch))
            else:
                save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print('best accuracy:', best_acc)
    
    
    with open('search-result.txt', 'a') as f:
        if opt.mc:
            if opt.model_s == 'myResNet18':
                f.write("T:{} => S:{}_{}, {}_mc, B1_weight:{}, B2_weight:{}, B3_weight:{}, B4_weight:{}, acc:{}\n".format(opt.model_t, opt.model_s, opt.block_depth, opt.distill, opt.B1_weight, opt.B2_weight, opt.B3_weight, opt.B4_weight, best_acc))            
            else:
                f.write("T:{} => S:{}_{}, {}_mc, B1_weight:{}, B2_weight:{}, B3_weight:{}, acc:{}\n".format(opt.model_t, opt.model_s, opt.block_depth, opt.distill, opt.B1_weight, opt.B2_weight, opt.B3_weight, best_acc))            
        else:
            f.write("T:{} => S:{}, {}, acc:{}\n".format(opt.model_t, opt.model_s, opt.distill, best_acc))

    
    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)
    return best_acc

def search():
    opt = parse_option()

    block_depth = { 'resnet110':18, 'resnet56':9, 'resnet44':7, 'resnet32':5, 'resnet20':3, 'resnet14':2, 'resnet8':1,
                    'myresnet110':18, 'myresnet56':9, 'myresnet44':7, 'myresnet32':5, 'myresnet20':3, 'myresnet14':2, 'myresnet8':1,
                    'wrn_40_2':6, 'wrn_16_2':2, 'wrn_10_2':1, 'wrn_40_1':6, 'wrn_16_1':2, 'wrn_10_1':1,
                    'mywrn_40_2':6, 'mywrn_16_2':2, 'mywrn_10_2':1, 'mywrn_40_1':6, 'mywrn_16_1':2, 'mywrn_10_1':1,
                    'resnext101':11, 'resnext29':3,
                    'myresnext101':11, 'myresnext29':3, 
                    'myResNet18': 2, }
    x_t, y_t, z_t = block_depth[opt.model_t], block_depth[opt.model_t], block_depth[opt.model_t]    # Teacher architecture
    x_s, y_s, z_s = block_depth[opt.model_s], block_depth[opt.model_s], block_depth[opt.model_s]    # Student architecture
    
    with open('search-result.txt', 'a') as f:
        f.write('Search for best student({}) architecture of teacher({}) using method({})\n'.format(opt.model_s, opt.model_t, opt.distill))

    x, y, z = x_s, y_s, z_s # searching
    x_opt, y_opt, z_opt = x, y, z
    best_acc = 0
    original_depth = block_depth[opt.model_s]

    while z >= 1 and y <= y_t:
        opt.block_depth = [x, y, z]
        print("Searching:{}, architecture:{}".format(opt.model_s, opt.block_depth))
        if opt.mc:
            opt.B1_weight = original_depth / float(x)
            opt.B2_weight = original_depth / float(y)
            opt.B3_weight = original_depth / float(z)
            print("B1:{}, B2:{}, B3:{}".format(opt.B1_weight, opt.B2_weight, opt.B3_weight))
        acc = train_search(opt)
        #acc = random.randint(0,100)
        #print("Searching:{}, architecture:{},{},{}, acc:{}".format(opt.model_s, x, y, z, acc))
        print("Searching:{}, architecture:{}, acc:{}".format(opt.model_s, opt.block_depth, acc))
        if acc > best_acc:
            best_acc = acc
            y_opt = y
            z_opt = z
        
        z = z - 1
        y = y + 4
        if y > y_t:
            break

    x, y, z = x_s+4, y_opt-1, z_opt
    while y >= 1 and x <= x_t:
        opt.block_depth = [x, y, z]
        print("Searching:{}, architecture:{}".format(opt.model_s, opt.block_depth))
        if opt.mc:
            opt.B1_weight = original_depth / float(x)
            opt.B2_weight = original_depth / float(y)
            opt.B3_weight = original_depth / float(z)
            print("B1:{}, B2:{}, B3:{}".format(opt.B1_weight, opt.B2_weight, opt.B3_weight))
        acc = train_search(opt)
        #acc = random.randint(0,100)
        #print("Searching:{}, architecture:{},{},{}, acc:{}".format(opt.model_s, x, y, z, acc))
        print("Searching:{}, architecture:{}, acc:{}".format(opt.model_s, opt.block_depth, acc))
        if acc > best_acc:
            best_acc = acc
            x_opt = x
            y_opt = y
        
        y = y - 1
        x = x + 4
        if x > x_t:
            break
    
    print("Teacher:{}, Student:{}, best architecture:{},{},{}, acc:{}".format(opt.model_t, opt.model_s, x_opt, y_opt, z_opt, best_acc))
    with open('search-result.txt', 'a') as f:
        if opt.mc:
            f.write("Teacher:{}, Student:{}, best architecture:{},{},{}, distill:{}_mc, acc:{}\n".format(opt.model_t, opt.model_s, x_opt, y_opt, z_opt, opt.distill, best_acc))
        else:
            f.write("Teacher:{}, Student:{}, best architecture:{},{},{}, distill:{}, acc:{}\n".format(opt.model_t, opt.model_s, x_opt, y_opt, z_opt, opt.distill, best_acc))
        f.write('\n')
            

if __name__ == '__main__':
    search()
