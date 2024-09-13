'''
Evaluate target domain performance.
'''

import argparse
import os, sys
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import network
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time
import json

from other_utils import eval_utils, misc

def image_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    te_txt_test = open(args.test_dset_path).readlines()

    dsets["test"] = ImageList(te_txt_test, transform_adapt=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs, shuffle=False,
        num_workers=args.worker, drop_last=False)

    return dset_loaders

def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net[0:8] == 'convnext':
        netF = network.ConvnextBase(convnext_name=args.net).cuda()
    elif args.net[0:4] == 'swin':
        netF = network.SwinBase(swin_name=args.net).cuda()
    netB = None
    netC = None

    # load weights
    if args.method_type == 'source':
        modelpathF = args.output_dir_src + '/source_F.pt'
    elif args.method_type == 'target':
        if args.method_name == 'proposed':
            modelpathF = args.final_output_dir + f'/target_adapt_F.pt'
        else:
            modelpathF = args.final_output_dir + f'/target_F_{args.final_savename}.pt'

    if args.method_type != 'pretrained':
        netF.load_state_dict(torch.load(modelpathF))
    netF.eval()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    if args.method_type == 'source':
        modelpathB = args.output_dir_src + '/source_B.pt'
        modelpathC = args.output_dir_src + '/source_C.pt'
    elif args.method_type == 'target':
        if args.method_name == 'proposed':
            modelpathB = args.final_output_dir + f'/target_adapt_B.pt'
            modelpathC = args.final_output_dir + f'/target_adapt_C.pt'
        else:
            modelpathB = args.final_output_dir + f'/target_B_{args.final_savename}.pt'
            modelpathC = args.final_output_dir + f'/target_C_{args.final_savename}.pt'
    netB.load_state_dict(torch.load(modelpathB))
    netC.load_state_dict(torch.load(modelpathC))
    netB.eval()
    netC.eval()

    # accuracy
    if args.return_output:
        _, _, _, all_labels, all_preds = eval_utils.cal_acc(dset_loaders['test'], netF, netB, netC, return_output=True)
        return all_labels, all_preds
    else:
        acc, accmean, acc_list = eval_utils.cal_acc(dset_loaders['test'], netF, netB, netC)
        acc_per_class = {f'acc_class_{k}_{names[args.t]}': v for k, v in enumerate(acc_list)}
        acc_consolidate = {f'acc_{names[args.t]}': acc, f'acc_classmean_{names[args.t]}': accmean}
        results_dict = {**acc_consolidate, **acc_per_class,}
        return results_dict


def print_args(args):
    s = '==========================================\n'
    for arg, content in args.__dict__.items():
        s += '{}:{}\n'.format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--method_name', type=str, default=None, help="method name")
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=50, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'domainnet-126'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet18, resnet50")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--ssl', type=float, default=0.0) 
    parser.add_argument('--issave', type=bool, default=True)
    # additional SHOT++ arguments
    parser.add_argument('--ps', type=float, default=0.0)
    parser.add_argument('--choice', type=str, default=None, choices=["maxp", "ent", "marginp", "random"])
    # additional FixMatch arguments
    parser.add_argument('--threshold', type=float, default=0)
    # additional arguments
    parser.add_argument('--method_type', type=str, default='target')
    parser.add_argument('--classifier_type', type=str, default='softmax')
    parser.add_argument('--final_savename', type=str, default=None)
    # detailed prediction per sample
    parser.add_argument('--return_output', type=bool, default=False)

    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['art', 'clipart', 'product', 'realworld']
        args.class_num = 65 
    elif args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    elif args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
        args.lr = 1e-3
    elif args.dset == 'domainnet-126':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    folder = '../data/'
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.method_type == 'source':
            args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s], str(SEED))  # directory to load source model
            args.final_output_dir = osp.join(args.output, 'initialization', args.da, args.dset, names[args.s]+'_'+names[args.t], str(SEED))
        elif args.method_type == 'pretrained':
            args.final_output_dir = osp.join(args.output, 'initialization', args.da, args.dset, 'pretrained_'+names[args.t], str(SEED))
        elif args.method_type == 'target':
            args.final_output_dir = osp.join(args.output, args.da, args.dset, names[args.s]+'_'+names[args.t], str(SEED))

        if not osp.exists(args.final_output_dir):
            os.system('mkdir -p ' + args.final_output_dir)

        # target methods
        if args.method_name == 'shot':
            args.savename = 'par_' + str(args.cls_par)
            if args.ssl > 0:
                args.savename += ('_ssl_' + str(args.ssl))
            args.final_savename = args.savename
        elif args.method_name == 'shotplus':
            args.savename = 'par_' + str(args.cls_par)
            if args.ssl > 0:
                args.savename += ('_ssl_' + str(args.ssl))
            args.log = 'ps_' + str(args.ps) + '_' + args.savename
            args.final_savename = "{:}_{:}".format(args.log, args.choice)
        elif 'fixmatch' in args.method_name:
            args.savename = 'srconly'
            args.log = 'ps_' + str(args.ps) + '_' + args.savename
            args.final_savename = "{:}_{:}{}".format(args.log, args.choice, args.threshold)
        elif args.method_name == 'proposed':
            args.final_savename += '_eval'
        elif args.method_name == 'source':
            args.final_savename = 'source'
        else:
            args.final_savename = args.classifier_type

        # save results to txt file
        args.out_file = open(osp.join(args.final_output_dir, f'log_{args.final_savename}.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()

        if args.return_output:  
            # save detailed prediction for each sample
            with torch.no_grad():
                all_labels, all_preds = test_target(args)       
            args.results_txt_path = os.path.join(args.final_output_dir, names[args.t] + '_list.txt')                        
            txt_test = open(args.test_dset_path).readlines()
            nextline_char = '\n'
            txt_test_w_preds = [f"{x.replace(nextline_char,'')} lab:{all_labels[i]} pred:{all_preds[i]}" for i, x in enumerate(txt_test)]
            with open(args.results_txt_path, 'w') as file:
                file.write('\n'.join(txt_test_w_preds))
        else:
            # save results to json file            
            with torch.no_grad():
                results_dict = test_target(args)            
            args.results_json_path = os.path.join(args.final_output_dir, f'results_{args.final_savename}.jsonl')
            with open(args.results_json_path, 'w') as file:
                file.write(json.dumps(results_dict, sort_keys=True) + "\n")


        args.out_file.close() 