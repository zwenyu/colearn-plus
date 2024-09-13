'''
Train on target domain by co-learning with models on two branches.

Initialization:
- branch 1, adaptation model model branch: source-initialized FE and classifier
- branch 2 pre-trained model branch: pretrained FE and NCC
- Training data loader that returns
    - image path, pseudolabel to train with, pseudolablling indicator
    - sample is not pseudolabeled if both branches have same confidence but different predictions
    - otherwise, sample is pseudolabeled with the matching or confident prediction

Iterative training and update every epoch
- 1. train FE of branch 1 with supervised loss
- 2. evaluate branch 1 and update training data loader
- 3. update branch 2 NCC with training data loader from step 2
- 4. train projection layer of branch 2 with supervised loss (if applicable)
- 5. evaluate branch 2 and update training data loader
'''

import argparse
import os, sys
import os.path as osp
import random
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import network
from data_list import ImageList, ImageList_w_ind
from other_utils import method_utils, eval_utils


@torch.no_grad()
def get_pred_conf(args, loader, netF, netB, netC, netP, model_type):
    '''
    Return predicted label and confidence indicator.

    :param args: all arguments of current run
    :param loader: data loader
    :param netF: feature extractor
    :param netB: bottleneck
    :param netC: classifier
    :param netP: projection
    :param model_type: pretrained or adapt branch
    :return:
        - prediction of samples
        - confidence indicator of samples
    '''
    start_target = True
    iter_target = iter(loader)
    load_img = 0 if model_type == 'adapt' else 2
    for _ in range(len(loader)):
        data = iter_target.next()
        inputs = data[load_img].cuda()
        labels = data[1].cuda()
        if model_type == 'adapt':
            logits = netC(netB(netF(inputs)))
        elif model_type == 'pretrained':
            feas = netF(inputs)
            if netB is not None:
                feas = netB(feas)
            feas = torch.cat((feas, torch.ones(feas.size(0), 1, device=feas.device)), 1)  # add feature column of 1's to prevent small norm
            feas = (feas.t() / torch.norm(feas, p=2, dim=1)).t()  # normalize features for each sample
            logits = F.cosine_similarity(feas.unsqueeze(2), netC, dim=1)
            if netP is not None:
                logits = netP(logits)
        else:
            raise NotImplementedError
        probs = F.softmax(logits, dim=1)
        confs, preds = torch.max(probs, 1)
        if start_target:
            all_confs = confs
            all_preds = preds
            start_target = False
        else:
            all_confs = torch.cat((all_confs, confs), 0)
            all_preds = torch.cat((all_preds, preds), 0)

    all_ind_confs = (all_confs > args.conf_threshold)

    return all_preds, all_ind_confs


def data_load(args):
    '''
    Return data loader for entire target dataset under test augmentation.

    :param args: all arguments of current run
    :return: dictionary of data loaders
    '''
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    target_txt = open(args.test_dset_path).readlines()

    dsets['target_all'] = ImageList(target_txt, transform_adapt=method_utils.image_test(net=args.source_net),
        transform_pretrained=method_utils.image_test(use_clip=(args.pretrained_net[0:4] == 'clip'), net=args.pretrained_net))
    dset_loaders['target_all'] = DataLoader(dsets['target_all'], batch_size=train_bs, shuffle=False,
        num_workers=args.worker, drop_last=False)
    return dset_loaders


def data_load_pred(args, dset_loaders, adapt_netF, adapt_netB, adapt_netC,
    pretrained_netF, pretrained_netB, pretrained_netC, pretrained_netP, branch):
    '''
    Prepare target dataset for training.

    :param args: all arguments of current run
    :param dset_loaders: dictionary of data loaders
    :param adapt_netF: adaptation model feature extractor
    :param adapt_netB: adaptation model bottleneck
    :param adapt_netC: adaptation model classifier
    :param pretrained_netF: pretrained model feature extractor
    :param pretrained_netB: pretrained model projection layer
    :param pretrained_netC: pretrained model classifier
    :param pretrained_netP: pretrained model projection layer
    :param branch: adapt or pretrained model branch
    :return: 
        - dictionary of data loaders
        - confidence indicator of samples by adaptation model
        - confidence indicator of samples by pretrained model
        - pseudolabeling indicator of samples
    '''
    dsets = {}
    train_bs = args.batch_size
    target_txt = open(args.test_dset_path).readlines()

    # prediction and confidence by adaptation model
    adapt_preds, adapt_ind_confs = get_pred_conf(args, dset_loaders['target_all'],
        adapt_netF, adapt_netB, adapt_netC, None, 'adapt')
    # prediction and confidence by pretrained model
    pretrained_preds, pretrained_ind_confs = get_pred_conf(args, dset_loaders['target_all'],
        pretrained_netF, pretrained_netB, pretrained_netC, pretrained_netP, 'pretrained')
    # consolidate to obtain pseudolabels for training
    preds = copy.deepcopy(adapt_preds)
    ind_pretrained_more_conf = pretrained_ind_confs & (~adapt_ind_confs)
    preds[ind_pretrained_more_conf] = pretrained_preds[ind_pretrained_more_conf]
    if args.pseudolabel_strategy == 'MatchOrConf':
        ind_no_pseudolabel = (adapt_ind_confs == pretrained_ind_confs) & (adapt_preds != pretrained_preds)
        ind_pseudolabel = (~ind_no_pseudolabel)
    elif args.pseudolabel_strategy == 'MatchAndConf':
        ind_pseudolabel = (adapt_preds == pretrained_preds) & adapt_ind_confs & pretrained_ind_confs
    elif args.pseudolabel_strategy == 'Match':
        ind_pseudolabel = (adapt_preds == pretrained_preds)
    elif args.pseudolabel_strategy == 'OtherConf':
        if branch == 'adapt':
            preds = pretrained_preds
            ind_pseudolabel = pretrained_ind_confs
        elif branch == 'pretrained':
            preds = adapt_preds
            ind_pseudolabel = adapt_ind_confs
    elif args.pseudolabel_strategy == 'SelfConf':
        if branch == 'adapt':
            preds = adapt_preds
            ind_pseudolabel = adapt_ind_confs
        elif branch == 'pretrained':
            preds = pretrained_preds
            ind_pseudolabel = pretrained_ind_confs

    # data loaders with predicted labels
    tr_target_pred_txt = [' '.join([v.split(' ')[0], str(int(preds[k])), str(int(ind_pseudolabel[k]))]) for k, v in enumerate(target_txt)]
    dsets['target_pred_train'] = ImageList_w_ind(tr_target_pred_txt, transform_adapt=method_utils.image_train(apply_randaugment=True, net=args.source_net),
        transform_pretrained=method_utils.image_train(use_clip=(args.pretrained_net[0:4] == 'clip'), net=args.pretrained_net))
    dset_loaders['target_pred_train'] = DataLoader(dsets['target_pred_train'], batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=True)
    return dset_loaders, adapt_ind_confs, pretrained_ind_confs, ind_pseudolabel


def finetune(args, dset_loaders, adapt_netF, adapt_netB, adapt_netC, pretrained_netF):
    """
    Finetune models.

    :param args: all arguments of current run
    :param dset_loaders: dictionary of data loaders
    :param adapt_netF: adaptation model feature extractor
    :param adapt_netB: adaptation model bottleneck
    :param adapt_netC: adaptation model classifier
    :param pretrained_netF: pretrained model feature extractor
    :return:
        - finetuned adapt_netF
        - finetuned classifier for pretrained model
    """
    # initialize parameters
    for param in adapt_netB.parameters():
        param.requires_grad = False
    for param in adapt_netC.parameters():
        param.requires_grad = False

    pretrained_feature_dim = pretrained_netF.model_clip.visual.output_dim
    pretrained_prototype_netC = network.ProtoCLS(in_dim=pretrained_feature_dim, out_dim=args.class_num)
    classnames = dset_loaders['target_all'].dataset.classnames
    templates = method_utils.get_templates(args.text_templates, args.dset, names[args.t])
    pretrained_prototype_netC.fc.weight.data, text_features, text_labels = method_utils.text_information(pretrained_netF.model_clip, classnames, templates, 'cuda')

    centroids = method_utils.get_centroids(args, dset_loaders['target_all'], adapt_netF, adapt_netB, adapt_netC, pretrained_netF, None, pretrained_prototype_netC,
        use_source_pseudolabel=True, model_type='adapt')
    pretrained_netC = centroids.t()
    pretrained_netP = nn.Linear(args.class_num, args.class_num, bias=False)
    pretrained_netP.weight.data = torch.eye(args.class_num, device=centroids.device) / args.pretrained_temperature
    pretrained_netP.eval()
    pretrained_ft_param = list()
    if 'fe' in args.pretrained_ft_comp:
        pretrained_ft_param += list(pretrained_netF.parameters())
    else:
        for param in pretrained_netF.parameters():
            param.requires_grad = False
    if 'clfproj' in args.pretrained_ft_comp:
        pretrained_ft_param += list(pretrained_netP.parameters())
    else:
        pretrained_netP.weight.requires_grad = False

    # track accuracy
    adapt_acc_init, adapt_accmean_init, _ = eval_utils.cal_acc(dset_loaders['target_all'], adapt_netF, adapt_netB, adapt_netC, None, 'linear', model_type='adapt')
    pretrained_acc_init, pretrained_accmean_init, _ = eval_utils.cal_acc(dset_loaders['target_all'], 
        pretrained_netF, None, pretrained_netC, pretrained_netP, 'NCC', model_type='pretrained')
    results_init = {f'adapt_acc_init_{names[args.t]}': adapt_acc_init, f'adapt_acc_classmean_init_{names[args.t]}': adapt_accmean_init,
        f'pretrained_acc_init_{names[args.t]}': pretrained_acc_init, f'pretrained_acc_classmean_init_{names[args.t]}': pretrained_accmean_init}

    # log
    log_str = 'Epoch:{}/{}; ' \
        'adapt_acc = {:.6f}, adapt_accmean = {:.6f}, pretrained_acc = {:.6f}, pretrained_accmean = {:.6f}'.format(
        -1, args.max_epoch,
        adapt_acc_init, adapt_accmean_init, pretrained_acc_init, pretrained_accmean_init)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    log_dict = {'epoch': -1,
        'adapt_acc': adapt_acc_init, 'adapt_accmean': adapt_accmean_init,
        'pretrained_acc': pretrained_acc_init, 'pretrained_accmean': pretrained_accmean_init}
    args.out_json.write(json.dumps(log_dict) + '\n')

    # optimizers
    if args.optimizer == 'sgd':
        adapt_optimizer = optim.SGD(adapt_netF.parameters(), lr=args.adapt_lr)
    elif args.optimizer == 'adam':
        adapt_optimizer = optim.Adam(adapt_netF.parameters(), lr=args.adapt_lr)
    adapt_scheduler = optim.lr_scheduler.StepLR(adapt_optimizer, step_size=args.scheduler_step_size, gamma=0.1)
    if len(pretrained_ft_param) > 0:
        if args.optimizer == 'sgd':
            pretrained_optimizer = optim.SGD(pretrained_ft_param, lr=args.pretrained_lr)
        elif args.optimizer == 'adam':
            pretrained_optimizer = optim.Adam(pretrained_ft_param, lr=args.pretrained_lr)
        pretrained_scheduler = optim.lr_scheduler.StepLR(pretrained_optimizer, step_size=args.scheduler_step_size, gamma=0.1)

    # finetuning
    starttime = time.time()
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    for epoch in range(args.max_epoch):
        # update training data loader with predictions
        dset_loaders, adapt_ind_confs, pretrained_ind_confs, ind_pseudolabel = data_load_pred(args, dset_loaders, 
            adapt_netF, adapt_netB, adapt_netC, pretrained_netF, None, pretrained_netC, pretrained_netP, 'adapt')
        max_iter = len(dset_loaders['target_pred_train'])
        # update adaptation model branch
        adapt_netF.train()
        adapt_ce = []
        adapt_ent = []
        adapt_loss = []
        iter_num = 0
        for data in dset_loaders['target_pred_train']:
            iter_num += 1
            inputs = data[0].cuda()
            labels = data[1].cuda()
            ind_labels = data[2].cuda()
            ind_nolabels = ~ind_labels
            logits = adapt_netC(adapt_netB(adapt_netF(inputs)))

            ce_idv = ce_loss(logits, labels)
            ce = sum(ce_idv * ind_labels) / (sum(ind_labels) + args.epsilon)
            ent_idv = ce_loss(logits, F.softmax(logits, dim=1))
            ent = sum(ent_idv * ind_nolabels) / (sum(ind_nolabels) + args.epsilon)
            total_loss = ce + args.lambda_entropy * ent

            # log
            log_str = 'Adaptation model, Epoch:{}/{}, Iter:{}/{}; ce = {:.6f}, ent = {:.6f}, loss = {:.6f}'.format(
                epoch, args.max_epoch, iter_num, max_iter, ce, ent, total_loss)
            print(log_str + '\n')
            args.tb_writer.add_scalar('CE/adaptation', ce, epoch * max_iter + iter_num)
            args.tb_writer.add_scalar('Entropy/adaptation', ent, epoch * max_iter + iter_num)
            args.tb_writer.add_scalar('Loss/adaptation', total_loss, epoch * max_iter + iter_num)

            adapt_ce.append(ce.item())
            adapt_ent.append(ent.item())
            adapt_loss.append(total_loss.item())

            adapt_optimizer.zero_grad()
            total_loss.backward()
            adapt_optimizer.step()

        adapt_netF.eval()
        adapt_ce = np.mean(adapt_ce)
        adapt_ent = np.mean(adapt_ent)
        adapt_loss = np.mean(adapt_loss)
        adapt_scheduler.step()        

        # update training data loader with predictions
        dset_loaders, _, _, _ = data_load_pred(args, dset_loaders, adapt_netF, adapt_netB, adapt_netC, 
            pretrained_netF, None, pretrained_netC, pretrained_netP, 'pretrained')
        max_iter = len(dset_loaders['target_pred_train'])
        if 'clfncc' in args.pretrained_ft_comp:
            # update pretrained model branch
            centroids = method_utils.get_centroids(args, dset_loaders['target_all'], adapt_netF, adapt_netB, adapt_netC, pretrained_netF, None, pretrained_prototype_netC,
                use_source_pseudolabel=True, model_type='adapt')
            pretrained_netC = centroids.t()
        if 'fe' in args.pretrained_ft_comp:
            pretrained_netF.train()
        if 'clfproj':
            pretrained_netP.train()
        if len(pretrained_ft_param) > 0:
            pretrained_ce = []
            pretrained_ent = []
            pretrained_con = []
            pretrained_loss = []
            iter_num = 0
            for data in dset_loaders['target_pred_train']:
                iter_num += 1
                inputs = data[3].cuda()
                labels = data[1].cuda()
                ind_labels = data[2].cuda()
                ind_nolabels = ~ind_labels
                feas = pretrained_netF(inputs)
                feas = torch.cat((feas, torch.ones(feas.size(0), 1, device=feas.device)), 1)  # add feature column of 1's to prevent small norm
                feas = (feas.t() / torch.norm(feas, p=2, dim=1)).t()  # normalize features for each sample
                logits = F.cosine_similarity(feas.unsqueeze(2), pretrained_netC, dim=1)
                logits = pretrained_netP(logits)

                ce_idv = ce_loss(logits, labels)
                ce = sum(ce_idv * ind_labels) / (sum(ind_labels) + args.epsilon)
                ent_idv = ce_loss(logits, F.softmax(logits, dim=1))
                ent = sum(ent_idv * ind_nolabels) / (sum(ind_nolabels) + args.epsilon)
                total_loss = ce + args.lambda_entropy * ent

                # log
                log_str = 'Pretrained model, Epoch: {}/{}, Iter:{}/{}; ce = {:.6f}, ent = {:.6f}, loss = {:.6f}'.format(
                    epoch, args.max_epoch, iter_num, max_iter, ce, ent, total_loss)
                print(log_str + '\n')
                args.tb_writer.add_scalar('CE/pretrained', ce, epoch * max_iter + iter_num)
                args.tb_writer.add_scalar('Entropy/pretrained', ent, epoch * max_iter + iter_num)
                args.tb_writer.add_scalar('Loss/pretrained', total_loss, epoch * max_iter + iter_num)

                pretrained_ce.append(ce.item())
                pretrained_ent.append(ent.item())
                pretrained_loss.append(total_loss.item())

                pretrained_optimizer.zero_grad()
                total_loss.backward()
                pretrained_optimizer.step()

            pretrained_netF.eval()
            pretrained_netP.eval()
            pretrained_ce = np.mean(pretrained_ce)
            pretrained_ent = np.mean(pretrained_ent)
            pretrained_loss = np.mean(pretrained_loss)
            pretrained_scheduler.step()
        else:
            pretrained_ce = pretrained_ent = pretrained_loss = 0.

        if not args.istime:
            # track accuracy
            adapt_acc, adapt_accmean, _ = eval_utils.cal_acc(dset_loaders['target_all'], adapt_netF, adapt_netB, adapt_netC, None, 'linear', model_type='adapt')
            pretrained_acc, pretrained_accmean, _ = eval_utils.cal_acc(dset_loaders['target_all'],
                pretrained_netF, None, pretrained_netC, pretrained_netP, 'NCC', model_type='pretrained')

            # log
            log_str = 'Epoch:{}/{}; ' \
                'adapt_ce = {:.6f}, adapt_ent = {:.6f}, adapt_loss = {:.6f}, ' \
                'pretrained_ce = {:.6f}, pretrained_ent = {:.6f}, pretrained_loss = {:.6f}, ' \
                'adapt_acc = {:.6f}, adapt_accmean = {:.6f}, pretrained_acc = {:.6f}, pretrained_accmean = {:.6f}'.format(
                epoch, args.max_epoch,
                adapt_ce, adapt_ent, adapt_loss, pretrained_ce, pretrained_ent, pretrained_loss,
                adapt_acc, adapt_accmean, pretrained_acc, pretrained_accmean)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            log_dict = {'epoch': epoch, 'ind_pseudolabel': ind_pseudolabel.float().mean().item(),
                'adapt_ind_confs': adapt_ind_confs.float().mean().item(), 'pretrained_ind_confs': pretrained_ind_confs.float().mean().item(),
                'adapt_ce': adapt_ce, 'adapt_ent': adapt_ent, 'adapt_loss': adapt_loss,
                'adapt_ce': pretrained_ce, 'adapt_ent': pretrained_ent, 'pretrained_loss': pretrained_loss,
                'adapt_acc': adapt_acc, 'adapt_accmean': adapt_accmean,
                'pretrained_acc': pretrained_acc, 'pretrained_accmean': pretrained_accmean}
            args.out_json.write(json.dumps(log_dict) + '\n')

    endtime = time.time()
    return adapt_netF, results_init, (endtime-starttime)


def train_target(args):
    '''
    Training

    :param args: all arguments of current run
    :return: dictionary of results    
    '''
    # load data
    dset_loaders = data_load(args)

    # load pretrained model
    if args.pretrained_net[0:3] == 'res':
        pretrained_netF = network.ResBase(res_name=args.pretrained_net).cuda()
    elif args.pretrained_net[0:3] == 'vgg':
        pretrained_netF = network.VGGBase(vgg_name=args.pretrained_net).cuda()
    elif args.pretrained_net[0:4] == 'alex':
        pretrained_netF = network.AlexBase().cuda()
    elif args.pretrained_net[0:8] == 'convnext':
        pretrained_netF = network.ConvnextBase(convnext_name=args.pretrained_net).cuda()
    elif args.pretrained_net[0:4] == 'swin':
        pretrained_netF = network.SwinBase(swin_name=args.pretrained_net).cuda()
    elif args.pretrained_net[0:4] == 'clip':
        pretrained_netF = network.CLIPBase(clip_name=args.pretrained_net).cuda()

    if args.pretrained_model_type == 'source':
        pretrained_modelpathF = args.source_output_dir + '/source_F.pt'

    if args.pretrained_model_type != 'pretrained':
        pretrained_netF.load_state_dict(torch.load(pretrained_modelpathF))
    pretrained_netF.eval()

    # load adaptation model
    if args.source_net[0:3] == 'res':
        adapt_netF = network.ResBase(res_name=args.source_net).cuda()
    elif args.source_net[0:3] == 'vgg':
        adapt_netF = network.VGGBase(vgg_name=args.source_net).cuda()
    adapt_netB = network.feat_bottleneck(type=args.source_classifier, feature_dim=adapt_netF.in_features, bottleneck_dim=args.source_bottleneck).cuda()
    adapt_netC = network.feat_classifier(type=args.source_layer, class_num=args.class_num, bottleneck_dim=args.source_bottleneck).cuda()

    try:
        source_modelpathF = args.source_output_dir + '/source_F.pt'
        adapt_netF.load_state_dict(torch.load(source_modelpathF))
        source_modelpathB = args.source_output_dir + '/source_B.pt'
        adapt_netB.load_state_dict(torch.load(source_modelpathB))
        source_modelpathC = args.source_output_dir + '/source_C.pt'
        adapt_netC.load_state_dict(torch.load(source_modelpathC))
    except:
        source_modelpathF = args.source_output_dir + '/target_adapt_F.pt'
        adapt_netF.load_state_dict(torch.load(source_modelpathF))
        source_modelpathB = args.source_output_dir + '/target_adapt_B.pt'
        adapt_netB.load_state_dict(torch.load(source_modelpathB))
        source_modelpathC = args.source_output_dir + '/target_adapt_C.pt'
        adapt_netC.load_state_dict(torch.load(source_modelpathC))        
    adapt_netF.eval()
    adapt_netB.eval()
    adapt_netC.eval()

    # finetuning models
    adapt_netF, results_init, adapt_time = finetune(args, dset_loaders, adapt_netF, adapt_netB, adapt_netC, pretrained_netF)

    if args.issave:
        torch.save(adapt_netF.state_dict(), osp.join(args.final_output_dir, 'target_adapt_F.pt'))
        torch.save(adapt_netB.state_dict(), osp.join(args.final_output_dir, 'target_adapt_B.pt'))
        torch.save(adapt_netC.state_dict(), osp.join(args.final_output_dir, 'target_adapt_C.pt'))
    
    # save results
    acc, accmean, acc_list = eval_utils.cal_acc(dset_loaders['target_all'], adapt_netF, adapt_netB, adapt_netC, None, 'linear', model_type='adapt')
    acc_per_class = {f'acc_class_{k}_{names[args.t]}': v for k, v in enumerate(acc_list)}
    acc_consolidate = {f'acc_{names[args.t]}': acc, f'acc_classmean_{names[args.t]}': accmean}
    results_dict = {**results_init, **acc_consolidate, **acc_per_class, 'time': adapt_time}

    return results_dict


def print_args(args):
    s = '==========================================\n'
    for arg, content in args.__dict__.items():
        s += '{}:{}\n'.format(arg, content)
    return s

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Target training')
    parser.add_argument('--da', type=str, default='uda')
    parser.add_argument('--expt_name', type=str, default='target_training_two_branch', help='experiment name')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help='device id to run')
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'domainnet-126', 'cub'])
    parser.add_argument('--s', type=int, default=0, help='source')
    parser.add_argument('--t', type=int, default=None, help='target')

    # pretrained model arguments
    parser.add_argument('--pretrained_net', type=str, default='resnet50', help='resnet50, resnet101')
    parser.add_argument('--pretrained_model_type', type=str, default='pretrained',
        choices=['source', 'pretrained'],
        help='type of model to evaluate')
    parser.add_argument('--pretrained_temperature', type=float, default=0.01)

    # source model arguments
    parser.add_argument('--source_output', type=str, default='results/source/resnet/vanilla')
    parser.add_argument('--source_net', type=str, default='resnet50', help='resnet50, renet101')
    parser.add_argument('--source_bottleneck', type=int, default=256)
    parser.add_argument('--source_layer', type=str, default='wn', choices=['linear', 'wn'])
    parser.add_argument('--source_classifier', type=str, default='bn', choices=['ori', 'bn'])

    # training arguments
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='confidence threshold for training data selection')
    parser.add_argument('--pseudolabel_strategy', type=str, default='MatchOrConf', help='strategy to select pseudolabels for supervision')
    parser.add_argument('--lambda_entropy', type=float, default=0., help='hyperparameter for entropy')
    parser.add_argument('--pretrained_ft_comp', type=str, default=[], nargs='+', help='pretrained model components to finetune in training')
    parser.add_argument('--max_epoch', type=int, default=15, help="max number of epochs for each branch")
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--batch_size', type=int, default=50, help='batch_size')
    parser.add_argument('--worker', type=int, default=4, help='number of workers')
    parser.add_argument('--adapt_lr', type=float, default=1e-2, help='learning rate for adaptation model')
    parser.add_argument('--pretrained_lr', type=float, default=1e-2, help='learning rate for pretrained model')
    parser.add_argument('--scheduler_step_size', type=int, default=10)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--issave', action='store_true', default=False)
    parser.add_argument('--istime', type=bool, default=False)
    # additional arguments
    parser.add_argument('--text_templates', type=str, default='ensemble',
        choices=['classname', 'vanilla', 'handcrafted', 'ensemble', 'template_mining'],
        help='text template for CLIP text classifier')

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
    elif args.dset == 'domainnet-126':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126
    elif args.dset == 'cub':
        names = ['CUB200Painting', 'CUB2002011']
        args.class_num = 200        

    if args.s >= len(names):
        sys.exit('args.s is not in available domains')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
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

        # directory to load model
        if args.pretrained_model_type == 'source':
            args.source_output_dir = osp.join(args.source_output, args.da, args.dset, names[args.s], str(SEED))  # directory to load source model
            args.final_output_dir = osp.join(args.expt_name, args.output, args.da, args.dset, names[args.s] + '_' + names[args.t], str(SEED))
        elif args.pretrained_model_type == 'pretrained':
            args.final_output_dir = osp.join(args.expt_name, args.output, args.da, args.dset, names[args.s] + '_' + names[args.t], str(SEED))

        # directory to load source model
        args.source_output_dir = osp.join(args.source_output, args.da, args.dset, names[args.s], str(SEED))

        if not osp.exists(args.final_output_dir):
            os.system('mkdir -p ' + args.final_output_dir)
        if not osp.exists(args.final_output_dir):
            os.mkdir(args.final_output_dir)

        # file name
        pretrained_ft_comp = '-'.join(args.pretrained_ft_comp)
        args.final_savename = f'{args.pseudolabel_strategy}_{pretrained_ft_comp}_threshold{args.conf_threshold}_ent{args.lambda_entropy}' \
            f'_opt{args.optimizer}_adaptlr{args.adapt_lr}_pretrainedlr{args.pretrained_lr}_bs{args.batch_size}'

        # check if run already completed
        dir_list = os.listdir(args.final_output_dir)
        if (f'results_{args.final_savename}.jsonl') in dir_list:
            print('Experiment already completed.')
        else:
            # save logs to txt and json file
            args.out_file = open(osp.join(args.final_output_dir, f'log_{args.final_savename}.txt'), 'w')
            args.out_file.write(print_args(args) + '\n')
            args.out_file.flush()
            args.out_json = open(os.path.join(args.final_output_dir, f'log_{args.final_savename}.jsonl'), 'w')
            # save training process to tensorboard
            args.tb_writer = SummaryWriter(log_dir=os.path.join(args.final_output_dir, args.final_savename))

            results_dict = train_target(args)
            args.out_file.close()
            args.out_json.close()
            args.tb_writer.close()

            # save results to json file
            args.results_json = os.path.join(args.final_output_dir, f'results_{args.final_savename}.jsonl')
            with open(args.results_json, 'w') as file:
                file.write(json.dumps(results_dict, sort_keys=True) + '\n')