import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import loss
from sklearn.metrics import confusion_matrix

# helper functions

def get_logit(feas, classifier, projection=None, classifier_type='linear', temperature=1.):
    if classifier_type in ['linear', 'prototype']:
        logits = classifier(feas)
    elif classifier_type in ['NCC']:
        feas = torch.cat((feas, torch.ones(feas.size(0), 1, device=feas.device)), 1)
        feas = (feas.t() / torch.norm(feas, p=2, dim=1)).t()
        logits = F.cosine_similarity(feas.unsqueeze(2), classifier, dim=1)
    elif classifier_type in ['NCC_mixed']:
        netC = classifier['netC']
        prototype_netC = classifier['prototype_netC']
        logits_text = prototype_netC(feas) * 0.05  # temperature applied for text classifier
        feas = torch.cat((feas, torch.ones(feas.size(0), 1, device=feas.device)), 1)
        feas = (feas.t() / torch.norm(feas, p=2, dim=1)).t()
        logits_image = F.cosine_similarity(feas.unsqueeze(2), netC, dim=1)

        temperature_text = torch.std(logits_text, dim=1, keepdim=True) / torch.std(logits_image, dim=1, keepdim=True)
        logits = (logits_image + logits_text / temperature_text) / 2
    if projection is not None:
        logits = projection(logits)
    return logits / temperature

def get_prob(logits, classifier_type):
    if classifier_type in ['linear', 'NCC', 'prototype', 'NCC_mixed']:
        probs = nn.Softmax(dim=1)(logits)
    return probs

# calculate metric values
@torch.no_grad()
def cal_acc(loader, netF, netB, netC, netP=None, classifier_type='linear', temperature=1.,
    return_output=False, model_type=None):
    '''
    Return accuracy, mean accuracy across classes and per-class-accuracy.
    '''
    loader_len = len(loader)

    start_test = True
    iter_test = iter(loader)
    load_img = 0 if model_type == 'adapt' else 2
    for i in range(loader_len):
        data = iter_test.next()
        inputs = data[load_img].cuda()
        labels = data[1]
        if netB is None:
            logits = get_logit(netF(inputs), netC, netP, classifier_type, temperature)
        else:
            logits = get_logit(netB(netF(inputs)), netC, netP, classifier_type, temperature)
        probs = get_prob(logits, classifier_type)
        if start_test:
            all_probs = probs.float().cpu()
            all_labels = labels.float()
            start_test = False
        else:
            all_probs = torch.cat((all_probs, probs.float().cpu()), 0)
            all_labels = torch.cat((all_labels, labels.float()), 0)

    _, all_preds = torch.max(all_probs, 1)

    # accuracy
    accuracy = 100 * torch.sum(torch.squeeze(all_preds).float() == all_labels).item() / float(all_labels.size()[0])
    # per-class accuracy
    matrix = confusion_matrix(all_labels, torch.squeeze(all_preds).float())
    accuracy_per_class = 100 * matrix.diagonal() / matrix.sum(axis=1)
    accuracy_mean = accuracy_per_class.mean()

    if return_output:
        return accuracy, accuracy_mean, accuracy_per_class, all_labels, all_preds
    else:
        return accuracy, accuracy_mean, accuracy_per_class