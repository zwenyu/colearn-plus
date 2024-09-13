import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import loss
from sklearn.metrics import confusion_matrix

# helper functions

def get_logit(feas, classifier, classifier_type):
    if classifier_type in ['softmax']:
        logits = classifier(feas)
    elif classifier_type == 'NCC':
        feas = torch.cat((feas, torch.ones(feas.size(0), 1, device=feas.device)), 1)
        logits = F.cosine_similarity(feas.unsqueeze(2), classifier, dim=1)
    return logits

def get_prob(logits, classifier_type):
    if classifier_type in ['softmax', 'NCC']:
        probs = nn.Softmax(dim=1)(logits)
    return probs

# calculate metric values
@torch.no_grad()
def cal_acc(loader, netF, netB, netC, classifier_type):
    '''
    Return accuracy, mean accuracy across classes and per-class-accuracy.
    '''
    start_test = True
    iter_test = iter(loader)
    for i in range(len(loader)):
        data = iter_test.next()
        inputs = data[0].cuda()
        labels = data[1]
        if netB is None:
            logits = get_logit(netF(inputs), netC, classifier_type)
        else:
            logits = get_logit(netB(netF(inputs)), netC, classifier_type)
        if start_test:
            all_logits = logits.float().cpu()
            all_labels = labels.float()
            start_test = False
        else:
            all_logits = torch.cat((all_logits, logits.float().cpu()), 0)
            all_labels = torch.cat((all_labels, labels.float()), 0)

    all_probs = get_prob(all_logits, classifier_type)
    _, all_preds = torch.max(all_probs, 1)

    # accuracy
    accuracy = 100 * torch.sum(torch.squeeze(all_preds).float() == all_labels).item() / float(all_labels.size()[0])
    # per-class accuracy
    matrix = confusion_matrix(all_labels, torch.squeeze(all_preds).float())
    matrix = matrix[np.unique(all_labels).astype(int), :]
    accuracy_per_class = 100 * matrix.diagonal() / matrix.sum(axis=1)
    accuracy_mean = accuracy_per_class.mean()

    return accuracy, accuracy_mean, accuracy_per_class