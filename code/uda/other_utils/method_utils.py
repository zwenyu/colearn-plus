import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import clip
import templates


def image_totensor(resize_size=256):
    '''
    Transform image to tensor.

    :param resize_size: resize size
    :return: image augmentation
    '''    
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor()
    ])


def image_train(apply_randaugment=False, use_clip=False, net=None):
    '''
    Augmentations on training samples.

    :param use_clip: use CLIP config if true
    :param apply_randaugment: include RandAugment if true, exclude if false
    :return: image augmentation
    '''
    if use_clip:
        norm_mean = [0.48145466, 0.4578275, 0.40821073]
        norm_std = [0.26862954, 0.26130258, 0.27577711]
    else:
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
    if net == "clip_vit":
        resize_size = 336
        crop_size = 336
    elif "clip" in net:
        resize_size = 224
        crop_size = 224
    else:
        resize_size = 256
        crop_size = 224
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    if apply_randaugment:
        return  transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return  transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])


def image_test(use_clip=False, net=None):
    '''
    Augmentations on testing samples.

    :param use_clip: use CLIP config if true
    :return: image augmentation
    '''    
    if use_clip:
        norm_mean = [0.48145466, 0.4578275, 0.40821073]
        norm_std = [0.26862954, 0.26130258, 0.27577711]
    else:
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
    if net == "clip_vit":
        resize_size = 336
        crop_size = 336
    elif "clip" in net:
        resize_size = 224
        crop_size = 224
    else:
        resize_size = 256
        crop_size = 224
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


@torch.no_grad()
def get_centroids(args, loader, source_netF, source_netB, source_netC, netF, netB=None, netC=None,
    use_source_pseudolabel=True, source_temperature=1., temperature=1., class_num=None, model_type=None):
    '''
    Compute centroids according to soft target pseudolabels from pseudolabel model.

    :param args: all arguments of current run
    :param loader: data loader
    :param source_netF: source feature extractor
    :param source_netB: source bottleneck
    :param source_netC: source classifier
    :param netF: feature extractor
    :param netB: projection layer
    :param use_source_pseudolabel: use source pseudolabel if True, use true label if False
    :param class_num: number of classes
    :param model_type: type of model for source pseudo labels
    :return: centroids of netF features, with size = # classes x # features
    '''
    class_num = args.class_num if class_num is None else class_num
    start_target = True
    iter_target = iter(loader)
    load_img = 0 if model_type == 'adapt' else 2
    counter = 0
    for _ in range(len(loader)):
        data = iter_target.next()
        inputs = data[load_img].cuda()
        labels = data[1].cuda()
        pretrained_inputs = data[-1].cuda()
        pretrained_feas = netF(pretrained_inputs)
        if netB is not None:
            pretrained_feas = netB(pretrained_feas)
        pretrained_feas = torch.cat((pretrained_feas, torch.ones(pretrained_feas.size(0), 1, device=pretrained_feas.device)), 1)  # add feature column of 1's to prevent small norm
        pretrained_feas = (pretrained_feas.t() / torch.norm(pretrained_feas, p=2, dim=1)).t()  # normalize features for each sample
        if use_source_pseudolabel:
            source_logits = source_netC(source_netB(source_netF(inputs))) if source_netB is not None else source_netC(source_netF(inputs))
            if source_temperature == 0:
                source_probs = F.one_hot(source_logits.max(dim=1).indices, num_classes=class_num)
            else:
                source_probs = nn.Softmax(dim=1)(source_logits / source_temperature)
            if netC is not None:
                pretrained_logits = netC(netB(netF(pretrained_inputs))) if netB is not None else netC(netF(pretrained_inputs))
                if temperature is None:
                    pretrained_temperature = torch.std(pretrained_logits, dim=1, keepdim=True) / torch.std(source_logits, dim=1, keepdim=True)
                    pretrained_probs = nn.Softmax(dim=1)(pretrained_logits / pretrained_temperature)
                elif temperature == 0:
                    pretrained_probs = F.one_hot(pretrained_logits.max(dim=1).indices, num_classes=class_num)
                else:
                    pretrained_temperature = temperature
                    pretrained_probs = nn.Softmax(dim=1)(pretrained_logits / pretrained_temperature)
                mixed_probs = source_probs * pretrained_probs
                mixed_probs = F.normalize(mixed_probs, p=1.0, dim=1)
            else:
                mixed_probs = source_probs
        else:
            source_probs = F.one_hot(labels, num_classes=class_num)
            mixed_probs = source_probs
        counter += len(inputs)

        if start_target:
            centroids = torch.tensordot(mixed_probs.float().t(), pretrained_feas, dims=1)
            all_source_probs = source_probs
            if use_source_pseudolabel and (netC is not None):
                all_pretrained_probs = pretrained_probs
            start_target = False
        else:
            centroids += torch.tensordot(mixed_probs.float().t(), pretrained_feas, dims=1)
            all_source_probs = torch.cat((all_source_probs, source_probs), 0)
            if use_source_pseudolabel and (netC is not None):
                all_pretrained_probs = torch.cat((all_pretrained_probs, pretrained_probs), 0)

    if use_source_pseudolabel and (netC is not None):
        all_mixed_probs = all_source_probs * all_pretrained_probs
        all_mixed_probs = F.normalize(all_mixed_probs, p=1.0, dim=1)
    else:
        all_mixed_probs = all_source_probs

    centroids = centroids / (args.epsilon + all_mixed_probs.sum(axis=0)[:, None])  # class centroids, size = # classes x # features

    return centroids


@torch.no_grad()
def get_centroids_reclip(args, loader, v_model, t_model,
    temperature=1., class_num=None, model_type=None,
    reorder=None, classnames=None):
    '''
    Compute centroids according to soft target pseudolabels from pseudolabel model.

    :param args: all arguments of current run
    :param loader: data loader
    :param source_netF: source feature extractor
    :param source_netB: source bottleneck
    :param source_netC: source classifier
    :param netF: feature extractor
    :param netB: projection layer
    :param use_source_pseudolabel: use source pseudolabel if True, use true label if False
    :param class_num: number of classes
    :param model_type: type of model for source pseudo labels
    :return: centroids of netF features, with size = # classes x # features
    '''
    class_num = args.class_num if class_num is None else class_num
    start_target = True
    iter_target = iter(loader)
    load_img = 0 if model_type == 'adapt' else 2
    counter = 0
    for _ in range(len(loader)):
        data = iter_target.next()
        inputs = data[load_img].cuda()
        labels = data[1].cuda()
        pretrained_inputs = data[-1].cuda()
        pretrained_feas = v_model(pretrained_inputs)
        pretrained_feas = torch.cat((pretrained_feas, torch.ones(pretrained_feas.size(0), 1, device=pretrained_feas.device)), 1)  # add feature column of 1's to prevent small norm
        pretrained_feas = (pretrained_feas.t() / torch.norm(pretrained_feas, p=2, dim=1)).t()  # normalize features for each sample

        v_logits, _ = v_model.forward_reclip(pretrained_inputs)
        v_logits = v_logits[:, reorder]
        t_logits, _ = t_model(pretrained_inputs, classnames)
        mixed_logits = 0.5 * (t_logits + v_logits)
        mixed_probs = nn.Softmax(dim=1)(mixed_logits)
        counter += len(inputs)

        if start_target:
            centroids = torch.tensordot(mixed_probs.float().t(), pretrained_feas, dims=1)
            all_labels = labels
            all_mixed_probs = mixed_probs
            start_target = False
        else:
            centroids += torch.tensordot(mixed_probs.float().t(), pretrained_feas, dims=1)
            all_labels = torch.cat((all_labels, labels))
            all_mixed_probs = torch.cat((all_mixed_probs, mixed_probs), 0)

    clip_acc = torch.eq(all_labels, torch.argmax(all_mixed_probs, dim=-1)).float().mean() * 100.
    centroids = centroids / (args.epsilon + all_mixed_probs.sum(axis=0)[:, None])  # class centroids, size = # classes x # features

    return centroids, clip_acc


@torch.no_grad()
def get_centroids_pouf(args, loader, model,
    temperature=1., class_num=None, model_type=None, reorder=None):
    '''
    Compute centroids according to soft target pseudolabels from pseudolabel model.

    :param args: all arguments of current run
    :param loader: data loader
    :param source_netF: source feature extractor
    :param source_netB: source bottleneck
    :param source_netC: source classifier
    :param netF: feature extractor
    :param netB: projection layer
    :param use_source_pseudolabel: use source pseudolabel if True, use true label if False
    :param class_num: number of classes
    :param model_type: type of model for source pseudo labels
    :return: centroids of netF features, with size = # classes x # features
    '''
    class_num = args.class_num if class_num is None else class_num
    start_target = True
    iter_target = iter(loader)
    load_img = 0 if model_type == 'adapt' else 2
    counter = 0
    for _ in range(len(loader)):
        data = iter_target.next()
        inputs = data[load_img].cuda()
        labels = data[1].cuda()
        pretrained_inputs = data[-1].cuda()
        pretrained_feas = model(pretrained_inputs)
        pretrained_feas = torch.cat((pretrained_feas, torch.ones(pretrained_feas.size(0), 1, device=pretrained_feas.device)), 1)  # add feature column of 1's to prevent small norm
        pretrained_feas = (pretrained_feas.t() / torch.norm(pretrained_feas, p=2, dim=1)).t()  # normalize features for each sample

        # compute output
        sim_t = model(pretrained_inputs)
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * sim_t
        if reorder is not None:
            logits = logits[:, reorder]
        mixed_probs = nn.Softmax(dim=1)(logits)
        counter += len(inputs)

        if start_target:
            centroids = torch.tensordot(mixed_probs.float().t(), pretrained_feas, dims=1)
            all_labels = labels
            all_mixed_probs = mixed_probs
            start_target = False
        else:
            centroids += torch.tensordot(mixed_probs.float().t(), pretrained_feas, dims=1)
            all_labels = torch.cat((all_labels, labels))
            all_mixed_probs = torch.cat((all_mixed_probs, mixed_probs), 0)

    clip_acc = torch.eq(all_labels, torch.argmax(all_mixed_probs, dim=-1)).float().mean() * 100.
    centroids = centroids / (args.epsilon + all_mixed_probs.sum(axis=0)[:, None])  # class centroids, size = # classes x # features

    return centroids, clip_acc


@torch.no_grad()
def get_probs_reclip(args, loader, v_model, t_model,
    temperature=1., class_num=None, model_type=None,
    reorder=None, classnames=None):
    '''
    Compute centroids according to soft target pseudolabels from pseudolabel model.

    :param args: all arguments of current run
    :param loader: data loader
    :param source_netF: source feature extractor
    :param source_netB: source bottleneck
    :param source_netC: source classifier
    :param netF: feature extractor
    :param netB: projection layer
    :param use_source_pseudolabel: use source pseudolabel if True, use true label if False
    :param class_num: number of classes
    :param model_type: type of model for source pseudo labels
    :return: centroids of netF features, with size = # classes x # features
    '''
    class_num = args.class_num if class_num is None else class_num
    start_target = True
    iter_target = iter(loader)
    load_img = 0 if model_type == 'adapt' else 2
    counter = 0
    for _ in range(len(loader)):
        data = iter_target.next()
        inputs = data[load_img].cuda()
        labels = data[1].cuda()
        pretrained_inputs = data[-1].cuda()

        v_logits, _ = v_model.forward_reclip(pretrained_inputs)
        if reorder is not None:
            v_logits = v_logits[:, reorder]
        t_logits, _ = t_model(pretrained_inputs, classnames)
        mixed_logits = 0.5 * (t_logits + v_logits)
        mixed_probs = nn.Softmax(dim=1)(mixed_logits)
        counter += len(inputs)

        if start_target:
            all_labels = labels
            all_mixed_probs = mixed_probs
            start_target = False
        else:
            all_labels = torch.cat((all_labels, labels))
            all_mixed_probs = torch.cat((all_mixed_probs, mixed_probs), 0)

    clip_acc = torch.eq(all_labels, torch.argmax(all_mixed_probs, dim=-1)).float().mean() * 100.

    return all_mixed_probs, clip_acc.item()


@torch.no_grad()
def get_probs_pouf(args, loader, model,
    temperature=1., class_num=None, model_type=None,
    reorder=None):
    '''
    Compute centroids according to soft target pseudolabels from pseudolabel model.

    :param args: all arguments of current run
    :param loader: data loader
    :param source_netF: source feature extractor
    :param source_netB: source bottleneck
    :param source_netC: source classifier
    :param netF: feature extractor
    :param netB: projection layer
    :param use_source_pseudolabel: use source pseudolabel if True, use true label if False
    :param class_num: number of classes
    :param model_type: type of model for source pseudo labels
    :return: centroids of netF features, with size = # classes x # features
    '''
    class_num = args.class_num if class_num is None else class_num
    start_target = True
    iter_target = iter(loader)
    load_img = 0 if model_type == 'adapt' else 2
    counter = 0
    for _ in range(len(loader)):
        data = iter_target.next()
        inputs = data[load_img].cuda()
        labels = data[1].cuda()
        pretrained_inputs = data[-1].cuda()

        # compute output
        sim_t = model(pretrained_inputs)
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * sim_t
        if reorder is not None:
            logits = logits[:, reorder]
        mixed_probs = nn.Softmax(dim=1)(logits)
        counter += len(inputs)

        if start_target:
            all_labels = labels
            all_mixed_probs = mixed_probs
            start_target = False
        else:
            all_labels = torch.cat((all_labels, labels))
            all_mixed_probs = torch.cat((all_mixed_probs, mixed_probs), 0)

    clip_acc = torch.eq(all_labels, torch.argmax(all_mixed_probs, dim=-1)).float().mean() * 100.

    return all_mixed_probs, clip_acc.item()


# CLIP utils

def text_information(clip_model, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        embeddings = []
        labels = []
        label = 0
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = clip_model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            embeddings.append(class_embeddings)
            labels += [label for i in range(len(class_embeddings))]
            label += 1
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.T, embeddings, torch.tensor(labels)

def get_templates(text_templates, dset, target):
    assert text_templates in templates.templates_types
    return templates.get_templates(text_templates, dset, target)