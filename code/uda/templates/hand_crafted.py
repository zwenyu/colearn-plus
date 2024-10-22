# Hand crafted templates selected by Tip-Adapter
imagenet_templates = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}."
]

flowers_templates = [
    'a photo of a {}, a type of flower.'
]

aircraft_templates = [
    'a photo of a {}, a type of aircraft.'
]


food_templates = [
    'a photo of {}, a type of food.'
]

pets_templates = [
    'a photo of a {}, a type of pet.'
]

cars_templates = [
    'a photo of a {}.'
]

dtd_templates = [
    '{} texture.'
]

caltech101_templates = [
    'a photo of a {}.'
]

ucf101_templates = [
    'a photo of a person doing {}.'
]

sun397_templates = [
    'a photo of a {}.'
]

eurosat_templates = [
    'a centered satellite photo of {}.'
]

officehome_templates = {
    "art": ["art of {}."],
    "clipart": ["clipart of {}."],
    "product": ["product image of {}."],
    "realworld": ["photo of {}"]
}

office_templates = {
    "amazon": ["product image of {}."],
    "dslr": ["dslr photo of {}."],
    "webcam": ["webcam photo of {}."]
}


visda_templates = {
    "train": ["synthetic image of 3D model of {}."],
    "validation": {"photo of {}."}
}

domainnet_templates = {
    "clipart": ["clipart of {}."],
    "painting": ["painting of {}."],
    "real": ["photo of {}."],
    "sketch": ["sketch of {}."]
}


TIP_ADAPTER_TEMPLATES = {
    "oxford_pets": pets_templates,
    "oxford_flowers": flowers_templates,
    "fgvc_aircraft": aircraft_templates,
    "dtd": dtd_templates,
    "eurosat": eurosat_templates,
    "stanford_cars": cars_templates,
    "food101": food_templates,
    "sun397": sun397_templates,
    "caltech101": caltech101_templates,
    "ucf101": ucf101_templates,
    "imagenet": imagenet_templates,
    "imagenet_sketch": imagenet_templates,
    "imagenetv2": imagenet_templates,
    "imagenet_a": imagenet_templates,
    "imagenet_r": imagenet_templates,
    "office-home": officehome_templates,
    "office": office_templates,
    "VISDA-C": visda_templates,
    "domainnet-126": domainnet_templates
}