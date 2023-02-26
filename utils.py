def get_feature_size(arch):
    if 'ResNet' in arch:
        c = 512
    elif arch == 'mobilenet_v3_small':
        c = 576
    elif arch == 'mobilenet_v3_large':
        c = 960
    else:
        raise ValueError('arch not found: ' + arch)
    return c