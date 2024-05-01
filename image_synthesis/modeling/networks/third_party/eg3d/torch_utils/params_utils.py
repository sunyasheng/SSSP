def freeze_model(net):
    for param in net.parameters():
        param.requires_grad = False