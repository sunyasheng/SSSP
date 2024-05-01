def freeze_network(network):
    for p in network.parameters():
        p.requires_grad = False


def unfreeze_network(network):
    for p in network.parameters():
        p.requires_grad = True


