import torch


def freeze_model(net):
    for param in net.parameters():
        param.requires_grad = False


def copy_state_dict(state_dict, model, strip=None, replace=None, add_prefix=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        # if add_prefix is not None:
        #     name = add_prefix + name
        #     print('name; ', name)
        if strip is not None and replace is None and name.startswith(strip):
            name = name[len(strip):]
        if strip is not None and replace is not None:
            name = name.replace(strip, replace)
        if name not in tgt_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)
