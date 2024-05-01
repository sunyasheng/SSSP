import torch
import numpy as np
from torch.utils.data import ConcatDataset
from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.distributed.distributed import is_distributed

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_test_dataloader(config, meta_path, args=None, return_dataset=False):
    dataset_cfg = config['dataloader']
    test_dataset = []
    for ds_cfg in dataset_cfg['test_datasets']:
        # ds_cfg['params']['path'] = dataset_cfg.get('path', '')
        ds_cfg['params']['meta_sketch_path'] = meta_path
        ds = instantiate_from_config(ds_cfg)
        # import pdb; pdb.set_trace();
        test_dataset.append(ds)
    if len(test_dataset) > 1:
        test_dataset = ConcatDataset(test_dataset)
    else:
        test_dataset = test_dataset[0]

    num_workers = dataset_cfg['num_workers']
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=dataset_cfg['batch_size'], 
                                            shuffle=False, #(val_sampler is None),
                                            num_workers=num_workers, 
                                            pin_memory=True, 
                                            sampler=None, 
                                            drop_last=True,
                                            persistent_workers=False)
    
    dataload_info = {
        'test_loader': test_loader,
    }
    
    if return_dataset:
        dataload_info['test_dataset'] = test_dataset
        
    return dataload_info

def build_dataloader(config, args=None, return_dataset=False):
    dataset_cfg = config['dataloader']
    # import pdb; pdb.set_trace()
    train_dataset = []
    for ds_cfg in dataset_cfg['train_datasets']:
        # ds_cfg['params']['path'] = dataset_cfg.get('path', '')
        # ds_cfg['params']['path'] = ''
        ds = instantiate_from_config(ds_cfg)
        train_dataset.append(ds)
    if len(train_dataset) > 1:
        train_dataset = ConcatDataset(train_dataset)
    else:
        train_dataset = train_dataset[0]
    
    val_dataset = []
    for ds_cfg in dataset_cfg['validation_datasets']:
        # ds_cfg['params']['path'] = dataset_cfg.get('path', '')
        # ds_cfg['params']['path'] = ''
        ds = instantiate_from_config(ds_cfg)
        val_dataset.append(ds)
    if len(val_dataset) > 1:
        val_dataset = ConcatDataset(val_dataset)
    else:
        val_dataset = val_dataset[0]
    
    if args is not None and args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        train_iters = len(train_sampler) // dataset_cfg['batch_size']
        val_iters = len(val_sampler) // dataset_cfg['batch_size']
    else:
        train_sampler = None
        val_sampler = None
        train_iters = len(train_dataset) // dataset_cfg['batch_size']
        val_iters = len(val_dataset) // dataset_cfg['batch_size']

    num_workers = dataset_cfg['num_workers']
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=dataset_cfg['batch_size'], 
                                               shuffle=(train_sampler is None),
                                               num_workers=num_workers, 
                                               pin_memory=True, 
                                               sampler=train_sampler, 
                                               drop_last=True,
                                               persistent_workers=False,
                                               worker_init_fn=worker_init_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=dataset_cfg['batch_size'], 
                                             shuffle=False, #(val_sampler is None),
                                             num_workers=num_workers, 
                                             pin_memory=True, 
                                             sampler=val_sampler, 
                                             drop_last=True,
                                             persistent_workers=False,
                                             worker_init_fn=worker_init_fn)

    dataload_info = {
        'train_loader': train_loader,
        'validation_loader': val_loader,
        'train_iterations': train_iters,
        'validation_iterations': val_iters
    }
    
    if return_dataset:
        dataload_info['train_dataset'] = train_dataset
        dataload_info['validation_dataset'] = val_dataset

    return dataload_info
