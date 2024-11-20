'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data
import torch.distributed as dist


def create_dataloader(dataset, dataset_opt, phase,rank,world_size):
    '''create dataloader '''
    if phase == 'train':
        if dist.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,num_replicas=world_size,rank=rank)
            shuffle=False #it is actually shuffled in the main loop
        else:
            train_sampler = None
            shuffle=dataset_opt['use_shuffle']

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=shuffle,
            num_workers=dataset_opt['num_workers'],
            pin_memory=True,
            sampler=train_sampler)
        return dataloader,train_sampler
    elif phase == 'val':
        if dist.is_initialized():
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,num_replicas=world_size,rank=rank)
        else:
            val_sampler=None
        return torch.utils.data.DataLoader(
            dataset, batch_size=world_size,
            #shuffle=False,
            num_workers=1, pin_memory=True,
            sampler=val_sampler)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    from data.LRHR_dataset import LRHRDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                datatype=dataset_opt['datatype'],
                l_resolution=dataset_opt['l_resolution'],
                r_resolution=dataset_opt['r_resolution'],
                split=phase,
                data_len=dataset_opt['data_len'],
                need_LR=(mode == 'LRHR')
                )
    logger = logging.getLogger('base')
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    if rank==0:
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                               dataset_opt['name']))
    return dataset
