import logging
import torch.distributed as dist
logger = logging.getLogger('base')


def create_model(opt):
    from .model import DDPM as M
    m = M(opt)
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    if rank==0:
        logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
