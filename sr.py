import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np

#import habana_frameworks.torch.core as htcore
import torch.distributed as dist

def sync_between_proc(value,world_size,device):
    value_tensor=torch.tensor(value,device=device)
    if dist.is_initialized():
        dist.all_reduce(value_tensor,op=dist.ReduceOp.SUM)
        value_tensor=value_tensor/world_size
    return value_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')
    parser.add_argument('--distributed', action='store_true',help='To run in distributed mode on multiple devices')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    #Initialize parallelization if available (and in the config file)
    if dist.is_available() and args.distributed:
        #from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
        #world_size, rank, local_rank = initialize_distributed_hpu()
        #dist.init_process_group(backend='hccl',
        #        world_size=world_size,
        #        rank=rank)
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print('Hi Im Worker:',rank)

    else:
        print('Running on a single device!')
        rank=0
        world_size=1

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    if rank==0:
        logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None


    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader,train_sampler = Data.create_dataloader(
                train_set, dataset_opt, phase,
                rank, world_size)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase,
                rank, world_size)
    if rank==0:
        logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    if rank==0:
        logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        if rank==0:
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(
                current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            if dist.is_initialized():
                train_sampler.set_epoch(current_epoch) #shuffle the sampler each epoch

            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                if dist.is_initialized():
                    dist.barrier()

                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        v_sync=sync_between_proc(v,world_size,diffusion.device)
                        message += '{:s}: {:.4e} '.format(k, v_sync)
                        if rank==0:
                            tb_logger.add_scalar(k, v_sync, current_step)
                            logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                        # generation
                        Metrics.save_img(
                            hr_img, '{}/{}_{}_{}_hr.png'.format(result_path, current_step, rank, idx))
                        Metrics.save_img(
                            sr_img, '{}/{}_{}_{}_sr.png'.format(result_path, current_step, rank, idx))
                        Metrics.save_img(
                            lr_img, '{}/{}_{}_{}_lr.png'.format(result_path, current_step, rank, idx))
                        Metrics.save_img(
                            fake_img, '{}/{}_{}_{}_inf.png'.format(result_path, current_step, rank, idx))
                        if rank==0:
                            tb_logger.add_image(
                                'Iter_{}'.format(current_step),
                                np.transpose(np.concatenate(
                                    (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                                idx)
                        avg_psnr += Metrics.calculate_psnr(
                            sr_img, hr_img)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )

                    avg_psnr=sync_between_proc(avg_psnr,world_size,diffusion.device)    
                    avg_psnr = avg_psnr / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    if rank==0:
                        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                        logger_val = logging.getLogger('val')  # validation logger
                        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                            current_epoch, current_step, avg_psnr))
                        # tensorboard logger
                        tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0 and rank==0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        if rank==0:
            logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, rank, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}_{}_{}_sr_process.png'.format(result_path, current_step, rank, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_{}_sr.png'.format(result_path, current_step, rank, idx))

            Metrics.save_img(
                hr_img, '{}/{}_{}_{}_hr.png'.format(result_path, current_step, rank, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_{}_lr.png'.format(result_path, current_step, rank, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_{}_inf.png'.format(result_path, current_step, rank, idx))

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        avg_psnr=sync_between_proc(avg_psnr,world_size,diffusion.device)
        avg_ssim=sync_between_proc(avg_ssim,world_size,diffusion.device)
        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        if rank==0:
            logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
            logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
            logger_val = logging.getLogger('val')  # validation logger
            logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
                current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
