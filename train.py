import torch
import os
import argparse
import json
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from dataset_supervised import HazyTransAirClean_Hazy_Dataset,PairLoader
from utils import AverageMeter, CosineScheduler, pad_img
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from norm_layer import *
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
import vgg16
import math
from SSIM_method import SSIM
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from models.PhDnet import PhDnet_t,PhDnet_s,PhDnet_b,PhDnet_d
from ptflops import get_model_complexity_info
import time
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# print(torch.cuda.is_available())
# print(torch.version.cuda)

"""def smooth(transMap:torch.tensor):
    m = transMap.size(2)
    n = transMap.size(3)
    grad_r = torch.zeros([m - 2, n - 2])
    grad_c = torch.zeros([m - 2, n - 2])
    sum = 0
    for x in range(1, m - 1):
        for y in range(1, n - 1):
            # print(x,y)
            grad_r[x-1, y-1] = (transMap[0,0,x + 1, y] - transMap[0,0,x - 1, y]) / 2
            # print(grad_r)
            grad_c[x-1, y-1] = (transMap[0,0,x, y + 1] - transMap[0,0,x, y - 1]) / 2
            # print(grad_c)
            sum += abs(grad_r[x-1, y-1])*math.pow(math.e,torch.norm(grad_r[x-1, y-1]))+abs(grad_c[x-1, y-1]) * math.pow(math.e, torch.norm(grad_c[x-1, y-1]))
    smooth_result = sum/((m-2)*(n-2))
    #print(smooth)
    return smooth_result"""
def smooth(transMap:torch.tensor):
    m = transMap.size(2)
    n = transMap.size(3)
    sum = 0
    for x in range(1, m - 1):
        for y in range(1, n - 1):
            sum = 0 + abs((transMap[0,0,x + 1, y] - transMap[0,0,x - 1, y]) / 2)*math.pow(math.e,torch.norm((transMap[0,0,x + 1, y] - transMap[0,0,x - 1, y]) / 2))\
                + abs((transMap[0,0,x, y + 1] - transMap[0,0,x, y - 1]) / 2) * math.pow(math.e, torch.norm((transMap[0,0,x, y + 1] - transMap[0,0,x, y - 1]) / 2))
    smooth_result = sum/((m-2)*(n-2))
    return smooth_result

def Dark_prior(img,batchsize):
    w0 = 0.95
    h = img.size()[2]
    w = img.size()[3]
    #print(img.size())
    darkchannel_img,darkchannel_img_indice = torch.max(img,1)
    #print(darkchannel_img.size())
    darkchannel_img = torch.reshape(darkchannel_img, (batchsize, 1, h, w))
    #print(darkchannel_img.size())
    #Air = torch.max(img)
    #t = 1-w0*(darkchannel_img/Air)
    return darkchannel_img  #t



def train(train_loader, network, criterion, optimizer, scaler, frozen_bn=False):
    losses = AverageMeter()
    C_MSEs = 0
    C_SSIMs = 0
    vgg = vgg16.VGGNet().cuda().eval()

    torch.cuda.empty_cache()

    network.eval() if frozen_bn else network.train()  # simplified implementation that other modules may be affected
    #loop = tqdm(train_loader, leave=True)
    #for idx,(source,target) in enumerate(loop):
    #    source_img = source.cuda()
    #    target_img = source.cuda()
    #for batch in enumerate(loop):
    for batch in train_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        #print(source_img[0,:,:,:].shape)

        with autocast(args.use_mp):
            #time_start = time.time()
            output = network(source_img)
            #print(source_img.shape,output.shape)
            #time_end = time.time()
            #time_c = time_end - time_start  # 运行所花时间
            #print('time cost', time_c, 's')
            loss = criterion(output, target_img)
            """syn_clean1, syn_clean2, syn_clean3, syn_clean4 = vgg(output)
            clean1, clean2, clean3, clean4 = vgg(target_img)
            loss_supervised_clean_perceptual = 1 / 4 * (pow(torch.norm(clean1 - syn_clean1), 2)
                                                        + pow(torch.norm(clean2 - syn_clean2), 2)
                                                        + pow(torch.norm(clean3 - syn_clean3), 2)
                                                        + pow(torch.norm(clean4 - syn_clean4), 2))
            #print(loss,loss_supervised_clean_perceptual)
            """
            """loss_supervised_clean_MSE = mse(output * 0.5 + 0.5, target_img * 0.5 + 0.5)
            C_MSEs += loss_supervised_clean_MSE.mean().item()
            loss_supervised_clean_SSIM = SSIM().forward(output * 0.5 + 0.5, target_img * 0.5 + 0.5)
            C_SSIMs += loss_supervised_clean_SSIM.mean().item()"""

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #loop.set_postfix(C_MSE=C_MSEs / (idx + 1), C_SSIM=C_SSIMs / (idx + 1),C_PSNR=10 * math.log10((idx + 1) / C_MSEs), )

        if args.use_ddp: loss = reduce_mean(loss, dist.get_world_size())
        losses.update(loss.item())

    #print(len(train_loader),"train MSE SSIM PSNR",C_MSEs / (len(train_loader)), C_SSIMs / (len(train_loader)),10 * math.log10((len(train_loader)) / C_MSEs))

    return losses.avg


def valid(val_loader, network):
    PSNR_value = AverageMeter()
    SSIM_value = AverageMeter()
    C_MSEs = 0
    C_SSIMs = 0
    mse = nn.MSELoss()
    torch.cuda.empty_cache()
    #batchsize=len(val_loader)
    bi=0

    network.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        batchsize, height, width = source_img.size(0), source_img.size(2), source_img.size(3)

        with torch.no_grad():
            H, W = source_img.shape[2:]
            hazy_img = source_img
            source_img = pad_img(source_img, network.module.patch_size if hasattr(network.module, 'patch_size') else 16)
            output = network(source_img).clamp_(-1, 1)
            output = output[:, :, :H, :W]
            loss_supervised_clean_MSE = mse(output * 0.5 + 0.5, target_img * 0.5 + 0.5)
            C_MSEs += loss_supervised_clean_MSE.mean().item()
            loss_supervised_clean_SSIM = SSIM().forward(output * 0.5 + 0.5, target_img * 0.5 + 0.5)
            C_SSIMs += loss_supervised_clean_SSIM.mean().item()

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        ssim = SSIM().forward(output * 0.5 + 0.5, target_img * 0.5 + 0.5).mean()

        for i in range(batchsize):
            if (bi*batchsize+i)% 10 == 0:
                save_image(torch.cat((torch.reshape(hazy_img[i, :, :, :], (1, 3, height, width)) * 0.5 + 0.5,
                                      torch.reshape(output[i, :, :, :], (1, 3, height, width)) * 0.5 + 0.5,
                                      torch.reshape(target_img[i, :, :, :], (1, 3, height, width)) * 0.5 + 0.5), 0),
                           f"test_results/test_hazy2clean_{bi*batchsize+i}.png")
                #print("out",source_img.shape,output.shape,target_img.shape)
                #save_image(source_img * 0.5 + 0.5,f"test_results/test_hazy_{bi * batchsize + i}.png")
                #save_image(output * 0.5 + 0.5, f"test_results/test_synclean_{bi * batchsize + i}.png")
                #save_image(target_img * 0.5 + 0.5, f"test_results/test_clean_{bi * batchsize + i}.png")

        # if args.use_ddp: psnr = reduce_mean(psnr, dist.get_world_size())		# comment this line for more accurate validation
        SSIM_value.update(ssim.item(), source_img.size(0))
        PSNR_value.update(psnr.item(), source_img.size(0))

        bi=bi+1

    print(len(val_loader), "val MSE SSIM PSNR", C_MSEs / (len(val_loader)), C_SSIMs / (len(val_loader)), 10 * math.log10((len(val_loader)) / C_MSEs))
    return SSIM_value.avg,PSNR_value.avg




parser = argparse.ArgumentParser()
parser.add_argument('--model', default='PhDnet_t', type=str, help='model name')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--use_mp', action='store_true', default=False, help='use Mixed Precision')
parser.add_argument('--use_ddp', action='store_true', default=False, help='use Distributed Data Parallel')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--train_set', default='RSHaze_G/train', type=str, help='train dataset name')#RSHaze_G,RSHaze_S,RSHaze_L,SOTS
parser.add_argument('--val_set', default='RSHaze_G/test', type=str, help='valid dataset name')
parser.add_argument('--exp', default='RSHaze_G', type=str, help='experiment setting')
args = parser.parse_args()

# training environment
if args.use_ddp:
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    if local_rank == 0: print('==> Using DDP.')
else:
    world_size = 1

# training config
with open(os.path.join('configs', args.exp, 'base.json'), 'r') as f:
    b_setup = json.load(f)

variant = args.model.split('_')[-1]
config_name = 'model_' + variant + '.json' if variant in ['t', 's', 'b',
                                                          'd'] else 'default.json'  # default.json as baselines' configuration file
with open(os.path.join('configs', args.exp, config_name), 'r') as f:
    m_setup = json.load(f)


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    # define network, and use DDP for faster training
    network = eval(args.model)()
    #network = eval(args.model.split('_')[0])
    network.cuda()

    macs, params = get_model_complexity_info(network, (3, 256, 256), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    if args.use_ddp:
        print("Use DDP!")
        network = DistributedDataParallel(network, device_ids=[local_rank], output_device=local_rank)
        if m_setup['batch_size'] // world_size < 16:
            if local_rank == 0: print('==> Using SyncBN because of too small norm-batch-size.')
            nn.SyncBatchNorm.convert_sync_batchnorm(network)
    else:
        print("Use DP!")
        network = DataParallel(network)#torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        if m_setup['batch_size'] // torch.cuda.device_count() < 16:
            print('==> Using SyncBN because of too small norm-batch-size.')
            convert_model(network)
    # define loss function
    criterion = nn.L1Loss()
    # define optimizer
    optimizer = torch.optim.AdamW(network.parameters(), lr=m_setup['lr'], weight_decay=b_setup['weight_decay'])
    lr_scheduler = CosineScheduler(optimizer, param_name='lr', t_max=b_setup['epochs'], value_min=m_setup['lr'] * 1e-2,
                                   warmup_t=b_setup['warmup_epochs'], const_t=b_setup['const_epochs'])
    wd_scheduler = CosineScheduler(optimizer, param_name='weight_decay', t_max=b_setup['epochs'])  # seems not to work
    scaler = GradScaler()

    # load saved model
    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(os.path.join(save_dir, args.model + '.pth')):
        best_psnr = 0
        cur_epoch = 0
    else:
        if not args.use_ddp or local_rank == 0: print('==> Loaded existing trained model.')
        model_info = torch.load(os.path.join(save_dir, args.model + '.pth'), map_location='cpu')
        network.load_state_dict(model_info['state_dict'])
        optimizer.load_state_dict(model_info['optimizer'])
        lr_scheduler.load_state_dict(model_info['lr_scheduler'])
        wd_scheduler.load_state_dict(model_info['wd_scheduler'])
        scaler.load_state_dict(model_info['scaler'])
        cur_epoch = model_info['cur_epoch']
        best_psnr = model_info['best_psnr']

    # define dataset
    print(args.train_set)
    train_dataset = PairLoader(os.path.join(args.data_dir, args.train_set), 'train',
                               b_setup['t_patch_size'],
                               b_setup['edge_decay'],
                               b_setup['data_augment'],
                               b_setup['cache_memory'])
    print("batchsize", m_setup['batch_size'] // (world_size))
    train_loader = DataLoader(train_dataset,
                              batch_size=m_setup['batch_size'] // (world_size),
                              sampler=RandomSampler(train_dataset, num_samples=b_setup['num_iter'] // (world_size)),
                              #shuffle=True,
                              num_workers=args.num_workers // (world_size),
                              pin_memory=True,
                              drop_last=True,
                              persistent_workers=True)  # comment this line for cache_memory

    val_dataset = PairLoader(os.path.join(args.data_dir, args.val_set), b_setup['valid_mode'],
                             b_setup['v_patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=max(int(m_setup['batch_size'] * b_setup['v_batch_ratio'] // (world_size)), 1),
                            # sampler=DistributedSampler(val_dataset, shuffle=False),		# comment this line for more accurate validation
                            num_workers=args.num_workers // (world_size),
                            pin_memory=True)

    # start training
    if not args.use_ddp or local_rank == 0:
        print('==> Start training, current model name: ' + args.model)
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

    for epoch in tqdm(range(cur_epoch, b_setup['epochs'] + 1)):
        frozen_bn = epoch > (b_setup['epochs'] - b_setup['frozen_epochs'])
        #print(epoch,"bn",frozen_bn)
        #avg_ssim, avg_psnr = valid(val_loader, network)
        loss = train(train_loader, network, criterion, optimizer, scaler, frozen_bn)
        lr_scheduler.step(epoch + 1)
        wd_scheduler.step(epoch + 1)

        if not args.use_ddp or local_rank == 0:
            writer.add_scalar('train_loss', loss, epoch)
            #print(loss)

        if epoch % b_setup['eval_freq'] == 0:
            avg_ssim, avg_psnr = valid(val_loader, network)
            writer.add_scalar('SSIM', avg_ssim, epoch)
            writer.add_scalar('PSNR', avg_psnr, epoch)

            if not args.use_ddp or local_rank == 0:
                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    torch.save({'cur_epoch': epoch + 1,
                                'best_psnr': best_psnr,
                                'state_dict': network.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'wd_scheduler': wd_scheduler.state_dict(),
                                'scaler': scaler.state_dict()},
                               os.path.join(save_dir, args.model + '.pth'))

                writer.add_scalar('valid_psnr', avg_psnr, epoch)
                writer.add_scalar('best_psnr', best_psnr, epoch)
                print("valhistory",loss, avg_psnr, best_psnr)

            if args.use_ddp: dist.barrier()


if __name__ == '__main__':
    main()
