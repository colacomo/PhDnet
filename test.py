import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
from utils import AverageMeter, pad_img
from dataset_supervised import PairLoader, write_img, chw_to_hwc, SingleLoader
from models import *
from SSIM_method import SSIM as SSIM_function
from models.PhDnet import PhDnet_t,PhDnet_s,PhDnet_b,PhDnet_d
from ptflops import get_model_complexity_info
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='PRFDN_s', type=str, help='model name')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--test_set', default='RSHaze_G', type=str, help='test dataset name')#RSHaze_S RSHaze_G RSHaze_L SateHaze1k SOTS
parser.add_argument('--exp', default='RSHaze_G', type=str, help='experiment setting')
args = parser.parse_args()


def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, network, result_dir):
	PSNR = AverageMeter()
	SSIM = AverageMeter()
	TIME = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
	os.makedirs(os.path.join(result_dir, 'res_imgs'), exist_ok=True)
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

	for idx, batch in enumerate(test_loader):
		input = batch['source'].cuda()
		target = batch['target'].cuda()

		filename = batch['filename'][0]

		with torch.no_grad():
			H, W = input.shape[2:]
			#print("input", input.shape)
			input = pad_img(input, network.patch_size if hasattr(network, 'patch_size') else 16)
			#input = pad_img(input, (max(H, W) // 64 + 1) * 64)
			#print("pad",input.shape)
			time_start = time.time()
			output = network(input).clamp_(-1, 1)
			#print("output",output.shape)
			time_end = time.time()
			time_c = time_end - time_start  # 运行所花时间
			#print('time cost', time_c, 's')

			output = output[:, :, :H, :W]

			# [-1, 1] to [0, 1]
			output = output * 0.5 + 0.5
			target = target * 0.5 + 0.5

			psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

			_, _, H, W = output.size()
			"""
			down_ratio = max(1, round(min(H, W) / 256))		# Zhou Wang
			ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))), 
							F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))), 
							data_range=1, size_average=False).item()
			"""
			ssim_val = SSIM_function().forward(output, target).mean().item()

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)
		TIME.update(time_c)

		print('Test: [{0}]\t'
			  'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
			  'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
			  .format(idx, psnr=PSNR, ssim=SSIM))

		f_result.write('%s,%.02f,%.03f\n'%(filename, psnr_val, ssim_val))

		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		write_img(os.path.join(result_dir, 'imgs', filename), out_img)
		res = torch.abs(output-target)
		print(torch.max(res))
		res = (res * 10).clamp(0,1)
		res_img = chw_to_hwc(res.detach().cpu().squeeze(0).numpy())
		write_img(os.path.join(result_dir, 'res_imgs', filename), res_img)

	f_result.close()
	print("AVG_Time",TIME.avg)
	print('%.03f | %.04f.csv' % (PSNR.avg, SSIM.avg))
	os.rename(os.path.join(result_dir, 'results.csv'),
			  os.path.join(result_dir, '%.03f_%.04f.csv' % (PSNR.avg, SSIM.avg)))


def test_single(test_loader, network, result_dir):

	torch.cuda.empty_cache()

	network.eval()

	for idx, batch in enumerate(test_loader):
		input = batch['img'].cuda()

		filename = batch['filename'][0]

		with torch.no_grad():
			H, W = input.shape[2:]

			#input = pad_img(input, network.patch_size if hasattr(network, 'patch_size') else 16)
			input = pad_img(input, (max(H, W) // 128 + 1) * 128)
			output = network(input).clamp_(-1, 1)
			output = output[:, :, :H, :W]
			# [-1, 1] to [0, 1]
			output = output * 0.5 + 0.5



		out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
		os.makedirs(os.path.join(result_dir, 'real_hazy'), exist_ok=True)# s1_real_test real_test real_hazy
		print(os.path.join(result_dir, 'real_hazy', filename))
		write_img(os.path.join(result_dir, 'real_hazy', filename), out_img)

	#f_result.close()



def main():
	network = eval(args.model)()
	network.cuda()

	macs, params = get_model_complexity_info(network, (3, 256, 256), as_strings=True,
											 print_per_layer_stat=True, verbose=True)
	print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
	print('{:<30}  {:<8}'.format('Number of parameters: ', params))

	saved_model_dir = os.path.join(args.save_dir, args.exp, 'baseline', args.model+'.pth')

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)

	dataset_dir = os.path.join(args.data_dir, args.test_set, 'test')
	test_dataset = PairLoader(dataset_dir, 'test')
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)
	result_dir = os.path.join(args.result_dir, args.test_set, args.model, args.exp)
	test(test_loader, network, result_dir)
	"""
	result_dir = os.path.join(args.result_dir, args.test_set, args.model)
	dataset_dir_testreal = os.path.join(args.data_dir, 'RSHaze_L','real_test')# 'RSHaze_G','real_test' 'RSHaze_L','real_test''RSHaze_S','real_test'
	testreal_dataset = SingleLoader(dataset_dir_testreal)
	testreal_loader = DataLoader(testreal_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)
	#result_dir = os.path.join(args.result_dir, args.test_set, args.exp, args.model)
	test_single(testreal_loader, network, result_dir)
"""



if __name__ == '__main__':
	main()
