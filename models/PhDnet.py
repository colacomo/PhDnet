import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
from torchsummary import summary
#from .norm_layer import *



class ConvLayer(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
		super().__init__()
		self.dim = dim

		self.net_depth = net_depth
		self.kernel_size = kernel_size

		self.Wv1 = nn.Sequential(
			nn.Conv2d(dim, dim, 1), #nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2),  # PWConv
			nn.InstanceNorm2d(dim, affine=True),#nn.InstanceNorm2d(dim, affine=True),
			nn.ReLU(),
			nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, padding_mode='reflect'),  # DWConv
		)

		self.Wv2 = nn.Sequential(
			nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2),  # ASConv
			nn.InstanceNorm2d(dim, affine=True),#nn.InstanceNorm2d(dim, affine=True),
			nn.ReLU(),
			nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, padding_mode='reflect'),
		)
		self.sig = nn.Sigmoid()

		self.gate = nn.Conv2d(dim * 2, 3, 3, 1, 1, bias=True)

		self.proj = nn.Conv2d(dim, dim, 1)  # PWConv

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.net_depth) ** (-1/4)    # self.net_depth ** (-1/2), the deviation seems to be too small, a bigger one may be better
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, X):
		x1 = self.Wv1(X)
		x2 = self.Wv2(X)
		#x3 = self.Wv3(X)
		gates = self.sig(self.gate(torch.cat((x1, x2), dim=1)))#gates = self.gate(torch.cat((x1, x2, x3), dim=1))
		#print(gates.shape)
		#gates = self.gate(torch.cat((x1, x2), dim=1))
		#print(gates[:, [0], :, :].shape)
		out = x1 * gates[:, 0, :, :].unsqueeze(1) + x2 * gates[:, 1, :, :].unsqueeze(1)# + x3 * gates[:, [2], :, :]
		#out = torch.cat((x1, x2), dim=1) * gates
		out = self.proj(out)
		return out


class BasicBlock(nn.Module):
	def __init__(self, net_depth, dim, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):
		super().__init__()
		self.norm = norm_layer(dim,affine=False)
		self.conv = conv_layer(net_depth, dim, kernel_size, gate_act)
	def forward(self, x):
		identity = x
		x = self.norm(x)
		x = self.conv(x)
		x = identity + x
		return x


class BasicLayer(nn.Module):
	def __init__(self, net_depth, dim, depth, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid):

		super().__init__()
		self.dim = dim
		self.depth = depth

		# build blocks
		self.blocks = nn.ModuleList([
			BasicBlock(net_depth, dim, kernel_size, conv_layer, norm_layer, gate_act)
			for i in range(depth)])

	def forward(self, x):
		for blk in self.blocks:
			x = blk(x)
		return x


class PatchEmbed(nn.Module):
	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.in_chans = in_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = patch_size

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
							  padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

	def forward(self, x):
		#print(x.shape)
		x = self.proj(x)
		return x


class PatchUnEmbed(nn.Module):
	def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.out_chans = out_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = 1

		self.proj = nn.Sequential(
			nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
					  padding=kernel_size//2, padding_mode='reflect'),
			nn.PixelShuffle(patch_size)
		)

	def forward(self, x):
		x = self.proj(x)
		return x


class SKFusion(nn.Module):
	def __init__(self, dim, height=2, reduction=8):
		super(SKFusion, self).__init__()

		self.height = height
		d = max(int(dim/reduction), 4)

		"""self.mlp = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(dim, d, 1, bias=False),
			nn.ReLU(True),
			nn.Conv2d(d, dim*height, 1, bias=False)
		)"""
		self.mlp = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(dim, d, 1, bias=False),
			nn.ReLU(True),
			nn.Conv2d(d, dim * height, 1, bias=False)
		)
		self.conv = nn.Conv2d(dim, dim, 1, bias=False)
		self.deconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)

		self.softmax = nn.Softmax(dim=1)

	def forward(self, in_feats):
		B, C, H, W = in_feats[0].shape

		in_feats = torch.cat(in_feats, dim=1)
		in_feats = in_feats.view(B, self.height, C, H, W)
		feats_sum = torch.sum(in_feats, dim=1)
		attn = self.mlp(feats_sum)
		attn = self.softmax(attn.view(B, self.height, C, 1, 1))

		out = torch.sum(in_feats*attn, dim=1)
		return out

def default_conv(in_channels, out_channels, kernel_size, bias=True):
	return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class PDU(nn.Module):  # physical block
	def __init__(self, channel):
		super(PDU, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.ka = nn.Sequential(
			nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
			nn.Sigmoid()
		)
		self.td = nn.Sequential(
			default_conv(channel, channel, 3),
			default_conv(channel, channel // 8, 3),
			nn.ReLU(inplace=True),
			default_conv(channel // 8, channel, 3),
			nn.Sigmoid()
		)

	def forward(self, x):
		a = self.avg_pool(x)
		a = self.ka(a)
		t = self.td(x)
		j = torch.mul((1 - t), a) + torch.mul(t, x)
		return j

class PhDnet(nn.Module):
	def __init__(self, kernel_size=5, base_dim=16, depths=[4, 4, 4, 4, 4, 4, 4], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion):
		super(PhDnet, self).__init__()
		# setting
		assert len(depths) % 2 == 1
		stage_num = len(depths)
		half_num = stage_num // 2
		net_depth = sum(depths)
		embed_dims = [2**i*base_dim for i in range(half_num)]
		embed_dims = embed_dims + [2**half_num*base_dim] + embed_dims[::-1]

		self.patch_size = 2 ** (stage_num // 2)
		self.stage_num = stage_num
		self.half_num = half_num

		# input convolution
		self.inconv = PatchEmbed(patch_size=1, in_chans=3, embed_dim=embed_dims[0], kernel_size=3)

		# backbone
		self.layers = nn.ModuleList()
		self.downs = nn.ModuleList()
		self.ups = nn.ModuleList()
		self.skips = nn.ModuleList()
		self.fusions = nn.ModuleList()
		self.relu = nn.ReLU(inplace=True)
		self.lrelu = nn.LeakyReLU(inplace=True)
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)
		self.ka = nn.Sequential(
			nn.Conv2d(embed_dims[half_num], embed_dims[half_num] // 8, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(embed_dims[half_num] // 8, embed_dims[half_num], 1, padding=0, bias=True),
			nn.Sigmoid()
		)
		self.td = nn.Sequential(
			default_conv(embed_dims[half_num], embed_dims[half_num], 3),
			default_conv(embed_dims[half_num], embed_dims[half_num] // 8, 3),
			nn.ReLU(inplace=True),
			default_conv(embed_dims[half_num] // 8, embed_dims[half_num], 3),
			nn.Sigmoid()
		)

		for i in range(self.stage_num):
			self.layers.append(BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size,
										  conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

		for i in range(self.half_num):
			self.downs.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.ups.append(PatchUnEmbed(patch_size=2, out_chans=embed_dims[i], embed_dim=embed_dims[i+1]))
			self.skips.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
			self.fusions.append(fusion_layer(embed_dims[i]))

		# output convolution
		self.outconv = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=embed_dims[-1], kernel_size=3)
		self.transoutconv = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=embed_dims[-1], kernel_size=3)
		self.airoutconv = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=embed_dims[-1], kernel_size=3)


	def forward(self, x):
		feat = self.inconv(x)

		skips = []

		for i in range(self.half_num):
			feat = self.layers[i](feat)
			skips.append(self.skips[i](feat))
			feat = self.downs[i](feat)

		feat = self.layers[self.half_num](feat)
		a = self.avg_pool(feat)
		a = self.ka(a)
		t = self.td(feat)
		airminushazy = a - feat
		feat = torch.mul((1/t - 1), airminushazy)
		haze = feat
		#mid = feat
		#print(mid.shape)

		for i in range(self.half_num-1, -1, -1):
			#print(i)
			feat = self.ups[i](feat)
			feat = self.fusions[i]([feat, skips[i]])
			feat = self.layers[self.stage_num-i-1](feat)

		#x = self.outconv(feat) + x
		residual = self.outconv(feat)
		#clean_image = self.relu(x - residual)
		clean_image = x - residual

		return clean_image


__all__ = ['PhDnet', 'PhDnet_t', 'PhDnet_s', 'PhDnet_b', 'PhDnet_d']

# Normalization batch size of 16~32 may be good

def PhDnet_t():	# 4 cards 2080Ti base_dim=24
	return PhDnet(kernel_size=5, base_dim=24, depths=[2, 2, 2, 4, 2, 2, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def PhDnet_s():	# 4 cards 3090
	return PhDnet(kernel_size=5, base_dim=24, depths=[4, 4, 4, 8, 4, 4, 4], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def PhDnet_b():	# 4 cards 3090
	return PhDnet(kernel_size=5, base_dim=24, depths=[8, 8, 8, 16, 8, 8, 8], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)

def PhDnet_d():	# 4 cards 3090
	return PhDnet(kernel_size=5, base_dim=24, depths=[16, 16, 16, 32, 16, 16, 16], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)


