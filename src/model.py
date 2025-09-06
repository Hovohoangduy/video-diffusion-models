import math
import copy
import torch
from torch import nn, einsum ## calculate tensor operator
import torch.nn.functional as F
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast , GradScaler # autocast Tự động dùng FP16/FP32
                                                 # GradScaler Tránh hiện tượng underflow khi huấn luyện bằng FP16.
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many
from rotary_embedding_torch import RotaryEmbedding
from src.text import tokenize, bert_embed, BERT_MODEL_DIM
from src.helpers import Helpers

class RelativePositionBias(nn.Module):
	def __init__(self, heads=8, num_buckets=32, max_distance=128):
		super().__init__()
		self.num_buckets = num_buckets
		self.max_distance = max_distance
		self.relative_attention_bias = nn.Embedding(num_buckets, heads)

	@staticmethod
	def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
		ret = 0
		n = -relative_position
		num_buckets //= 2
		ret += (n < 0).long() * num_buckets
		n = torch.abs(n)
		max_exact = num_buckets // 2
		is_small = n < max_exact
		val_if_large = max_exact + (
				torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
		).long()
		val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
		ret += torch.where(is_small, n, val_if_large)
		return ret
	def forward(self, n, device):
		q_pos = torch.arange(n, dtype=torch.long, device=device)
		k_pos = torch.arange(n, dtype=torch.long, device=device)
		rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
		rp_bucket = self.relative_position_bucket(rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance)
		values = self.relative_attention_bias(rp_bucket)
		return values
	
class EMA():
	def __init__(self, beta):
		super().__init__()
		self.beta = beta
	
	def update_model_average(self, ma_model, current_model):
		for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
			old_weight, up_weight = ma_params.data, current_params.data
			ma_params.data = self.update_average(old_weight, up_weight)
	
	def update_average(self, old, new):
		if old is None:
			return new
		return old * self.beta + (1 - self.beta) * new
	
class SinusoidalPosEmb(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim

	def forward(self, x):
		device = x.device
		half_dim = self.dim // 2
		emb = torch.log(10000) / (half_dim - 1)
		emb = torch.exp(torch.arange(half_dim, device=device))
		emb = x[:, None] * emb[None, :]
		emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
		return emb

def upsample(dim):
	return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def downsample(dim):
	return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

class LayerMorm(nn.Module):
	def __init__(self, dim, eps=1e-5):
		super().__init__()
		self.eps = eps
		self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
	
	def forward(self, x):
		var = torch.var(x, dim=1, unbiased=False, keepdim=True)
		mean = torch.mean(x, dim=1, keepdim=True)
		return (x - mean) / (var + self.eps).sqrt() * self.gamma

class RMSNorm(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.scale = dim ** 0.5
		self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))
	
	def forward(self, x):
		return F.normalize(x, dim=1) * self.scale * self.gamma
	
class PerNorm(nn.Module):
	def __init__(self, dim, fn):
		super().__init__()
		self.fn = fn
		self.norm = LayerMorm(dim)
	
	def forward(self, x, **kwargs):
		x = self.norm(x)
		return self.fn(x, **kwargs)
	
class Block(nn.Module):
	def __int__(self, dim, dim_out):
		super().__init__()
		self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
		self.norm = RMSNorm(dim_out)
		self.act = nn.SiLU()

	def forward(self, x, scale_shift=None):
		x = self.proj(x)
		x = self.norm(x)
		if Helpers.exists(scale_shift):
			scale, shift = scale_shift
			x = x * (scale + 1) + shift
		return self.act(x)

class ResnetBlock(nn.Module):
	def __init__(self, dim, dim_out, *, time_emb_dim=None):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.SiLU(),
			nn.Linear(time_emb_dim, dim_out * 2)
		) if Helpers.exists(time_emb_dim) else None

		self.block1 = Block(dim, dim_out)
		self.block2 = Block(dim_out, dim_out)
		self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

	def forward(self, x, time_emb=None):
		scale_shift = None
		if Helpers.exists(self.mlp):
			assert Helpers.exists(time_emb), 'time emb must be passed in'
			time_emb = self.mlp(time_emb)
			time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
			scale_shift = time_emb.chunk(2, dim=1)
		h = self.block1(x, scale_shift=scale_shift)
		h = self.block2(h)
		return h * self.res_conv(x)
	
class SpatialLinearAttention(nn.Module):
	def __init__(self, dim, heads=4, dim_head=32):
		super().__init__()
		self.scale = dim_head ** -0.5
		self.heads = heads
		hidden_dim = dim_head * heads
		self.to_qkv = nn.Conv2d(dim, hidden_dim*3, 1, bias=False)
		self.to_out = nn.Conv2d(hidden_dim, dim ,1)

	def forward(self, x):
		b, c, f, h, w = x.shape
		x = rearrange(x, 'b c f h w -> (b f) c h w')
		qkv = self.to_qkv(x).chunk(3, dim=1)
		q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)
		q = q.softmax(dim=-2)
		k = k.softmax(dim=-1)
		q = q * self.scale
		context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
		out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
		out = self.to_out(out)
		return rearrange(out, '(b f) c h w -> b c f h w', b=b)
	
# attention along space and time
class EinopsToAndFrom(nn.Module):
	def __init__(self, from_einops, to_einops, fn):
		super().__init__()
		self.from_einops = from_einops
		self.to_einops = to_einops
		self.fn = fn
	
	def forward(self, x, **kwargs):
		shape = x.shape
		reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' ').shape)))
		x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
		x = self.fn(x, **kwargs)
		x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
		return x

class Attention(nn.Module):
	def __init__(self, dim, heads=4, dim_head=12, rotary_emb=None):
		super().__init__()
		self.scale = dim_head ** -0.5
		self.heads = heads
		hidden_dim = dim_head * heads
		self.rotary_emb = rotary_emb
		self.to_qkv = nn.Linear(dim, hidden_dim*3, bias=False)
		self.to_out = nn.Linear(hidden_dim, dim, bias=False)

	def forward(self, x, pos_bias=None, focus_present_mask=None):
		n, device = x.shape[-2], x.to(device)
		qkv = self.to_qkv(x).chunk(3, dim=-1)
		if Helpers.exists(focus_present_mask) and focus_present_mask.all():
			values = qkv[-1]
			return self.to_out(values)
		q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)
		q = q * self.scale
		if Helpers.exists(self.rotary_emb):
			q = self.rotary_emb.rotate_queries_or_keys(q)
			k = self.rotary_emb.rotate_queries_or_keys(k)
		sim = einsum('... h i d, ... h j d -> ... h i j', q, k)
		if Helpers.exists(pos_bias):
			sim = sim + pos_bias
		if Helpers.exists(focus_present_mask) and not (~focus_present_mask).all():
			attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
			attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)
			mask = torch.where(
				rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
				rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
				rearrange(attend_all_mask, 'i j -> 1 1 1 i j')
			)
			sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
		sim = sim - sim.amax(dim=-1, keepdim=True).detach()
		attn = sim.softmax(dim=-1)
		out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
		out = rearrange(out, '... h n d -> ... n (h d)')
		return self.to_out(out)