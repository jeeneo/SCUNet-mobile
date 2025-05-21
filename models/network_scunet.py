# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_, DropPath


class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).permute(2, 0, 1)
        )

    def generate_mask(self, h: int, w: int, p: int, shift: int) -> torch.Tensor:
        """ generating the mask of SW-MSA
        Args:
            h, w: height and width
            p: window size
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # Convert integers to tensors and ensure they're on the right device
        h = torch.tensor(h, device=self.relative_position_params.device)
        w = torch.tensor(w, device=self.relative_position_params.device)
        p = torch.tensor(p, device=self.relative_position_params.device)
        shift = torch.tensor(shift, device=self.relative_position_params.device)
        
        # supporting square window
        attn_mask = torch.zeros(h.item(), w.item(), p.item(), p.item(), p.item(), p.item(), 
                              dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        # Reshape to 1, 1, (w1*w2), (p1*p2), (p3*p4)
        attn_mask = attn_mask.view(1, 1, h.item()*w.item(), p.item()*p.item(), p.item()*p.item())
        return attn_mask

    def relative_embedding(self):
        # Pre-compute all coordinates in a tensor form
        coords = torch.zeros((self.window_size * self.window_size, 2), dtype=torch.long, device=self.relative_position_params.device)
        idx = 0
        for i in range(self.window_size):
            for j in range(self.window_size):
                coords[idx, 0] = i
                coords[idx, 1] = j
                idx += 1
        
        # Calculate all possible coordinate differences
        relation = coords.unsqueeze(1) - coords.unsqueeze(0) + self.window_size - 1
        
        # Get the corresponding relative position embeddings
        return self.relative_position_params[:, relation[:, :, 0], relation[:, :, 1]]

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        b, h, w, c = x.shape
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        # Native PyTorch view and permute instead of einops.rearrange
        x = x.view(b, h // self.window_size, self.window_size, w // self.window_size, self.window_size, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        h_windows = x.size(1)
        w_windows = x.size(2)

        x = x.view(b, h_windows * w_windows, self.window_size * self.window_size, c)
        qkv = self.embedding_layer(x)
        
        # Native PyTorch view and permute instead of einops.rearrange
        qkv = qkv.view(b, -1, self.window_size * self.window_size, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 1, 2, 4, 5).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calculate attention scores
        q = q * self.scale
        attn = torch.einsum('bnphc,bnqhc->bnhpq', q, k)
        
        rel_pos_embedding = self.relative_embedding()
        window_area = self.window_size * self.window_size
        rel_pos_embedding = rel_pos_embedding.view(self.n_heads, window_area, window_area)
        rel_pos_embedding = rel_pos_embedding.unsqueeze(0).unsqueeze(0)
        attn = attn + rel_pos_embedding

        # Apply mask if needed
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            # attn: [b, n_windows, n_heads, window_area, window_area]
            # attn_mask: [1, 1, n_windows, window_area, window_area]
            b = x.shape[0]
            n_windows = h_windows * w_windows
            attn = attn.view(b * n_windows, self.n_heads, window_area, window_area)
            attn_mask = attn_mask.expand(b, 1, n_windows, window_area, window_area).reshape(b * n_windows, 1, window_area, window_area)
            attn = attn.masked_fill(attn_mask, float("-inf"))
            attn = attn.view(b, n_windows, self.n_heads, window_area, window_area)
        
        # Softmax and apply attention
        attn = torch.softmax(attn, dim=-1)
        output = torch.einsum('bnhpq,bnqhc->bnphc', attn, v)
        
        # Native PyTorch view and permute instead of einops.rearrange
        output = output.permute(1, 0, 2, 3, 4).contiguous().view(b, h_windows, w_windows, self.window_size, self.window_size, c)
        output = output.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, c)
        output = self.linear(output)
        
        # Native PyTorch view and permute instead of einops.rearrange
        output = output.view(b, h_windows, self.window_size, w_windows, self.window_size, c)
        output = output.permute(0, 1, 2, 3, 4, 5).contiguous()
        output = output.view(b, h_windows * self.window_size, w_windows * self.window_size, c)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type, self.input_resolution)
        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.conv_block = nn.Sequential(
                nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
                )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        
        # BCHW -> BHWC for transformer blocks
        trans_x = trans_x.permute(0, 2, 3, 1)
        trans_x = self.trans_block(trans_x)
        # BHWC -> BCHW for concatenation
        trans_x = trans_x.permute(0, 3, 1, 2)
        
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res

        return x


class SCUNet(nn.Module):

    def __init__(self, in_nc=3, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=256):
        super(SCUNet, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 8

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        # use in_nc for input channels (for greyscale, in_nc=1)
        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]

        begin = 0
        self.m_down1 = [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(config[0])] + \
                      [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[1])] + \
                      [nn.Conv2d(2*dim, 4*dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(config[2])] + \
                      [nn.Conv2d(4*dim, 8*dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [ConvTransBlock(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
                    for i in range(config[3])]

        begin += config[3]
        self.m_up3 = [nn.ConvTranspose2d(8*dim, 4*dim, 2, 2, 0, bias=False),] + \
                      [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(config[4])]
                      
        begin += config[4]
        self.m_up2 = [nn.ConvTranspose2d(4*dim, 2*dim, 2, 2, 0, bias=False),] + \
                      [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(config[5])]
                      
        begin += config[5]
        self.m_up1 = [nn.ConvTranspose2d(2*dim, dim, 2, 2, 0, bias=False),] + \
                    [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(config[6])]

        self.m_tail = [nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)  
        #self.apply(self._init_weights)

    def forward(self, x0):
        h, w = x0.shape[2], x0.shape[3]
        
        # Calculate padding sizes
        paddingBottom = int(math.ceil(h/64)*64-h)
        paddingRight = int(math.ceil(w/64)*64-w)
        
        # Apply padding
        x0_padded = nn.functional.pad(x0, (0, paddingRight, 0, paddingBottom), mode='replicate')

        x1 = self.m_head(x0_padded)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        # Remove padding
        x = x[:, :, :h, :w]
        
        return x


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SCUNetWrapper(nn.Module):
    """Wrapper for SCUNet to make it script-friendly."""
    def __init__(self, in_nc=3, config=[2, 2, 2, 2, 2, 2, 2], dim=64, drop_path_rate=0.0, input_resolution=256):
        super(SCUNetWrapper, self).__init__()
        self.model = SCUNet(in_nc=in_nc, config=config, dim=dim, drop_path_rate=drop_path_rate, input_resolution=input_resolution)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == '__main__':
    net = SCUNet()
    x = torch.randn((2, 3, 64, 128))
    x = net(x)
    print(x.shape)
