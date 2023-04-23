import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np
from models.CTrans import SwinTransformer

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, L, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, H, W, L, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows
def window_reverse(windows, window_size, H, W, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        L (int): Length of image
    Returns:
        x: (B, H, W, L, C)
    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, L, -1)
    return x
class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, rpe=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        self.rpe = rpe
        if self.rpe:
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkvx = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkvy = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_dropx = nn.Dropout(attn_drop)
        self.attn_dropy = nn.Dropout(attn_drop)
        self.projx = nn.Linear(dim, dim)
        self.projy = nn.Linear(dim, dim)
        self.proj_dropx = nn.Dropout(proj_drop)
        self.proj_dropy = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmaxx = nn.Softmax(dim=-1)
        self.softmaxy = nn.Softmax(dim=-1)

    def forward(self, x, y, maskx=None, masky=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww, Wt*Ww) or None
        """
        B_, N, C = x.shape #(num_windows*B, Wh*Ww*Wt, C)
        qkvx = self.qkvx(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qx, kx, vx = qkvx[0], qkvx[1], qkvx[2]  # make torchscript happy (cannot use tensor as tuple)
        qkvy = self.qkvy(y).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qy, ky, vy = qkvy[0], qkvy[1], qkvy[2]  # make torchscript happy (cannot use tensor as tuple)
        qx = qx * self.scale
        qy = qy * self.scale
        attnx = (qx @ ky.transpose(-2, -1))
        attny = (qy @ kx.transpose(-2, -1))
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
            attnx = attnx + relative_position_bias.unsqueeze(0)
            attny = attny + relative_position_bias.unsqueeze(0)

        if maskx is not None:
            nW = maskx.shape[0]
            attnx = attnx.view(B_ // nW, nW, self.num_heads, N, N) + maskx.unsqueeze(1).unsqueeze(0)
            attnx = attnx.view(-1, self.num_heads, N, N)
            attnx = self.softmaxx(attnx)
        else:
            attnx = self.softmaxx(attnx)
        if masky is not None:
            nW = masky.shape[0]
            attny = attny.view(B_ // nW, nW, self.num_heads, N, N) + masky.unsqueeze(1).unsqueeze(0)
            attny = attny.view(-1, self.num_heads, N, N)
            attny = self.softmaxy(attny)
        else:
            attny = self.softmaxy(attny)

        attnx = self.attn_dropx(attnx)
        attny = self.attn_dropy(attny)
        x = (attnx @ vy).transpose(1, 2).reshape(B_, N, C)
        y = (attny @ vx).transpose(1, 2).reshape(B_, N, C)

        x = self.projx(x)
        x = self.proj_dropx(x)
        y = self.projy(y)
        y = self.proj_dropy(y)
        return x,y
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8 // reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, H, W, T):
        """
        x: B, H*W*T, C
        """
        # print('PatchMerging',x.shape) # 1,36,64
        # print('H,W,T',H,W,T) # 3,3,4
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
class LeFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate, d, h, w):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.gule1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.depthwise_separable_conv3d1 = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
                                                         nn.Conv3d(dim, dim, kernel_size=1,stride=1, padding=0, groups=1))
        self.depthwise_separable_conv3d2 = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
                                                         nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1))
        self.dim = dim
        self.d, self.h, self.w = d, h, w
    def forward(self, x):
        rx = x
        x = self.linear1(x)
        x = self.gule1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        rx = rx.view(rx.size(0), self.d, self.h, self.w, self.dim)
        rx = rx.permute(0, 4, 1, 2, 3).contiguous()
        rx = self.depthwise_separable_conv3d1(rx)
        rx = self.depthwise_separable_conv3d2(rx)
        rx = rx.permute(0, 1, 2, 3, 4).contiguous()
        rx = rx.view(rx.size(0), -1, self.dim)
        x = rx + x
        return x
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, rpe=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, d=10, h=12, w=14):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(self.shift_size) < min(
            self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}".format(self.shift_size,
                                                                                                     self.window_size)

        self.norm1x = norm_layer(dim)
        self.norm1y = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, rpe=rpe, attn_drop=attn_drop, proj_drop=drop)

        self.drop_pathx = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_pathy = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2x = norm_layer(dim)
        self.norm2y = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        #self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlpx = LeFeedForward(dim, mlp_hidden_dim, drop, d=d, h=h, w=w)
        self.mlpy = LeFeedForward(dim, mlp_hidden_dim, drop, d=d, h=h, w=w)

        self.H = None
        self.W = None
        self.T = None

    def forward(self, x, y, mask_matrixx, mask_matrixy):
        H, W, T = self.H, self.W, self.T
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"

        shortcutx = x
        x = self.norm1x(x)
        x = x.view(B, H, W, T, C)

        shortcuty = y
        y = self.norm1y(y)
        y = y.view(B, H, W, T, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        y = nnf.pad(y, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        _, Hp, Wp, Tp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
            attn_maskx = mask_matrixx
        else:
            shifted_x = x
            attn_maskx = None
        if min(self.shift_size) > 0:
            shifted_y = torch.roll(y, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
            attn_masky = mask_matrixy
        else:
            shifted_y = y
            attn_masky = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2],
                                   C)  # nW*B, window_size*window_size*window_size, C
        y_windows = window_partition(shifted_y, self.window_size)  # nW*B, window_size, window_size, window_size, C
        y_windows = y_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2],
                                   C)  # nW*B, window_size*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windowsx, attn_windowsy = self.attn(x_windows, y_windows, maskx=attn_maskx, masky=attn_masky)  # nW*B, window_size*window_size*window_size, C

        # merge windows
        attn_windowsx = attn_windowsx.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windowsx, self.window_size, Hp, Wp, Tp)  # B H' W' L' C
        attn_windowsy = attn_windowsy.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_y = window_reverse(attn_windowsy, self.window_size, Hp, Wp, Tp)  # B H' W' L' C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            x = shifted_x
        if min(self.shift_size) > 0:
            y = torch.roll(shifted_y, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            y = shifted_y

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :T, :].contiguous()
        if pad_r > 0 or pad_b > 0:
            y = y[:, :H, :W, :T, :].contiguous()

        x = x.view(B, H * W * T, C)
        y = y.view(B, H * W * T, C)

        # FFN
        x = shortcutx + self.drop_pathx(x)
        x = x + self.drop_pathx(self.mlpx(self.norm2x(x)))
        y = shortcuty + self.drop_pathy(y)
        y = y + self.drop_pathy(self.mlpy(self.norm2y(y)))

        return x, y
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 rpe=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 d=10,
                 h=12,
                 w=14):
        super().__init__()
        self.dim = dim
        self.norm_layer = nn.LayerNorm(self.dim)
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (
                    window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                rpe=rpe,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=norm_layer,
                d=d,
                h=h,
                w=w,
            )
            for i in range(depth)])

        # patch merging layer
        #self.downsample = None

    def forward(self, x, y, H, W, T):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        img_maskx = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)  # 1 Hp Wp 1
        img_masky = torch.zeros((1, Hp, Wp, Tp, 1), device=y.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        t_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cntx = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_maskx[:, h, w, t, :] = cntx
                    cntx += 1

        cnty = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_masky[:, h, w, t, :] = cnty
                    cnty += 1

        mask_windowsx = window_partition(img_maskx, self.window_size)  # nW, window_size, window_size, 1
        mask_windowsx = mask_windowsx.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_maskx = mask_windowsx.unsqueeze(1) - mask_windowsx.unsqueeze(2)
        attn_maskx = attn_maskx.masked_fill(attn_maskx != 0, float(-100.0)).masked_fill(attn_maskx == 0, float(0.0))

        mask_windowsy = window_partition(img_masky, self.window_size)  # nW, window_size, window_size, 1
        mask_windowsy = mask_windowsy.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_masky = mask_windowsy.unsqueeze(1) - mask_windowsy.unsqueeze(2)
        attn_masky = attn_masky.masked_fill(attn_masky != 0, float(-100.0)).masked_fill(attn_masky == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W, blk.T = H, W, T
            x, y = blk(x, y, attn_maskx, attn_masky)

        x = self.norm_layer(x)
        y = self.norm_layer(y)
        return x, H, W, T, x, H, W, T, y, H, W, T, y, H, W, T
class RSSwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 # depths=[2, 2, 6, 2],
                 depth = 2,
                 # num_heads=[3, 6, 12, 24],
                 num_head = 8,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 #ape=False,
                 #spe=False,
                 rpe=True,
                 patch_norm=True,
                 # out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 d=10,
                 h=12,
                 w=14):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        #self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer
        #self.ape = ape
        #self.spe = spe
        self.rpe = rpe
        self.patch_norm = patch_norm
        #self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        '''self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)'''

        self.pos_dropx = nn.Dropout(p=drop_rate)
        self.pos_dropy = nn.Dropout(p=drop_rate)

        # stochastic depth
        #dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]  # stochastic depth decay rule

        # build layer
        self.layer = BasicLayer(dim=int(embed_dim),
                                depth=2,
                                num_heads=8,
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                rpe=rpe,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=0,
                                norm_layer=norm_layer,
                                downsample=None,
                                use_checkpoint=use_checkpoint,
                                pat_merg_rf=pat_merg_rf,
                                d=d,
                                h=h,
                                w=w, )
        num_features = int(embed_dim)
        self.num_features = num_features

        # add a norm layer for each output
        '''for i_layer in out_indices:
            layer = norm_layer(num_features)
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)'''
        #self.layer.append(self.norm_layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, y):
        """Forward function."""
        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_dropx(x)
        y = y.flatten(2).transpose(1, 2)
        y = self.pos_dropy(y)

        x_out, H, W, T, x, Wh, Ww, Wt, y_out, H, W, T, y, Wh, Ww, Wt = self.layer(x, y, Wh, Ww, Wt)
        xout = x_out.view(-1, H, W, T, self.num_features).permute(0, 4, 1, 2, 3).contiguous()
        yout = y_out.view(-1, H, W, T, self.num_features).permute(0, 4, 1, 2, 3).contiguous()

        return xout, yout

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(RSSwinTransformer, self).train(mode)
        self._freeze_stages()

class Channel_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(Channel_attention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.MLP = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()
        max_pool = self.max_pool(x).view([b, c])
        avg_pool = self.avg_pool(x).view([b, c])

        max_pool = self.MLP(max_pool)
        avg_pool = self.MLP(avg_pool)

        out = max_pool + avg_pool
        out = self.sigmoid(out).view([b, c, 1, 1, 1])
        return out * x
class Spacial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spacial_attention, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv3d(in_channels=2,
                              out_channels=1,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding,
                              bias=False
                              )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        out = torch.cat([max_pool, avg_pool], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out * x
class dual_Spacial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(dual_Spacial_attention, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv3d(in_channels=2,
                              out_channels=1,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding,
                              bias=False
                              )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        out = torch.cat([max_pool, avg_pool], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        out = out * y
        out = x + out
        return out
class AB(nn.Module):
    # dual attention block
    def __init__(self):
        super(AB, self).__init__()

        self.Spacial_attention = Spacial_attention(kernel_size=7)

    def forward(self, x):

        x = self.Spacial_attention(x)

        return x
class DAB(nn.Module):
    # dual attention block
    def __init__(self):
        super(DAB, self).__init__()

        self.dual_Spacial_attention = dual_Spacial_attention(kernel_size=7)

    def forward(self, x, y):

        x = self.dual_Spacial_attention(x, y)
        y = self.dual_Spacial_attention(y, x)
        out = torch.cat([x, y], dim=1)

        return out

class AMFF(nn.Module):
    def __init__(self, channel):
        super(AMFF, self).__init__()

        self.ed1 = nn.Sequential(
            *[nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
              nn.InstanceNorm3d(channel),
              nn.LeakyReLU(inplace=True)])
        self.ed2 = nn.Sequential(
            *[nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
              nn.InstanceNorm3d(channel),
              nn.LeakyReLU(inplace=True)])
        self.ed3 = nn.Sequential(
            *[nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=3, dilation=3, bias=True),
              nn.InstanceNorm3d(channel),
              nn.LeakyReLU(inplace=True)])
        self.ed4 = nn.Sequential(
            *[nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=4, dilation=4, bias=True),
              nn.InstanceNorm3d(channel),
              nn.LeakyReLU(inplace=True)])
        self.channel_attention = Channel_attention(channel * 5, ratio=16)
        self.spacial_attention = Spacial_attention(kernel_size=7)
        self.e = nn.Sequential(
            *[nn.Conv3d(channel * 5, channel, kernel_size=1, bias=True),
              nn.InstanceNorm3d(channel),
              nn.LeakyReLU(inplace=True)])

    def forward(self, x):
        x1 = self.ed1(x)
        x1 = self.spacial_attention(x1)
        x2 = self.ed2(x)
        x2 = self.spacial_attention(x2)
        x3 = self.ed3(x)
        x3 = self.spacial_attention(x3)
        x4 = self.ed4(x)
        x4 = self.spacial_attention(x4)
        x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = self.channel_attention(x)
        x = self.e(x)
        return x

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class CorrelationMorph(nn.Module):
    def __init__(self):
        super(CorrelationMorph, self).__init__()

        self.eninput = self.encoder(1, 16)
        self.ec1 = self.encoder(16, 16)
        self.ec2 = self.encoder(16, 32, stride=2)
        self.ec3 = self.encoder(32, 32, kernel_size=3, stride=1, padding=1)
        self.ec4 = self.encoder(32, 64, stride=2)
        self.ec5 = self.encoder(64, 64, kernel_size=3, stride=1, padding=1)
        self.ec6 = self.encoder(64, 128, stride=2)
        self.ec7 = self.encoder(128, 128, kernel_size=3, stride=1, padding=1)
        self.ec8 = self.encoder(128, 256, stride=2)
        self.ec9 = self.encoder(256, 256, kernel_size=3, stride=1, padding=1)
        self.ec10 = self.encoder(256, 512, stride=2)
        self.ec11 = self.encoder(512, 512, kernel_size=3, stride=1, padding=1)
        self.e2 = self.encoder(64 * 2, 64, kernel_size=3, stride=1, padding=1)

        self.rstransformer = RSSwinTransformer(patch_size=4,
                                             in_chans=64,
                                             embed_dim=64,
                                             depth=2,
                                             num_head=8,
                                             window_size=(5, 5, 5),
                                             mlp_ratio=4,
                                             qkv_bias=False,
                                             drop_rate=0,
                                             drop_path_rate=0.3,
                                             rpe=True,
                                             patch_norm=True,
                                             use_checkpoint=False,
                                             pat_merg_rf=4,
                                             d=40,
                                             h=48,
                                             w=56,
                                             )

        self.transformer3 = SwinTransformer(patch_size=4,
                                             in_chans=128,
                                             embed_dim=128,
                                             depth=2,
                                             num_head=8,
                                             window_size=(5, 5, 5),
                                             mlp_ratio=4,
                                             qkv_bias=False,
                                             drop_rate=0,
                                             drop_path_rate=0.3,
                                             rpe=True,
                                             patch_norm=True,
                                             use_checkpoint=False,
                                             pat_merg_rf=4,
                                             d=20,
                                             h=24,
                                             w=28,
                                             )

        self.transformer4 = SwinTransformer(patch_size=4,
                                            in_chans=256,
                                            embed_dim=256,
                                            depth=2,
                                            num_head=8,
                                            window_size=(5, 5, 5),
                                            mlp_ratio=4,
                                            qkv_bias=False,
                                            drop_rate=0,
                                            drop_path_rate=0.3,
                                            rpe=True,
                                            patch_norm=True,
                                            use_checkpoint=False,
                                            pat_merg_rf=4,
                                            d=10,
                                            h=12,
                                            w=14,
                                            )

        self.amff1 = AMFF(512)
        self.amff2 = AMFF(512)
        self.amff3 = AMFF(512)

        self.up0 = DecoderBlock(512, 256, skip_channels=256, use_batchnorm=False)
        self.up1 = DecoderBlock(256, 128, skip_channels=128, use_batchnorm=False)
        self.up2 = DecoderBlock(128, 64, skip_channels=128, use_batchnorm=False)
        self.up3 = DecoderBlock(64, 32, skip_channels=64, use_batchnorm=False)
        self.up4 = DecoderBlock(32, 16, skip_channels=32, use_batchnorm=False)

        self.dab0 = DAB()
        self.dab1 = DAB()
        self.dab2 = DAB()
        self.dab3 = AB()
        self.dab4 = AB()

        self.reg_head = RegistrationHead(
            in_channels=16,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer((160,192,224))

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True))
        return layer

    def forward(self, source):
        x = source[:, 0:1, :, :]
        y = source[:, 1:2, :, :]
        ex0 = self.eninput(x)
        ex0 = self.ec1(ex0)

        ex1 = self.ec2(ex0)
        ex1 = self.ec3(ex1)

        ex2 = self.ec4(ex1)
        ex2 = self.ec5(ex2)

        ey0 = self.eninput(y)
        ey0 = self.ec1(ey0)

        ey1 = self.ec2(ey0)
        ey1 = self.ec3(ey1)

        ey2 = self.ec4(ey1)
        ey2 = self.ec5(ey2)

        tx, ty = self.rstransformer(ex2, ey2)
        e2 = torch.cat((tx, ty), 1)
        e2 = self.e2(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        te3 = self.transformer3(e3)

        e4 = self.ec8(te3)
        e4 = self.ec9(e4)

        te4 = self.transformer4(e4)

        e5 = self.ec10(te4)
        e6 = self.ec11(e5)

        e6 = self.amff1(e6)
        e6 = self.amff2(e6)
        e6 = self.amff3(e6)

        de4 = self.dab4(e4)
        de3 = self.dab3(e3)
        de2 = self.dab2(ex2, ey2)
        de1 = self.dab1(ex1, ey1)
        de0 = self.dab0(ex0, ey0)

        d0 = self.up0(e6, de4)
        d1 = self.up1(d0, de3)
        d2 = self.up2(d1, de2)
        d3 = self.up3(d2, de1)
        d4 = self.up4(d3, de0)

        flow = self.reg_head(d4)
        out = self.spatial_trans(x, flow)
        return out, flow
