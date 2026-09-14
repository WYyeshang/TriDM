import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.DWT_Decomposition import Decomposition
import math
from torch.utils.checkpoint import checkpoint

@torch.jit.script
def fast_b_splines(x: torch.Tensor, grid: torch.Tensor, spline_order: int):
    
    x = x.unsqueeze(-1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for k in range(1, spline_order + 1):
        bases = (
            (x - grid[:, : -(k + 1)])
            / (grid[:, k:-1] - grid[:, : -(k + 1)])
            * bases[:, :, :-1]
        ) + (
            (grid[:, k + 1 :] - x)
            / (grid[:, k + 1 :] - grid[:, 1:(-k)])
            * bases[:, :, 1:]
        )
    return bases.contiguous()


class moving_avg(nn.Module):
    
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x) 
        res = x - moving_mean            
        return res, moving_mean


class TrendBlock(nn.Module):
    
    def __init__(self, seq_len, pred_len):
        super(TrendBlock, self).__init__()
        self.pred_len = pred_len
                
        self.fc1 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc2 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc3 = nn.Linear(pred_len // 2, pred_len)

    def forward(self, x):
        
        B, L, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, L) 
        
        x = self.fc1(x) 
        x = x.unsqueeze(1) 
        x = self.avgpool1(x).squeeze(1)
        x = self.ln1(x)

        x = self.fc2(x)
        x = x.unsqueeze(1)
        x = self.avgpool2(x).squeeze(1)  
        x = self.ln2(x)

        x = self.fc3(x) 
        
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1) 
        return x


class KANLinear(torch.nn.Module):
    
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        enable_checkpoint=True,
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size      
        self.spline_order = spline_order 
        self.enable_checkpoint = enable_checkpoint

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
       
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        
        assert x.dim() == 2 and x.size(1) == self.in_features
        return fast_b_splines(x, self.grid, self.spline_order)

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(
            A, B
        ).solution
        result = solution.permute(
            2, 0, 1
        )

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        if self.enable_checkpoint and x.requires_grad:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        spline_bases = self.b_splines(x) 
        
        spline_bases_flat = spline_bases.view(x.size(0), -1)
        
        weight_flat = self.scaled_spline_weight.view(self.out_features, -1)
        
        spline_output = F.linear(spline_bases_flat, weight_flat)
        output = base_output + spline_output
        
        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        enable_checkpoint=True,
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    enable_checkpoint=enable_checkpoint,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class Splitting(nn.Module):
    
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, ::2]

    def odd(self, x):
        return x[:, :, 1::2]

    def forward(self, x):
        return self.even(x), self.odd(x)


class KANCausalConvBlock(nn.Module):
    
    def __init__(self, d_model, d_hidden, kernel_size=5, dropout=0.0):
        super(KANCausalConvBlock, self).__init__()
        self.pad = nn.ReplicationPad1d((kernel_size - 1, kernel_size - 1))
        self.conv1 = nn.Conv1d(d_model, d_hidden, kernel_size=kernel_size)
        self.norm1 = nn.BatchNorm1d(d_hidden) 
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.mixing = nn.Linear(d_hidden, d_hidden)
        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(d_hidden, d_model, kernel_size=kernel_size)
        self.act2 = nn.Tanh()

    def forward(self, x):
        
        x = self.pad(x)
        x = self.conv1(x) 
        x = self.norm1(x)
        x = self.act1(x)
        
        x = x.transpose(1, 2)
        x = self.mixing(x)
        x = x.transpose(1, 2)
        
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.act2(x)
        return x 


class IntraSeriesInteractionBlock(nn.Module):
    
    def __init__(self, d_model, d_hidden, kernel_size=5, dropout=0.0):
        super(IntraSeriesInteractionBlock, self).__init__()
        self.splitting = Splitting()
        
        self.modules_even = KANCausalConvBlock(d_model, d_hidden, kernel_size, dropout)
        self.modules_odd = KANCausalConvBlock(d_model, d_hidden, kernel_size, dropout)
        self.interactor_even = KANCausalConvBlock(d_model, d_hidden, kernel_size, dropout)
        self.interactor_odd = KANCausalConvBlock(d_model, d_hidden, kernel_size, dropout)

    def forward(self, x):
       
        x_even, x_odd = self.splitting(x)
        
        x_even_temp = x_even.mul(torch.exp(self.modules_even(x_odd)))
        x_odd_temp = x_odd.mul(torch.exp(self.modules_odd(x_even)))

        x_even_update = x_even_temp + self.interactor_even(x_odd_temp)
        x_odd_update = x_odd_temp - self.interactor_odd(x_even_temp)

        return x_even_update, x_odd_update


class IntraSeriesBlock(nn.Module):
    
    def __init__(self, d_model, d_hidden=None, current_level=3, kernel_size=5, dropout=0.0):
        super(IntraSeriesBlock, self).__init__()
        self.current_level = current_level
        if d_hidden is None:
            d_hidden = d_model * 4 
            
        self.working_block = IntraSeriesInteractionBlock(d_model, d_hidden, kernel_size, dropout)

        if current_level != 0:
           
            self.Tree_odd = IntraSeriesBlock(d_model, d_hidden, current_level-1, kernel_size, dropout)
            self.Tree_even = IntraSeriesBlock(d_model, d_hidden, current_level-1, kernel_size, dropout)

    def forward(self, x):
        
        odd_flag = False
        if x.shape[2] % 2 == 1: 
            odd_flag = True
            x = F.pad(x, (0, 1), mode='replicate')
            
        x_even_update, x_odd_update = self.working_block(x)
        
        if self.current_level == 0:
            res = self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            
            res = self.zip_up_the_pants(self.Tree_even(x_even_update), self.Tree_odd(x_odd_update))
            
        if odd_flag:
            res = res[:, :, :-1]
        return res

    def zip_up_the_pants(self, even, odd):
       
        B, C, L_even = even.shape
        _, _, L_odd = odd.shape
        L = L_even + L_odd
        
        res = torch.empty((B, C, L), device=even.device, dtype=even.dtype)
        res[:, :, 0::2] = even
        res[:, :, 1::2] = odd
        return res


class KANEncoderLayer(nn.Module):
    
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", grid_size=5, spline_order=3, enable_checkpoint=True):
        super(KANEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        
        self.ffn = KAN([d_model, d_ff, d_model], grid_size=grid_size, spline_order=spline_order, base_activation=nn.SiLU, enable_checkpoint=enable_checkpoint)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        y = self.ffn(x)
        
        x = x + self.dropout(y)
        x = self.norm2(x)
        
        return x, attn


class KANTokenMixer(nn.Module):
    
    def __init__(self, input_seq, pred_seq, factor=3, d_model=None, dropout=0.1, grid_size=5, enable_checkpoint=True):
        super(KANTokenMixer, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = pred_seq
        self.factor = factor
       
        self.layer1 = KANLinear(input_seq, pred_seq * factor, grid_size=grid_size, scale_noise=0.01, enable_checkpoint=enable_checkpoint)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = KANLinear(pred_seq * factor, pred_seq, grid_size=grid_size, scale_noise=0.01, enable_checkpoint=enable_checkpoint)

    def forward(self, x):

        x = self.layer1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class KANMixer(nn.Module):
   
    def __init__(self, input_seq, out_seq, channel, d_model, dropout, tfactor, dfactor, grid_size=5, enable_checkpoint=True):
        super(KANMixer, self).__init__()
        self.norm1 = nn.BatchNorm2d(channel)
        self.norm2 = nn.BatchNorm2d(channel)
        self.dropoutLayer = nn.Dropout(dropout)
        
        self.tMixer = KANTokenMixer(input_seq, out_seq, factor=tfactor, d_model=d_model, dropout=dropout, grid_size=grid_size, enable_checkpoint=enable_checkpoint)
        
        self.embeddingMixer = nn.Sequential(
            KANLinear(d_model, d_model * dfactor, grid_size=grid_size, enable_checkpoint=enable_checkpoint),
            nn.GELU(),
            nn.Dropout(dropout),
            KANLinear(d_model * dfactor, d_model, grid_size=grid_size, enable_checkpoint=enable_checkpoint)
        )

    def forward(self, x):
       
        res = x
        x = self.norm1(x)
        
        y = x.permute(0, 3, 1, 2)
        y = self.tMixer(y)
        y = y.permute(0, 2, 3, 1) 
        x = res + self.dropoutLayer(y)
        
        res = x
        x = self.norm2(x)
        x = self.embeddingMixer(x)
        x = res + self.dropoutLayer(x)
        return x

class ResolutionBranch(nn.Module):
    
    def __init__(self, input_seq, pred_seq, channel, d_model, dropout, tfactor, dfactor, patch_len, patch_stride, grid_size=5, enable_checkpoint=True):
        super(ResolutionBranch, self).__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_num = int((input_seq - patch_len) / patch_stride + 2)
        
        self.patch_norm = nn.BatchNorm2d(channel)
        self.patch_embedding_layer = nn.Linear(patch_len, d_model)
       
        self.mixer = KANMixer(self.patch_num, self.patch_num, channel, d_model, dropout, tfactor, dfactor, grid_size, enable_checkpoint=enable_checkpoint)
        
        self.norm = nn.BatchNorm2d(channel)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(self.patch_num * d_model, pred_seq)
        )

    def do_patching(self, x):
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        x_patch = x_new.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return x_patch

    def forward(self, x):
        x_patch = self.do_patching(x)
        x_patch = self.patch_norm(x_patch)
        x_emb = self.patch_embedding_layer(x_patch)
        
        out = self.mixer(x_emb)
        out = self.norm(out)
        
        out = self.head(out)
        return out

class WaveletMixingBlock(nn.Module):
    
    def __init__(self, input_length, pred_length, channel, d_model, dropout=0.1, 
                 wavelet_name='db1', level=2, tfactor=3, dfactor=3, patch_len=16, patch_stride=8):
        super(WaveletMixingBlock, self).__init__()
        self.pred_length = pred_length
       
        self.decomposition = Decomposition(input_length=input_length, pred_length=pred_length, 
                                           wavelet_name=wavelet_name, level=level, 
                                           batch_size=1, channel=channel, d_model=d_model,
                                           tfactor=tfactor, dfactor=dfactor, device=torch.device('cpu'),
                                           no_decomposition=False, use_amp=False)
                                           
        input_w_dim = self.decomposition.input_w_dim
        pred_w_dim = self.decomposition.pred_w_dim
        
        self.resolution_branches = nn.ModuleList()
        for i in range(len(input_w_dim)):
            g_size = 5 if i == 0 else 3
            self.resolution_branches.append(
                ResolutionBranch(input_w_dim[i], pred_w_dim[i], channel, d_model, dropout, 
                             tfactor, dfactor, patch_len, patch_stride, grid_size=g_size, enable_checkpoint=True)
            )

    def forward(self, x):
       
        x = x.permute(0, 2, 1)
       
        xA, xD = self.decomposition.transform(x)
        
        yA = self.resolution_branches[0](xA)
        
        yD = []
        for i in range(len(xD)):
            yD.append(self.resolution_branches[i+1](xD[i]))
        
        y = self.decomposition.inv_transform(yA, yD)
        
        y = y.permute(0, 2, 1)
        return y[:, -self.pred_length:, :]


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.kan_grid_size = getattr(configs, 'grid_size', 3) 
    
        
        kernel_size = 25
        self.decomp = series_decomp(kernel_size)
        
        self.trend_block = TrendBlock(self.seq_len, self.pred_len)
        
        self.wavelet_block = WaveletMixingBlock(
            input_length=self.seq_len, 
            pred_length=self.pred_len,
            channel=configs.enc_in,
            d_model=configs.d_model,
            dropout=configs.dropout,
            patch_len=getattr(configs, 'patch_len', 16),
            patch_stride=getattr(configs, 'patch_stride', 8)
        )
      
        self.gate_wavelet = nn.Parameter(torch.ones(1, 1, configs.enc_in) * 0.1)
                
        self.intra_series_block = IntraSeriesBlock(d_model=1, d_hidden=32, current_level=1, kernel_size=5, dropout=configs.dropout)
        
        self.gate_intra = nn.Parameter(torch.ones(1, configs.enc_in, 1) * 0.1) 
        
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        self.inter_series_block = Encoder(
            [
                KANEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      ), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,                    
                    grid_size=self.kan_grid_size,
                    spline_order=3,
                    enable_checkpoint=True
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            
            self.projection = KAN([configs.d_model, configs.d_model // 2, configs.pred_len], grid_size=self.kan_grid_size, grid_range=[-3, 3], enable_checkpoint=True)        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
       
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        seasonal_init, trend_init = self.decomp(x_enc)
        
        trend_output = self.trend_block(trend_init) 
        
        wavelet_output = self.wavelet_block(seasonal_init) 

        B, L, N = seasonal_init.shape
        x_intra_in = seasonal_init.permute(0, 2, 1) 
        x_intra_in = x_intra_in.reshape(B * N, 1, L)
        
        x_intra_out = self.intra_series_block(x_intra_in)
        x_intra_out = x_intra_out.reshape(B, N, L).permute(0, 2, 1) 
        
        gate = self.gate_intra.permute(0, 2, 1) 
        seasonal_part = seasonal_init + gate * x_intra_out
        
        enc_out = self.enc_embedding(seasonal_part, x_mark_enc) 
        enc_out, attns = self.inter_series_block(enc_out, attn_mask=None)
        
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        
        final_out = dec_out + trend_output + self.gate_wavelet * wavelet_output
        
        final_out = final_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        final_out = final_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return final_out
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]        
        return None
