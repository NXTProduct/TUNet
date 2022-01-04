import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from performer_pytorch import Performer
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.audio.si_sdr import SI_SDR

from config import CONFIG
from dataset import CustomDataset
from loss import MRSTFTLossDDP


class TFiLM(nn.Module):
    def __init__(self, block_size, input_dim, **kwargs):
        super(TFiLM, self).__init__(**kwargs)
        self.block_size = block_size
        self.max_pool = nn.MaxPool1d(kernel_size=self.block_size)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True)

    def make_normalizer(self, x_in):
        """ Pools to downsample along 'temporal' dimension and then
            runs LSTM to generate normalization weights.
        """
        x_in_down = self.max_pool(x_in).permute([0, 2, 1])
        x_rnn, _ = self.lstm(x_in_down)
        return x_rnn.permute([0, 2, 1])

    def apply_normalizer(self, x_in, x_norm):
        """
        Applies normalization weights by multiplying them into their respective blocks.
        """
        # channel first
        n_blocks = x_in.shape[2] // self.block_size
        n_filters = x_in.shape[1]

        # reshape input into blocks
        x_norm = torch.reshape(x_norm, shape=(-1, n_filters, n_blocks, 1))
        x_in = torch.reshape(x_in, shape=(-1, n_filters, n_blocks, self.block_size))

        # multiply
        x_out = x_norm * x_in

        # return to original shape
        x_out = torch.reshape(x_out, shape=(-1, n_filters, n_blocks * self.block_size))

        return x_out

    def forward(self, x):
        assert len(x.shape) == 3, 'Input should be tensor with dimension \
                                   (batch_size, steps, num_features).'
        assert x.shape[2] % self.block_size == 0, 'Number of steps must be a \
                                                   multiple of the block size.'

        x_norm = self.make_normalizer(x)
        x = self.apply_normalizer(x, x_norm)
        return x


class Encoder(nn.Module):
    def __init__(self, max_len, kernel_sizes, strides, out_channels, tfilm, n_blocks):
        super(Encoder, self).__init__()
        self.tfilm = tfilm

        n_layers = len(strides)
        paddings = [(kernel_sizes[i] - strides[i]) // 2 for i in range(n_layers)]

        if self.tfilm:
            b_size = max_len // (n_blocks * strides[0])
            self.tfilm_d = TFiLM(block_size=b_size, input_dim=out_channels[0])
            b_size //= strides[1]
            self.tfilm_d1 = TFiLM(block_size=b_size, input_dim=out_channels[1])

        self.downconv = nn.Conv1d(in_channels=1, out_channels=out_channels[0], kernel_size=kernel_sizes[0],
                                  stride=strides[0], padding=paddings[0], padding_mode='replicate')
        self.downconv1 = nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1],
                                   kernel_size=kernel_sizes[1],
                                   stride=strides[1], padding=paddings[1], padding_mode='replicate')
        self.downconv2 = nn.Conv1d(in_channels=out_channels[1], out_channels=out_channels[2],
                                   kernel_size=kernel_sizes[2],
                                   stride=strides[2], padding=paddings[2], padding_mode='replicate')
        self.out_len = max_len // (strides[0] * strides[1] * strides[2])

    def forward(self, x):
        x1 = F.leaky_relu(self.downconv(x), 0.2)  # 2048
        if self.tfilm:
            x1 = self.tfilm_d(x1)
        x2 = F.leaky_relu(self.downconv1(x1), 0.2)  # 1024
        if self.tfilm:
            x2 = self.tfilm_d1(x2)
        x3 = F.leaky_relu(self.downconv2(x2), 0.2)  # 512
        return [x1, x2, x3]


class Decoder(nn.Module):
    def __init__(self, in_len, kernel_sizes, strides, out_channels, tfilm, n_blocks):
        super(Decoder, self).__init__()
        self.tfilm = tfilm
        n_layers = len(strides)
        paddings = [(kernel_sizes[i] - strides[i]) // 2 for i in range(n_layers)]

        if self.tfilm:
            in_len *= strides[2]
            self.tfilm_u1 = TFiLM(block_size=in_len // n_blocks, input_dim=out_channels[1])
            in_len *= strides[1]
            self.tfilm_u = TFiLM(block_size=in_len // n_blocks, input_dim=out_channels[0])

        self.convt3 = nn.ConvTranspose1d(in_channels=out_channels[2], out_channels=out_channels[1], stride=strides[2],
                                         kernel_size=kernel_sizes[2], padding=paddings[2])
        self.convt2 = nn.ConvTranspose1d(in_channels=out_channels[1], out_channels=out_channels[0], stride=strides[1],
                                         kernel_size=kernel_sizes[1], padding=paddings[1])
        self.convt1 = nn.ConvTranspose1d(in_channels=out_channels[0], out_channels=1, stride=strides[0],
                                         kernel_size=kernel_sizes[0], padding=paddings[0])
        self.dropout = nn.Dropout(0.0)

    def forward(self, x_list):
        x, x1, x2, bottle_neck = x_list
        x_dec = self.dropout(F.leaky_relu(self.convt3(bottle_neck), 0.2))
        if self.tfilm:
            x_dec = self.tfilm_u1(x_dec)
        x_dec = x2 + x_dec

        x_dec = self.dropout(F.leaky_relu(self.convt2(x_dec), 0.2))
        if self.tfilm:
            x_dec = self.tfilm_u(x_dec)
        x_dec = x1 + x_dec
        x_dec = x + torch.tanh(self.convt1(x_dec))
        return x_dec


class BaseModel(pl.LightningModule):
    def __init__(self, train_dataset=None, val_dataset=None):
        super(BaseModel, self).__init__()
        self.hparams['loss_type'] = CONFIG.TRAIN.loss_type
        self.hparams['max_len'] = CONFIG.DATA.window_size
        self.learning_rate = CONFIG.TRAIN.lr
        self.hparams['batch_size'] = CONFIG.TRAIN.batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_sisdr = SI_SDR()
        self.valid_sisdr = SI_SDR()
        self.time_loss = nn.MSELoss()
        self.freq_loss = MRSTFTLossDDP(n_bins=64, sample_rate=CONFIG.DATA.sr, device="cpu", scale='mel')

    def forward(self, x):
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=False, batch_size=self.hparams.batch_size,
                          num_workers=CONFIG.TRAIN.workers, collate_fn=CustomDataset.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.hparams.batch_size,
                          num_workers=CONFIG.TRAIN.workers, collate_fn=CustomDataset.collate_fn)

    def forward_loss(self, x, y):
        if self.hparams.loss_type == 1:
            loss = self.time_loss(x, y)
        else:
            loss = self.freq_loss(x, y) + self.time_loss(x, y) * CONFIG.TRAIN.mse_weight
        return loss

    def training_step(self, batch, batch_idx):
        x_in, y = batch
        x = self(x_in)
        loss = self.forward_loss(x, y)
        self.train_sisdr(x, y)
        self.log('train_loss', loss, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x_in, y = val_batch
        x = self(x_in)
        loss = self.forward_loss(x, y)
        self.valid_sisdr(x, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)
        self.log('val_sisdr', self.valid_sisdr, on_step=False, on_epoch=True, logger=True, prog_bar=True,
                 sync_dist=True)

        if batch_idx == 0:
            i = torch.randint(0, x_in.shape[0], (1,)).item()
            lr, hr, recon = torch.squeeze(x_in[i]), torch.squeeze(y[i]), torch.squeeze(x[i])
            self.trainer.logger.log_spectrogram(hr, lr, recon, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=CONFIG.TRAIN.patience, factor=CONFIG.TRAIN.factor,
                                         verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]


class TUNet(BaseModel):
    def __init__(self, train_dataset, val_dataset):
        super(TUNet, self).__init__(train_dataset, val_dataset)
        self.hparams['out_channels'] = CONFIG.MODEL.out_channels
        self.hparams['kernel_sizes'] = CONFIG.MODEL.kernel_sizes
        self.hparams['bottleneck_type'] = CONFIG.MODEL.bottleneck_type
        self.hparams['strides'] = CONFIG.MODEL.strides
        self.hparams['tfilm'] = CONFIG.MODEL.tfilm
        self.hparams['n_blocks'] = CONFIG.MODEL.n_blocks
        self.encoder = Encoder(max_len=self.hparams.max_len,
                               kernel_sizes=self.hparams.kernel_sizes,
                               strides=self.hparams.strides,
                               out_channels=self.hparams.out_channels,
                               tfilm=self.hparams.tfilm,
                               n_blocks=self.hparams.n_blocks)
        bottleneck_size = self.hparams.max_len // np.array(self.hparams.strides).prod()

        if self.hparams.bottleneck_type == 'performer':
            self.bottleneck = Performer(dim=self.hparams.out_channels[2], depth=CONFIG.MODEL.TRANSFORMER.depth,
                                        heads=CONFIG.MODEL.TRANSFORMER.heads, causal=False,
                                        dim_head=CONFIG.MODEL.TRANSFORMER.dim_head, local_window_size=bottleneck_size)
        elif self.hparams.bottleneck_type == 'lstm':
            self.bottleneck = nn.LSTM(input_size=self.hparams.out_channels[2], hidden_size=self.hparams.out_channels[2],
                                      num_layers=CONFIG.MODEL.TRANSFORMER.depth, batch_first=True)

        self.decoder = Decoder(in_len=self.encoder.out_len,
                               kernel_sizes=self.hparams.kernel_sizes,
                               strides=self.hparams.strides,
                               out_channels=self.hparams.out_channels,
                               tfilm=self.hparams.tfilm,
                               n_blocks=self.hparams.n_blocks)

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        if self.hparams.bottleneck_type is not None:
            x3 = x3.permute([0, 2, 1])
            if self.hparams.bottleneck_type == 'performer':
                bottle_neck = self.bottleneck(x3)
            elif self.hparams.bottleneck_type == 'lstm':
                bottle_neck = self.bottleneck(x3)[0].clone()
            else:
                bottle_neck = self.bottleneck(inputs_embeds=x3)[0]
            bottle_neck += x3
            bottle_neck = bottle_neck.permute([0, 2, 1])
        else:
            bottle_neck = x3
        x_dec = self.decoder([x, x1, x2, bottle_neck])
        return x_dec
