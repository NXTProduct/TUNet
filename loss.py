import torch
from auraloss.freq import STFTLoss, MultiResolutionSTFTLoss, apply_reduction


class STFTLossDDP(STFTLoss):

    def forward(self, x, y):
        # compute the magnitude and phase spectra of input and target
        self.window = self.window.to(x.device)
        x_mag, x_phs = self.stft(x.view(-1, x.size(-1)))
        y_mag, y_phs = self.stft(y.view(-1, y.size(-1)))

        # apply relevant transforms
        if self.scale is not None:
            x_mag = torch.matmul(self.fb.to(x_mag.device), x_mag)
            y_mag = torch.matmul(self.fb.to(y_mag.device), y_mag)

        # normalize scales
        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag ** 2).sum([-2, -1]))
            y_mag = y_mag * alpha.unsqueeze(-1)

        # compute loss terms
        sc_loss = self.spectralconv(x_mag, y_mag) if self.w_sc else 0.0
        mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0
        lin_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0.0

        # combine loss terms
        loss = (self.w_sc * sc_loss) + (self.w_log_mag * mag_loss) + (self.w_lin_mag * lin_loss)
        loss = apply_reduction(loss, reduction=self.reduction)

        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sc_loss, mag_loss


class MRSTFTLossDDP(MultiResolutionSTFTLoss):
    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window",
                 w_sc=1.0,
                 w_log_mag=1.0,
                 w_lin_mag=0.0,
                 w_phs=0.0,
                 sample_rate=None,
                 scale=None,
                 n_bins=None,
                 scale_invariance=False,
                 **kwargs):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLossDDP(fs,
                                             ss,
                                             wl,
                                             window,
                                             w_sc,
                                             w_log_mag,
                                             w_lin_mag,
                                             w_phs,
                                             sample_rate,
                                             scale,
                                             n_bins,
                                             scale_invariance,
                                             **kwargs)]
