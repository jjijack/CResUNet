import torch
import numpy as np
import numpy.ma as ma


class AdaptiveNormalizer:

    def __init__(self, output_dim, seasonality=''):
        self.output_dim = output_dim
        self.seasonality = seasonality
        self.min_max = []

    def norm(self, x):

        batch_size, seq_len, height, width, d_dim = x.shape
        x = np.maximum(x, 0)
        tt = []

        for i in range(seq_len):
            s = x[:, i, :, :, :]
            mask = s == 0
            mx = ma.array(s, mask=mask)
            mean = np.mean(mx)
            idx1 = np.where(s == 0)
            s[idx1] = mean
            tt.append(s)
        x = torch.stack(tt, dim=1)
        out = []

        for d in range(d_dim):
            a = x[..., d]
            aa = a.contiguous().view(a.size(0), -1)

            min_a = aa.min(dim=1, keepdim=True)[0]
            max_a = aa.max(dim=1, keepdim=True)[0]

            aa -= min_a
            aa /= (max_a - min_a)

            aa = aa.view(batch_size, seq_len, height, width)
            out.append(aa)
            self.min_max.append((min_a, max_a))

        out = torch.stack(out, dim=-1)
        return out

    def inv_norm(self, x, device):

        x = x.permute(0, 1, 3, 4, 2)
        batch_size, seq_len, height, width, d_dim = x.shape
        x = x.contiguous().view(x.size(0), -1)
        min_, max_ = self.min_max[self.output_dim]
        x = x * (max_.to(device) - min_.to(device)) + min_.to(device)
        x = x.view(batch_size, seq_len, height, width, d_dim)
        x = x.permute(0, 1, 4, 2, 3)
        return x
