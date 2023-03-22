import torch
import torch.nn as nn
import torchvision
from models.resnet import pretrained_resnet101
import torch.nn.functional as F

class NonLocalBlock(nn.Module):
    def __init__(self, dim_in=2048, dim_out=2048, dim_inner=256):
        super(NonLocalBlock, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = dim_inner
        self.dim_out = dim_out

        self.theta = nn.Linear(dim_in, dim_inner)
        self.phi = nn.Linear(dim_in, dim_inner)
        self.g = nn.Linear(dim_in, dim_inner)

        self.out = nn.Linear(dim_inner, dim_out)
        self.bn = nn.BatchNorm1d(dim_out)
        self.alpha=nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        residual = x

        batch_size,seq = x.shape[:2]
        x=x.view(batch_size*seq,-1)

        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)

        theta,phi,g=theta.view(batch_size,seq,-1).transpose(1,2).contiguous(),phi.view(batch_size,seq,-1).transpose(1,2).contiguous(),g.view(batch_size,seq,-1).transpose(1,2).contiguous()

        theta_phi = torch.bmm(theta.transpose(1, 2), phi)  # (8, 16, 784) * (8, 1024, 784) => (8, 784, 784)

        theta_phi_sc = theta_phi * (self.dim_inner ** -.5)
        p = F.softmax(theta_phi_sc, dim=-1)

        t = torch.bmm(g, p.transpose(1, 2))
        t = t.transpose(1,2).contiguous().view(batch_size*seq,-1)

        out = self.out(t)
        out = self.bn(out)
        out=out.view(batch_size,seq,-1)

        out = self.alpha*out + residual
        return out

class VisualStream(nn.Module):
    def __init__(self,
                 snippet_duration,
                 sample_size,
                 n_classes,
                 seq_len,
                 pretrained_resnet101_path):
        super(VisualStream, self).__init__()
        self.snippet_duration = snippet_duration
        self.sample_size = sample_size
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.ft_begin_index = 5
        self.pretrained_resnet101_path = pretrained_resnet101_path
        self._init_norm_val()
        self._init_hyperparameters()
        self._init_encoder()
        self._init_nonlocal()

    def _init_norm_val(self):
        self.NORM_VALUE = 255.0
        self.MEAN = 100.0 / self.NORM_VALUE

    def _init_encoder(self):
        resnet, _ = pretrained_resnet101(snippet_duration=self.snippet_duration,
                                         sample_size=self.sample_size,
                                         n_classes=self.n_classes,
                                         ft_begin_index=self.ft_begin_index,
                                         pretrained_resnet101_path=self.pretrained_resnet101_path)
        children = list(resnet.children())
        self.resnet = nn.Sequential(*children[:-1])  # delete the last fc
        for param in self.resnet.parameters():
            param.requires_grad = False

    def _init_hyperparameters(self):
        self.hp = {
            'nc': 2048,
            'k': 512,
            'm': 16,
            'hw': 4
        }

    def _init_module(self, m):
        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _init_nonlocal(self):
        self.nl=NonLocalBlock()
        self.fc = nn.Linear(2048, self.n_classes)


    def forward(self, input: torch.Tensor):
        # input.shape=[batch, seq_len,  3, 16, 112, 112]
        input.div_(self.NORM_VALUE).sub_(self.MEAN)
        batch, seq_len, nc, snippet_duration, sample_size, _ = input.size()
        input = input.view(batch*seq_len, nc, snippet_duration, sample_size, sample_size)
        with torch.no_grad():
            output = self.resnet(input).squeeze()
        output=output.view(batch,seq_len,-1)
        output=self.nl(output)
        output=output.transpose(1,2).contiguous()
        output = torch.mean(output, dim=2)

        output=self.fc(output)
        return output,None
