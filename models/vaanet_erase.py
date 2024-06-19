import torch
import torch.nn as nn
import torchvision
from models.resnet import pretrained_resnet101
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value), attention


class MultiHeadAttentionOp(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttentionOp, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y, attn = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y, attn

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )

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

        out = out + self.alpha*residual
        return out


class VAANetErase(nn.Module):
    def __init__(self,
                 snippet_duration,
                 sample_size,
                 n_classes,
                 seq_len,
                 pretrained_resnet101_path,
                 audio_embed_size=768,
                 audio_time=100,
                 audio_n_segments=10,):
        super(VAANetErase, self).__init__()
        self.snippet_duration = snippet_duration
        self.sample_size = sample_size
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.ft_begin_index = 5
        
        self.audio_n_segments = audio_n_segments
        self.audio_embed_size = audio_embed_size
        
        a_resnet = torchvision.models.resnet18(pretrained=True)
        a_conv1 = nn.Conv2d(1, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias=False)
        a_avgpool = nn.AvgPool2d(kernel_size=[4, 8])
        a_modules = [a_conv1] + list(a_resnet.children())[1:-2] + [a_avgpool]
        self.a_resnet = nn.Sequential(*a_modules)
        self.a_fc = nn.Sequential(
            nn.Linear(a_resnet.fc.in_features, self.audio_embed_size),
            nn.BatchNorm1d(self.audio_embed_size),
            nn.Tanh()
        )
        
        
        self.pretrained_resnet101_path = pretrained_resnet101_path
        self.drop = nn.Dropout(p=.2)
        self._init_norm_val()
        self._init_hyperparameters()
        self._init_encoder()
        self._init_nonlocal()
        self._init_attention_subnets()

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

    def _init_attention_subnets(self):

        self.ta_net = nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv1d(2048+self.audio_embed_size, 1, 1, bias=False),
                nn.BatchNorm1d(1),
                nn.Tanh(),
            ),
            'fc': nn.Linear(self.seq_len, self.seq_len, bias=True),
            'relu': nn.ReLU()
        })
        self.fc = nn.Linear(2048+self.audio_embed_size, self.n_classes)

    def _init_module(self, m):
        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _init_nonlocal(self):
        self.nl=nn.Sequential(NonLocalBlock())#,NonLocalBlock(),NonLocalBlock())
        self.nl_a=nn.Sequential(NonLocalBlock())#,NonLocalBlock(),NonLocalBlock())
        self.v2a_attn=MultiHeadAttentionOp(in_features=2048, head_num=8)
        self.a2v_attn=MultiHeadAttentionOp(in_features=2048, head_num=8)


    def forward(self, input: torch.Tensor):
        v, a = input
        # input.shape=[batch, seq_len,  3, 16, 112, 112]
        v.div_(self.NORM_VALUE).sub_(self.MEAN)
        batch, seq_len, nc, snippet_duration, sample_size, _ = v.size()
        v = v.view(batch*seq_len, nc, snippet_duration, sample_size, sample_size)
        with torch.no_grad():
            v = self.resnet(v).squeeze()
        v=v.view(batch,seq_len,-1)# B S D
        v=self.nl(v)# B S D
        
        bs = a.size(0)
        a = a.transpose(0, 1).contiguous()
        a = a.chunk(self.audio_n_segments, dim=0)
        a = torch.stack(a, dim=0).contiguous()
        a = a.transpose(1, 2).contiguous()  # [16 x bs x 256 x 32]
        a = torch.flatten(a, start_dim=0, end_dim=1)  # [B x 256 x 32]
        a = torch.unsqueeze(a, dim=1)
        a = self.a_resnet(a)
        a = torch.flatten(a, start_dim=1).contiguous()
        a = self.a_fc(a)
        a = a.view(self.audio_n_segments, bs, self.audio_embed_size).contiguous()
        a = a.permute(1, 0, 2).contiguous() # B S D
        a = self.nl_a(a)# B S D
        
        v2a, _ = self.v2a_attn(q=a, k=v, v=v)
        a2v, _ = self.a2v_attn(q=v, k=a, v=a)
        v2 = v + v2a
        a2 = a + a2v

        output=torch.cat((v2,a2),dim=-1)
        output=output.transpose(1,2).contiguous()
        Ht = self.ta_net['conv'](output)
        Ht = torch.squeeze(Ht, dim=1)
        Ht = self.ta_net['fc'](Ht)
        At = self.ta_net['relu'](Ht)
        gamma = At.view(batch, seq_len)

        output = torch.mul(output, torch.unsqueeze(At, dim=1).repeat(1, 2048+self.audio_embed_size, 1))
        output = torch.mean(output, dim=2)
        output = self.drop(output)
        output = self.fc(output)
        return output, gamma

if __name__ == '__main__':
    model=VAANetErase(
        snippet_duration=16,
        sample_size=112,
        n_classes=8,
        seq_len=16,
        audio_embed_size=2048,
        audio_n_segments=16,
        pretrained_resnet101_path='/home/ubuntu/zzc/code/vsenti/VAANet-master/data/r3d101.pth'
    ).cuda()

    visual=torch.randn(32,16,3,16,112,112).cuda()
    audio=torch.randn(32,1600,128).cuda()

    output,gamma=model([visual,audio])