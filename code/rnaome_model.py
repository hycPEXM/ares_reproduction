import argparse as ap
import collections as col
from functools import partial

import pytorch_lightning as pl
import torch
from torch_scatter import scatter_mean

from e3nn.kernel import Kernel
from e3nn.linear import Linear
from e3nn import o3
from e3nn.non_linearities.norm import Norm
from e3nn.non_linearities.nonlin import Nonlinearity
from e3nn.point.message_passing import Convolution
from e3nn.radial import GaussianRadialModel


class ARESModel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ap.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-2)
        return parser

    def __init__(self, learning_rate=1e-2, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.predictions = col.defaultdict(list)
        '''
        >>> import collections as col
        >>> test = col.defaultdict(list)
        >>> test
        defaultdict(<class 'list'>, {})
        >>> test['ss'].append(213.232)
        >>> test['ss'].append('dsad')
        >>> test
        defaultdict(<class 'list'>, {'ss': [213.232, 'dsad']})

        '''
        

        # Define the input and output representations
        Rs0 = [(3, 0)]
        Rs1 = [(24, 0)]
        Rs20 = [(24, 0)]
        Rs21 = [(24, 1)]
        Rs22 = [(24, 2)]
        Rs3 = [(12, 0), (12, 1), (12, 2)]
        Rs30 = [(12, 0)]
        Rs31 = [(12, 1)]
        Rs32 = [(12, 2)]
        # To account for multiple output paths of conv.
        Rs30_exp = [(3 * 12, 0)]
        Rs31_exp = [(6 * 12, 1)]
        Rs32_exp = [(6 * 12, 2)]
        Rs4 = [(4, 0), (4, 1), (4, 2)]
        Rs40 = [(4, 0)]
        Rs41 = [(4, 1)]
        Rs42 = [(4, 2)]
        Rs40_exp = [(3 * 4, 0)]
        Rs41_exp = [(6 * 4, 1)]
        Rs42_exp = [(6 * 4, 2)]

        relu = torch.nn.ReLU()
        # Radial model:  R+ -> R^d
        RadialModel = partial(
            GaussianRadialModel, max_radius=12.0, number_of_basis=12, h=12,
            L=1, act=relu)

        ssp = ShiftedSoftplus()
        self.elu = torch.nn.ELU()

        # kernel: composed on a radial part that contains the learned
        # parameters and an angular part given by the spherical hamonics and
        # the Clebsch-Gordan coefficients
        selection_rule = partial(o3.selection_rule_in_out_sh, lmax=2)
        K = partial(
            Kernel, RadialModel=RadialModel, selection_rule=selection_rule)

        self.lin1 = Linear(Rs0, Rs1)

        self.conv10 = Convolution(K(Rs1, Rs20))
        self.conv11 = Convolution(K(Rs1, Rs21))
        self.conv12 = Convolution(K(Rs1, Rs22))

        self.norm = Norm()

        self.lin20 = Linear(Rs20, Rs20)
        self.lin21 = Linear(Rs21, Rs21)
        self.lin22 = Linear(Rs22, Rs22)

        self.nonlin10 = Nonlinearity(Rs20, act=ssp)
        self.nonlin11 = Nonlinearity(Rs21, act=ssp)
        self.nonlin12 = Nonlinearity(Rs22, act=ssp)

        self.lin30 = Linear(Rs20, Rs30)
        self.lin31 = Linear(Rs21, Rs31)
        self.lin32 = Linear(Rs22, Rs32)

        def filterfn_def(x, f):
            return x == f

        self.conv2 = torch.nn.ModuleDict()
        for i in range(3):
            for f in range(3):
                for o in range(abs(f - i), min(i + f + 1, 3)):
                    filterfn = partial(filterfn_def, f=f)
                    selection_rule = \
                        partial(o3.selection_rule, lmax=2, lfilter=filterfn)
                    K = partial(Kernel, RadialModel=RadialModel,
                                selection_rule=selection_rule)
                    self.conv2[str((i, f, o))] = \
                        Convolution(K([Rs3[i]], [Rs3[o]]))

        self.lin40 = Linear(Rs30_exp, Rs30)
        self.lin41 = Linear(Rs31_exp, Rs31)
        self.lin42 = Linear(Rs32_exp, Rs32)

        self.nonlin20 = Nonlinearity(Rs30, act=ssp)
        self.nonlin21 = Nonlinearity(Rs31, act=ssp)
        self.nonlin22 = Nonlinearity(Rs32, act=ssp)

        self.lin50 = Linear(Rs30, Rs40)
        self.lin51 = Linear(Rs31, Rs41)
        self.lin52 = Linear(Rs32, Rs42)

        self.conv3 = torch.nn.ModuleDict()
        for i in range(3):
            for f in range(3):
                for o in range(abs(f - i), min(i + f + 1, 3)):
                    filterfn = partial(filterfn_def, f=f)
                    selection_rule = \
                        partial(o3.selection_rule, lmax=2, lfilter=filterfn)
                    K = partial(Kernel, RadialModel=RadialModel,
                                selection_rule=selection_rule)
                    self.conv3[str((i, f, o))] = \
                        Convolution(K([Rs4[i]], [Rs4[o]]))

        self.lin60 = Linear(Rs40_exp, Rs40)
        self.lin61 = Linear(Rs41_exp, Rs41)
        self.lin62 = Linear(Rs42_exp, Rs42)

        self.nonlin30 = Nonlinearity(Rs40, act=ssp)
        self.nonlin31 = Nonlinearity(Rs41, act=ssp)
        self.nonlin32 = Nonlinearity(Rs42, act=ssp)

        self.dense1 = torch.nn.Linear(4, 4, bias=True)
        self.dense2 = torch.nn.Linear(4, 256, bias=True)
        self.dense3 = torch.nn.Linear(256, 1, bias=True)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = torch.nn.functional.smooth_l1_loss(y_hat, batch.label.float())
        self.log('loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = torch.nn.functional.smooth_l1_loss(y_hat, batch.label.float())
        self.log('val_loss', loss, prog_bar=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = torch.nn.functional.smooth_l1_loss(y_hat, batch.label.float()) #batch: The output of your DataLoader.
        self.predictions['ares'].extend(y_hat.cpu().numpy())  #'pred' -> 'ares'
        self.predictions['tag'].extend(batch.id)  #'id' -> 'tag'
        self.predictions['rms'].extend(batch.label.cpu().numpy())
        self.predictions['file_path'].extend(batch.file_path)
        for i in range(self.fe.cpu().shape[1]):
            #fe_list=torch.transpose(self.fe.cpu(),0,1).tolist()
            #self.predictions['fe{}'.format(i)].extend(fe_list[i])
            self.predictions['fe{}'.format(i)].extend(self.fe.cpu()[:,i].tolist())
        return {'test_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def forward(self, d):
        out = self.lin1(d.x)

        out0 = self.conv10(out, d.edge_index, d.edge_attr)
        out1 = self.conv11(out, d.edge_index, d.edge_attr)
        out2 = self.conv12(out, d.edge_index, d.edge_attr)

        out0 = self.norm(out0)
        out1 = self.norm(out1)
        out2 = self.norm(out2)

        out0 = self.lin20(out0)
        out1 = self.lin21(out1)
        out2 = self.lin22(out2)

        out0 = self.nonlin10(out0)
        out1 = self.nonlin11(out1)
        out2 = self.nonlin12(out2)

        out0 = self.lin30(out0)
        out1 = self.lin31(out1)
        out2 = self.lin32(out2)

        ins = {0: out0, 1: out1, 2: out2}
        tmp = col.defaultdict(list)
        for i in range(3):
            for f in range(3):
                for o in range(abs(f - i), min(i + f + 1, 3)):
                    curr = self.conv2[str((i, f, o))](
                        ins[i], d.edge_index, d.edge_attr)
                    tmp[o].append(curr)
        out0 = torch.cat(tmp[0], axis=1)
        out1 = torch.cat(tmp[1], axis=1)
        out2 = torch.cat(tmp[2], axis=1)

        out0 = self.norm(out0)
        out1 = self.norm(out1)
        out2 = self.norm(out2)

        out0 = self.lin40(out0)
        out1 = self.lin41(out1)
        out2 = self.lin42(out2)

        out0 = self.nonlin20(out0)
        out1 = self.nonlin21(out1)
        out2 = self.nonlin22(out2)

        out0 = self.lin50(out0)
        out1 = self.lin51(out1)
        out2 = self.lin52(out2)

        ins = {0: out0, 1: out1, 2: out2}
        tmp = col.defaultdict(list)
        for i in range(3):
            for f in range(3):
                for o in range(abs(f - i), min(i + f + 1, 3)):
                    curr = self.conv3[str((i, f, o))](
                        ins[i], d.edge_index, d.edge_attr)
                    tmp[o].append(curr)

        out0 = torch.cat(tmp[0], axis=1)
        out1 = torch.cat(tmp[1], axis=1)
        out2 = torch.cat(tmp[2], axis=1)

        out0 = self.norm(out0)
        out1 = self.norm(out1)
        out2 = self.norm(out2)

        out0 = self.lin60(out0)
        out1 = self.lin61(out1)
        out2 = self.lin62(out2)

        out0 = self.nonlin30(out0)
        out1 = self.nonlin31(out1)
        out2 = self.nonlin32(out2)

        # Per-channel mean.
        out = scatter_mean(out0, d.batch, dim=0)

        out = self.dense1(out)
        out = self.elu(out)
        out = self.dense2(out)
        
        self.fe=out.detach()
        #可以在rnaome_predict.py里print(tfnn.fe.cpu().shape)获知其形状为
        #torch.Size([7, 256]，这里7的含义应该是1575%16=7，因为最后一批测试样本的大小为7
        
        # print(out.shape)  #print没用，无法打印结果
        # self.fe=torch.squeeze(out).detach().tolist()
        # for i in range(len(self.fe)):
            # self.predictions['fe{}'.format(i)].append(self.fe[i])
            
            
        out = self.dense3(out)
        out = torch.squeeze(out, axis=1) #推荐将axis=1换成dim=1
        return out


class ShiftedSoftplus:
    def __init__(self):
        self.shift = torch.nn.functional.softplus(torch.zeros(())).item()

    def __call__(self, x):
        return torch.nn.functional.softplus(x).sub(self.shift)  
        #beta=1, η(x) = log(1+e^x)-log2=log(0.5+0.5e^x)
