import numpy as np
import torch
from torch import diag, symeig, mean, dot, eye 
class linear_cca():
    def __init__(self):
        self.w = [None, None] 
        self.m = [None, None] # mean

    def fit(self, H1, H2, outdim_size1, outdim_size2):
        """
        An implementation of linear CCA
        # Arguments:
            H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
            outdim_size: specifies the number of new features
        # Returns
            A and B: the linear transformation matrices
            mean1 and mean2: the means of data for both views
        """
        r1 = 1e-4
        r2 = 1e-4

        # print(f'H1 shape : {H1.shape}, H2 shape {H2.shape}')

        m = H1.shape[0]
        o1 = H1.shape[1]
        o2 = H2.shape[1]
        
        self.m[0] = mean(H1, axis=0)
        self.m[1] = mean(H2, axis=0)
        H1bar = H1 - self.m[0].repeat(m, 1).view(m, -1)
        H2bar = H2 - self.m[1].repeat(m, 1).view(m, -1)
        assert H1bar.shape == H1.shape
        assert H2bar.shape == H2.shape

        SigmaHat12 = (1.0 / (m - 1)) * torch.mm(H1bar.T, H2bar)
        SigmaHat11 = (1.0 / (m - 1)) * torch.mm(H1bar.T,
                                                 H1bar) + r1 * eye(o1)
        SigmaHat22 = (1.0 / (m - 1)) * torch.mm(H2bar.T,
                                                 H2bar) + r2 * eye(o2)
        
        [D1, V1] = symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = symeig(SigmaHat22, eigenvectors=True)
        SigmaHat11RootInv = torch.mm(torch.mm(V1, diag(D1 ** -0.5)), V1.T)
        SigmaHat22RootInv = torch.mm(torch.mm(V2, diag(D2 ** -0.5)), V2.T)

        Tval = torch.mm(torch.mm(SigmaHat11RootInv,SigmaHat12), SigmaHat22RootInv)

        [U, D, V] = Tval.svd()
        V = V.T
        self.w[0] = torch.mm(SigmaHat11RootInv, U[:, 0:outdim_size1])
        self.w[1] = torch.mm(SigmaHat22RootInv, V[:, 0:outdim_size2])
        D = D[0:outdim_size1]

    def _get_result(self, x, idx):
        if (self.m[0] is None) or (self.w[0] is None):
            mean_x = torch.mean(x, axis=0)
            return x - mean_x.repeat(x.shape[0], 1).view(x.shape[0], -1)
        result = x - self.m[idx].repeat(x.shape[0], 1).view(x.shape[0], -1)
        result = torch.mm(result, self.w[idx])
        return result

    def test(self, H1, H2):
        return self._get_result(H1, 0), self._get_result(H2, 1)

class linear_gcca():
    def __init__(self):
        self.G = None
        self.U = [None, None, None] 
        self.m = [None, None, None] # mean

    def fit(self, H1, H2, H3, outdim_sizes):
        """
        An implementation of linear CCA
        # Arguments:
            H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
            outdim_size: specifies the number of new features
        # Returns
            A and B: the linear transformation matrices
            mean1 and mean2: the means of data for both views
        """
        r = 1e-4

        # print(f'H1 shape : {H1.shape}, H2 shape {H2.shape}')

        H1, H2, H3 = H1.t(), H2.t(), H3.t()

        assert torch.isnan(H1).sum().item() == 0 
        assert torch.isnan(H2).sum().item() == 0
        assert torch.isnan(H3).sum().item() == 0

        o1 = H1.size(0)
        o2 = H2.size(0)
        o3 = H3.size(0)
        m = H1.size(1)

        top_k = torch.min(torch.tensor(outdim_sizes)) 

        H1bar = H1 - H1.mean(dim=1).repeat(m, 1).view(-1, m)
        H2bar = H2 - H2.mean(dim=1).repeat(m, 1).view(-1, m)
        H3bar = H3 - H3.mean(dim=1).repeat(m, 1).view(-1, m)
        assert torch.isnan(H1bar).sum().item() == 0
        assert torch.isnan(H2bar).sum().item() == 0
        assert torch.isnan(H3bar).sum().item() == 0

        print(f'Hbar shape : {H1bar.shape}, Hbar.T shape : {H1bar.T.shape}')
        SigmaHat11 = (1.0 / (m - 1)) * torch.mm(H1bar,
                                                 H1bar.T) + r * eye(o1)
        SigmaHat22 = (1.0 / (m - 1)) * torch.mm(H2bar,
                                                 H2bar.T) + r * eye(o2)
        SigmaHat33 = (1.0 / (m - 1)) * torch.mm(H3bar,
                                                 H3bar.T) + r * eye(o3)


        [D1, V1] = symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = symeig(SigmaHat22, eigenvectors=True)
        [D3, V3] = symeig(SigmaHat33, eigenvectors=True)
        SigmaHat11Inv = torch.mm(torch.mm(V1, diag(D1 ** -1)), V1.T)
        SigmaHat22Inv = torch.mm(torch.mm(V2, diag(D2 ** -1)), V2.T)
        SigmaHat33Inv = torch.mm(torch.mm(V3, diag(D3 ** -1)), V3.T)


        P1 = torch.mm(torch.mm(H1bar.t(), SigmaHat11Inv), H1bar)
        P2 = torch.mm(torch.mm(H2bar.t(), SigmaHat22Inv), H2bar)
        P3 = torch.mm(torch.mm(H3bar.t(), SigmaHat33Inv), H3bar)

        assert torch.isnan(P1).sum().item() == 0
        assert torch.isnan(P2).sum().item() == 0
        assert torch.isnan(P3).sum().item() == 0
        assert P1.shape != P2.shape
        assert P1.shape != P3.shape

        M = torch.add(torch.add(P1, P2), P3)


        [U, V] = M.svd()
        V = V.T
        self.G =  V[:, 0:top_k]

        assert torch.isnan(self.G).sum().item() == 0

        self.U[0] = torch.mm(torch.mm(SigmaHat11Inv, H1bar), G.t())
        self.U[1] = torch.mm(torch.mm(SigmaHat22Inv, H2bar), G.t())
        self.U[2] = torch.mm(torch.mm(SigmaHat33Inv, H3bar), G.t())

        assert torch.isnan(self.U[0]).sum().item() == 0
        assert torch.isnan(self.U[1]).sum().item() == 0
        assert torch.isnan(self.U[2]).sum().item() == 0

    def _get_result(self, x, idx):
        if (self.m[0] is None) or (self.U[0] is None):
            mean_x = torch.mean(x, axis=0)
            return x - mean_x.repeat(x.shape[0], 1).view(x.shape[0], -1)
        result = x - self.m[idx].repeat(x.shape[0], 1).view(x.shape[0], -1)
        result = torch.mm(result, self.U[idx])
        return result

    def test(self, H1, H2, H3): 
        return self.G, self._get_result(H1, 0), self._get_result(H2, 1), self._get_result(H3, 2) 
