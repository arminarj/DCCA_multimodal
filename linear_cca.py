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

    def fit(self, H1, H2, H3, outdim_sizes, device='cuda'):
        """
        An implementation of linear GCCA
        # Arguments:
            H1 and H2 and H3: the matrices containing the data for view 1 and view 2 and 3. Each row is a sample.
            outdim_sizes: list of numbers input shapes
        # Returns
            G : the linear transformation matrices
        """
        r = 1e-4
        eps = 1e-8

        

        # H1, H2, H3 = H1.t(), H2.t(), H3.t()

        print(f'H1 shape ( N X feature) : {H1.shape}')

        assert torch.isnan(H1).sum().item() == 0 
        assert torch.isnan(H2).sum().item() == 0
        assert torch.isnan(H3).sum().item() == 0

        o1 = H1.size(0)  # N
        o2 = H2.size(0)
        o3 = H3.size(0)
        m = H1.size(1)   # out_dim

        top_k = torch.min(torch.tensor(outdim_sizes)) 

        H1bar = H1 - H1.mean(dim=1).repeat(m, 1).view(-1, m)
        H2bar = H2 - H2.mean(dim=1).repeat(m, 1).view(-1, m)
        H3bar = H3 - H3.mean(dim=1).repeat(m, 1).view(-1, m)
        assert torch.isnan(H1bar).sum().item() == 0
        assert torch.isnan(H2bar).sum().item() == 0
        assert torch.isnan(H3bar).sum().item() == 0

        A1, S1, B1 = H1bar.svd(some=True, compute_uv=True)
        A2, S2, B2 = H2bar.svd(some=True, compute_uv=True)
        A3, S3, B3 = H3bar.svd(some=True, compute_uv=True)

        A1, A2, A3 = A1[:, :top_k], A2[:, :top_k], A3[:, :top_k]

        assert torch.isnan(A1).sum().item() == 0
        assert torch.isnan(A2).sum().item() == 0
        assert torch.isnan(A3).sum().item() == 0

        S_thin_1, S_thin_2, S_thin_3 = S1[:top_k], S2[:top_k], S3[:top_k]

        S2_inv_1 = 1. / (torch.mul( S_thin_1, S_thin_1 ) + eps)
        S2_inv_2 = 1. / (torch.mul( S_thin_2, S_thin_2 ) + eps)
        S2_inv_3 = 1. / (torch.mul( S_thin_3, S_thin_3 ) + eps)

        assert torch.isnan(S2_inv_1).sum().item() == 0
        assert torch.isnan(S2_inv_2).sum().item() == 0
        assert torch.isnan(S2_inv_3).sum().item() == 0

        T2_1 = torch.mul( torch.mul( S_thin_1, S2_inv_1 ), S_thin_1 )
        T2_2 = torch.mul( torch.mul( S_thin_2, S2_inv_2 ), S_thin_2 )
        T2_3 = torch.mul( torch.mul( S_thin_3, S2_inv_3 ), S_thin_3 )

        assert torch.isnan(T2_1).sum().item() == 0
        assert torch.isnan(T2_2).sum().item() == 0
        assert torch.isnan(T2_3).sum().item() == 0

        T2_1 = torch.where(T2_1>eps, T2_1, (torch.ones(T2_1.shape)*eps).to(device).double())
        T2_2 = torch.where(T2_2>eps, T2_2, (torch.ones(T2_2.shape)*eps).to(device).double())
        T2_3 = torch.where(T2_3>eps, T2_3, (torch.ones(T2_3.shape)*eps).to(device).double())


        T_1 = torch.diag(torch.sqrt(T2_1))
        T_2 = torch.diag(torch.sqrt(T2_2))
        T_3 = torch.diag(torch.sqrt(T2_3)) 

        assert torch.isnan(T_1).sum().item() == 0
        assert torch.isnan(T_2).sum().item() == 0
        assert torch.isnan(T_3).sum().item() == 0

        T_unnorm_1 = torch.diag( S_thin_1 + eps )
        T_unnorm_2 = torch.diag( S_thin_2 + eps )
        T_unnorm_3 = torch.diag( S_thin_3 + eps )

        assert torch.isnan(T_unnorm_1).sum().item() == 0
        assert torch.isnan(T_unnorm_2).sum().item() == 0
        assert torch.isnan(T_unnorm_3).sum().item() == 0

        AT_1 = torch.mm(A1, T_1)
        AT_2 = torch.mm(A2, T_2)
        AT_3 = torch.mm(A3, T_3)

        M_tilde = torch.cat([AT_1, AT_2, AT_3], dim=1)

        print(f'M_tilde shape : {M_tilde.shape}')

        assert torch.isnan(M_tilde).sum().item() == 0

        Q, R = M_tilde.qr()

        assert torch.isnan(R).sum().item() == 0
        assert torch.isnan(Q).sum().item() == 0

        U, lbda, _ = R.svd(some=False, compute_uv=True)

        assert torch.isnan(U).sum().item() == 0
        assert torch.isnan(lbda).sum().item() == 0

        self.G = Q.mm(U[:,:top_k])
        print(f'G shape : {self.G.shape}')
        assert torch.isnan(self.G).sum().item() == 0


    # def _get_result(self, x, idx):
    #     if (self.m[0] is None) or (self.U[0] is None):
    #         return x
    #     result = x - self.m[idx].repeat(x.shape[0], 1).view(x.shape[0], -1)
    #     result = torch.mm(result, self.U[idx])
    #     return result

    def test(self, H1, H2, H3): 
        return self.G, H1, H2, H3
