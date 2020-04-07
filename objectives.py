import torch
from torch import diag, symeig

class cca_loss():
    def __init__(self, outdim_size1, outdim_size2, use_all_singular_values, device):
        self.outdim_size1 = outdim_size1
        self.outdim_size2 = outdim_size2
        self.top_k = min(outdim_size1, outdim_size2)
        self.use_all_singular_values = use_all_singular_values
        self.device = device
 
    def loss(self, H1, H2):
        """

        It is the loss function of CCA as introduced in the original paper. There can be other formulations.

        """
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-7
        lambda_b = 1e-7

        H1, H2 = H1.to(self.device), H2.to(self.device)
        H1, H2 = H1.t(), H2.t()
        if torch.isnan(H1).sum().item() != 0 :
            print(f'H1 : {H1}')
        assert torch.isnan(H1).sum().item() == 0 
        assert torch.isnan(H2).sum().item() == 0

        o1 = H1.size(0)
        o2 = H2.size(0)
        m = H1.size(1)
        # print(H1.size())

        H1bar = H1 - H1.mean(dim=1).repeat(m, 1).view(m, -1)
        H2bar = H2 - H2.mean(dim=1).repeat(m, 1).view(m, -1)
        assert torch.isnan(H1bar).sum().item() == 0
        assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        assert torch.isnan(SigmaHat11).sum().item() == 0
        assert torch.isnan(SigmaHat12).sum().item() == 0
        assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
        assert torch.isnan(D1).sum().item() == 0
        assert torch.isnan(D2).sum().item() == 0
        assert torch.isnan(V1).sum().item() == 0
        assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        # print(posInd1.size())
        # print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)
#         print(Tval.size())

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.trace(torch.matmul(Tval.t(), Tval))
            # print(tmp)
            corr = torch.sqrt(tmp)
            assert torch.isnan(corr).item() == 0
        else:
            # just the top self.top_k singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, torch.eye(trace_TT.shape[0])*r1) # regularization for more tability
            U, V = torch.symeig(trace_TT, eigenvectors=True)
            # _, S, _= torch.svd(trace_TT, compute_uv=True)
            # assert torch.isnan(U).item() == 0

            # U = U[torch.gt(U, eps).nonzero()[:, 0]]

            if U.le(eps).sum() != 0 :
                print(f'number of unstability : {U.le(eps).sum()}, index : {U[U.le(eps)]}')
            U = torch.where(U>eps, U, torch.ones(U.shape).double()*lambda_b)
            U = U.topk(self.top_k)[0]
            corr = torch.sum(torch.sqrt(U))
            # corr = torch.sum(S)
            assert torch.isnan(corr).item() == 0
        return -corr

class gcca_loss():
    def __init__(self, outdim_sizes, use_all_singular_values, device):
        self.outdim_size1 = outdim_sizes[0]
        self.outdim_size2 = outdim_sizes[1]
        self.outdim_size3 = outdim_sizes[2]
        self.top_k = min(self.outdim_size1, self.outdim_size2, self.outdim_size3)
        self.use_all_singular_values = use_all_singular_values
        self.device = device
 
    def loss(self, H1, H2, H3):
        """

        It is the loss function of CCA as introduced in the original paper. There can be other formulations.

        """

        r = 1e-3
        eps = 1e-7
        lambda_b = 1e-4

        H1, H2, H3 = H1.t(), H2.t(), H3.t()

        assert torch.isnan(H1).sum().item() == 0 
        assert torch.isnan(H2).sum().item() == 0
        assert torch.isnan(H3).sum().item() == 0
        # print(f'H1 : {H1.shape}, H2 : {H2.shape}, H3 : {H3.shape}')
        o1 = H1.size(0)
        o2 = H2.size(0)
        o3 = H3.size(0)
        m = H1.size(1)
        # print(H1.size())
        ### check this from paper DeepCCA,
        H1bar = H1 - H1.mean(dim=1).repeat(m, 1).view(-1, m)
        H2bar = H2 - H2.mean(dim=1).repeat(m, 1).view(-1, m)
        H3bar = H3 - H3.mean(dim=1).repeat(m, 1).view(-1, m)
        assert torch.isnan(H1bar).sum().item() == 0
        assert torch.isnan(H2bar).sum().item() == 0
        assert torch.isnan(H3bar).sum().item() == 0 

        # SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r * torch.eye(o2, device=self.device)
        SigmaHat33 = (1.0 / (m - 1)) * torch.matmul(H3bar,
                                                    H3bar.t()) + r * torch.eye(o3, device=self.device)                                                
        assert torch.isnan(SigmaHat11).sum().item() == 0
        assert torch.isnan(SigmaHat22).sum().item() == 0
        assert torch.isnan(SigmaHat33).sum().item() == 0

        P1 = torch.mm(torch.mm(H1bar.t(), SigmaHat11), H1bar)
        P2 = torch.mm(torch.mm(H2bar.t(), SigmaHat22), H2bar)
        P3 = torch.mm(torch.mm(H3bar.t(), SigmaHat33), H3bar)

        assert torch.isnan(P1).sum().item() == 0
        assert torch.isnan(P2).sum().item() == 0
        assert torch.isnan(P3).sum().item() == 0
        # print(f'P1 : {P1.shape}, P2 : {P2.shape}, P3 : {P3.shape}')
        # assert P1.shape != P2.shape
        # assert P1.shape != P3.shape

        M = torch.add(torch.add(P1, P2), P3)
        [D1, V1] = symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = symeig(SigmaHat22, eigenvectors=True)
        [D3, V3] = symeig(SigmaHat33, eigenvectors=True)
        SigmaHat11Inv = torch.mm(torch.mm(V1, diag(D1 ** -1)), V1.T)
        SigmaHat22Inv = torch.mm(torch.mm(V2, diag(D2 ** -1)), V2.T)
        SigmaHat33Inv = torch.mm(torch.mm(V3, diag(D3 ** -1)), V3.T)

        M = torch.add(torch.add(P1, P2), P3)


        U, V = torch.symeig(M, eigenvectors=True)
        U = torch.where(U>eps, U, torch.ones(U.shape).double()*eps)
        if not self.use_all_singular_values:
            U= U.topk(self.top_k)[0]
        corr = torch.sum(torch.sqrt(U))
        assert torch.isnan(corr).item() == 0
        return -corr ## Eq. to Jr(constant number) - sum(Lamda(M))
