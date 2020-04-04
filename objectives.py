import torch
from linear_cca import linear_gcca

class cca_loss():
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device
 
    def loss(self, H1, H2):
        """

        It is the loss function of CCA as introduced in the original paper. There can be other formulations.

        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()
        if torch.isnan(H1).sum().item() != 0 :
            print(f'H1 : {H1}')
        assert torch.isnan(H1).sum().item() == 0 
        assert torch.isnan(H2).sum().item() == 0

        o1 = o2 = H1.size(0)

        m = H1.size(1)
        # print(H1.size())

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
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
            # just the top self.outdim_size singular values are used
            U, V = torch.symeig(torch.matmul(
                Tval.t(), Tval), eigenvectors=True)
            # assert torch.isnan(V).item() == 0
            # U = U[torch.gt(U, eps).nonzero()[:, 0]]
            U = torch.where(U>eps, U, torch.ones(U.shape).double()*eps)
            U = U.topk(self.outdim_size)[0]
            # print(f'U : {U}')
            corr = torch.sum(torch.sqrt(U))
            assert torch.isnan(corr).item() == 0
        return -corr

class gcca_loss():

    def __init__(self, outdim_size, F, k, V=3, device='cpu', verbos=True, backend='pytorch'):
        self.outdim_size = outdim_size
        self.device = device
        self.F = F
        self.k = k
        self.V = 3
        self.backend = backend
        self.gccaModule = linear_gcca(self.V,
                                        self.F,
                                        self.k,
                                        verbose=verbos,
                                        backend=backend,
                                        device=device) 

    def loss_torch(self, outputs):
        '''
        output : list of NN outputs as [f(Xj)]
        '''
        self.fit(outputs)
        print('pytorch fitting is done')
        G, Us = self.gccaModule.G, self.gccaModule.U
        # print(G)
        grads = []
        for Uval, output in zip(Us, outputs):
            output = output.double()
            Ushared = Uval.double()
            grad = ( G - output.mm(Uval) ).mm(Uval.T)
            grads.append(grad)
        self.Us = Us
        totall_loss = self.gccaModule 
        return grads 

    def loss_np(self, outputs):
        '''
        output : list of NN outputs as [f(Xj)]
        '''
        self.fit(outputs)
        G, Us = self.gccaModule.G, self.gccaModule.U
        grads = []
        for Uval, output in zip(Us, outputs):
            output = output.numpy()
            Ushared = Uval
            grad = ( G - output.dot(Uval) ).dot(Uval.T)
            grads.append(grad)
        self.Us = Us 
        return grads

    def loss(self, outputs):
        if self.backend is 'pytorch':
            return self.loss_torch(outputs)
        else: return self.loss_np(outputs)

    def fit(self, outputs):
        self.gccaModule.fit(outputs)


