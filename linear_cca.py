import numpy as np
import torch
from torch import diag, symeig, mean, dot, eye

import pickle, gzip, os, sys, time

import scipy
import scipy.sparse
import scipy.linalg

class linear_cca():
    def __init__(self):
        self.w = [None, None] 
        self.m = [None, None] # mean

    def fit(self, H1, H2, outdim_size):
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
        self.w[0] = torch.mm(SigmaHat11RootInv, U[:, 0:outdim_size])
        self.w[1] = torch.mm(SigmaHat22RootInv, V[:, 0:outdim_size])
        D = D[0:outdim_size]

    def _get_result(self, x, idx):
        if (self.m[0] is None) or (self.w[0] is None):
            return x 
        result = x - self.m[idx].repeat(x.shape[0], 1).view(x.shape[0], -1)
        result = torch.mm(result, self.w[idx])
        return result

    def test(self, H1, H2):
        return self._get_result(H1, 0), self._get_result(H2, 1)


class linear_gcca():
    def __init__(self, V, F, k, eps=1e-7, truncParam=1000, viewWts=None, verbose=True, backend='pytorch', device='cpu'):
        super().__init__()
        self.V = V # Number of views
        self.F = F # Number of features per view
        self.k = k # Dimensionality of embedding we want to learn
        self.truncParam = truncParam # Take rank-truncParam SVD of data matrices
        
        # Regularization for each view
        self.eps = eps
        self.W  = [np.float32(v) for v in viewWts] if viewWts is not None else [np.float32(1.) for v in range(V)] # How much we should weight each view -- defaults to equal weighting
        if backend is 'pytorch':
            self.W = torch.ones((V, 1)).squeeze().double().to(device) 

        self.U = None # Projection from each view to shared space
        self.G = None # Embeddings for training examples
        self.G_scaled = None # Scaled by SVs of covariance matrix sum
        
        self.verbose = verbose
        self.device = device
        
        self.backend = backend

        self.mean = [0, 0, 0]

    def _compute_numpy(self, views, K=None, incremental=True):
        '''
        Compute G by first taking low-rank decompositions of each view, stacking
        them, then computing the SVD of this matrix.
        '''
        ## Mean-cenetring views :
        self.mean = []
        for index, view in enumerate(views):
            m = view.shape[0]
            mean = np.mean(view, axis=0)
            self.mean.append(mean)
            view = view - np.tile(mean, (m, 1))

        # K ignores those views we have no data for.  If it is not provided,
        # then we use all views for all examples.  All we need to know is
        # K^{-1/2}, which just weights each example based on number of non-zero
        # views.  Will fail if there are any empty examples.
        eps = self.eps

        if K is None:
            K = np.float32(np.ones((views[0].shape[0], len(views))))
        else:
            K = np.float32(K)

        # We do not want to count missing views we are downweighting heavily/zeroing out, so scale K by W
        K = K.dot(np.diag(self.W))
        Ksum = np.sum(K, axis=1)
        
        # If we have some missing rows after weighting, then make these small & positive.
        Ksum[Ksum==0.] = 1.e-8

        K_invSqrt = scipy.sparse.dia_matrix( ( 1. / np.sqrt( Ksum ), np.asarray([0]) ), shape=(K.shape[0], K.shape[0]) )

        # Left singular vectors for each view along with scaling matrices
        
        As = []
        Ts = []
        Ts_unnorm = []
        
        N = views[0].shape[0]
        
        _Stilde  = np.float32(np.zeros(self.k))
        _Gprime = np.float32(np.zeros((N, self.k)))
        _Stilde_scaled = np.float32(np.zeros(self.k))
        _Gprime_scaled = np.float32(np.zeros((N, self.k)))
        
        # Take SVD of each view, to calculate A_i and T_i
        for i, view in enumerate(views):
            A, S, B = scipy.linalg.svd(view, full_matrices=False, check_finite=False)
            
            # Find T by just manipulating singular values.  Matrices are all diagonal,
            # so this should be fine.
            
            S_thin = S[:self.truncParam]
            
            S2_inv = 1. / (np.multiply( S_thin, S_thin ) + eps)
            
            T = np.diag(
                    np.sqrt(
                    np.multiply( np.multiply( S_thin, S2_inv ), S_thin )
                    )
                )
            
            # Keep singular values
            T_unnorm = np.diag( S_thin + eps )
            
            if incremental:
                ajtj = K_invSqrt.dot( np.sqrt(self.W[i]) * A.dot(T) )
                ajtj_scaled = K_invSqrt.dot( np.sqrt(self.W[i]) * A.dot(T_unnorm) )
                
                _Gprime, _Stilde = self._batch_incremental_pca(ajtj,
                                                                    _Gprime,
                                                                    _Stilde)
                _Gprime_scaled, _Stilde_scaled = self._batch_incremental_pca(ajtj_scaled,
                                                                                    _Gprime_scaled,
                                                                                    _Stilde_scaled)
            else:
                # Keep the left singular vectors of view j
                As.append(A[:,:self.truncParam])
                Ts.append(T)
                Ts_unnorm.append(T_unnorm)
            
            if self.verbose:
                print ('Decomposed data matrix for view %d' % (i))
            
            
        if incremental:
            self.G        = _Gprime
            self.G_scaled = _Gprime_scaled
            
            self.lbda = _Stilde
            self.lbda_scaled = _Stilde_scaled
        else:
            # In practice M_tilde may be really big, so we would
            # like to perform this SVD incrementally, over examples.
            M_tilde = K_invSqrt.dot( np.bmat( [ np.sqrt(w) * A.dot(T) for w, A, T in zip(self.W, As, Ts) ] ) )
            
            Q, R = scipy.linalg.qr( M_tilde, mode='economic')
            
            # Ignore right singular vectors
            U, lbda, V_toss = scipy.linalg.svd(R, full_matrices=False, check_finite=False)
            
            self.G = Q.dot(U[:,:self.k])
            self.lbda = lbda
            
            # Unnormalized version of G -> captures covariance between views
            M_tilde = K_invSqrt.dot( np.bmat( [ np.sqrt(w) * A.dot(T) for w, A, T in zip(self.W, As, Ts_unnorm) ] ) )
            Q, R = scipy.linalg.qr( M_tilde, mode='economic')
            
            # Ignore right singular vectors
            U, lbda, V_toss = scipy.linalg.svd(R, full_matrices=False, check_finite=False)
            
            self.lbda_scaled = lbda
            self.G_scaled = self.G.dot(np.diag(self.lbda_scaled[:self.k]))
            
            if self.verbose:
                print ('Decomposed M_tilde / solved for G')
        
        self.U = [] # Mapping from views to latent space
        self.U_unnorm = [] # Mapping, without normalizing variance
        #self._partUs = []
        
        # Get mapping to shared space
        for idx, (f, view) in enumerate(zip(self.F, views)):
            R = scipy.linalg.qr(view, mode='r')[0]
            Cjj_inv = np.linalg.inv( (R.T.dot(R) + eps * np.eye( f )) )
            pinv = Cjj_inv.dot( view.T )
            
            #self._partUs.append(pinv)
            
            self.U.append(pinv.dot( self.G ))
            self.U_unnorm.append(pinv.dot( self.G_scaled ))
            
            if self.verbose:
                print ('Solved for U in view %d' % (idx))
    
    @staticmethod
    def _batch_incremental_pca(x, G, S):
        r = G.shape[1]
        b = x.shape[0]
        
        xh = G.T.dot(x)
        H  = x - G.dot(xh)
        J, W = scipy.linalg.qr(H, overwrite_a=True, mode='full', check_finite=False)
        
        Q = np.bmat( [[np.diag(S), xh], [np.zeros((b,r), dtype=np.float32), W]] )
        
        G_new, St_new, Vtoss = scipy.linalg.svd(Q, full_matrices=False, check_finite=False)
        St_new=St_new[:r]
        G_new= np.asarray(np.bmat([G, J]).dot( G_new[:,:r] ))
        
        return G_new, St_new
    
    def fit(self, views, K=None, incremental=False):
        '''
        Learn WGCCA embeddings on training set of views.  Set incremental to true if you have
        many views.
        '''
        
        self._compute(views, K, incremental)
        if self.verbose:
            print('GCCA learning process is Done.')

    def _compute(self, views, K=None, incremental=False):
        print(self.backend)
        if self.backend is 'numpy':
            return self._compute_numpy(views, K, incremental)
        return  self._compute_torch(views, K, incremental) 

    def test(self, views, K=None, scaleBySv=False):
        '''
        Extracts WGCCA embedding for new set of examples.  Maps each present view with
        $U_i$ and takes mean of embeddings.
        
        If scaleBySv is true, then does not normalize variance of each canonical
        direction.  This corresponds to GCCA-sv in "Learning iview embeddings
        of twitter users."  Applying WGCCA to a single view with scaleBySv set to true
        is equivalent to PCA.
        '''
        
        Us        = self.U_unnorm if scaleBySv else self.U
        projViews = []
        
        N = views[0].shape[0]
        
        if K is None:
            K = np.ones((N, self.V)) # Assume we have data for all views/examples
        
        for U, v in zip(Us, views):
            projViews.append( v.dot(U) )
        projViewsStacked = np.stack(projViews)
        
        # Get mean embedding from all views, weighting each view appropriately
        
        weighting = np.iply(K, self.W) # How much to weight each example/view
        
        # If no views are present, embedding for that example will be NaN
        denom = weighting.sum(axis=1).reshape((N, 1))
        
        # Weight each view
        weightedViews = weighting.T.reshape((self.V, N, 1)) * projViewsStacked
        
        Gsum = weightedViews.sum(axis=0)
        Gprime = Gsum/denom
        
        Gprime = np.iply(Gsum, 1./denom)
        
        return Gprime

    def _compute_torch(self, views, K=None, incremental=False):
        '''
        Compute G by first taking low-rank decompositions of each view, stacking
        them, then computing the SVD of this matrix.
        '''
        # mean centering views :
        self.mean = []
        for index, view in enumerate(views):
            m = view.shape[0]
            mean = torch.mean(view, axis=0)
            self.mean.append(mean)
            view = view - mean.repeat(m, 1).view(m, -1)
        # K ignores those views we have no data for.  If it is not provided,
        # then we use all views for all examples.  All we need to know is
        # K^{-1/2}, which just weights each example based on number of non-zero
        # views.  Will fail if there are any empty examples.
        eps = self.eps
        if K is None:
            K = torch.ones( size=(views[0].shape[0], len(views)) ).double()
        else:
            K = K.double()

        # We do not want to count missing views we are downweighting heavily/zeroing out, so scale K by W
        K = K.mm(torch.diag(self.W))
        Ksum = torch.sum(K, axis=1)
        
        # If we have some missing rows after weighting, then make these small & positive.
        Ksum = torch.where( Ksum>self.eps, Ksum, torch.ones(Ksum.shape).double()*self.eps)
        
        K_invSqrt = torch.diag( 1. / torch.sqrt( Ksum ) )
        
        # Left singular vectors for each view along with scaling matrices
        
        As = []
        Ts = []
        Ts_unnorm = []
        
        N = views[0].shape[0]
        
        _Stilde  = torch.zeros(self.k).double().to(self.device)
        _Gprime = torch.zeros((N, self.k)).double().to(self.device)
        _Stilde_scaled = torch.zeros(self.k).double().to(self.device)
        _Gprime_scaled = torch.zeros((N, self.k)).double().to(self.device)
        
        # Take SVD of each view, to calculate A_i and T_i
        for i, view in enumerate(views):
            A, S, B = torch.svd(view, some=True)
            
            # Find T by just manipulating singular values.  Matrices are all diagonal,
            # so this should be fine.
            # S = S.unsqueeze(-1)
            S_thin = S[:self.truncParam]

            # print(f'S_thin shape : {S_thin}')

            # print(f'S_thin shape : {S_thin.shape}')
            
            S2_inv = (1. / (torch.mul(S_thin, S_thin) + self.eps))

            # print(f'S2_inv shape : {S2_inv}') 

            T = torch.diag(
                    torch.sqrt(
                    torch.mul( torch.mul(S_thin, S2_inv) , S_thin )
                    )
                )
            assert not torch.isnan(T).any()
            # Keep singular values
            T_unnorm = torch.diag( S_thin + eps )
            
            if incremental:
                ajtj = K_invSqrt.dot( torch.sqrt(self.W[i]) * A.dot(T) )
                ajtj_scaled = K_invSqrt.dot( torch.sqrt(self.W[i]) * A.dot(T_unnorm) )
                
                _Gprime, _Stilde = self._batch_incremental_pca_torch(ajtj,
                                                                    _Gprime,
                                                                    _Stilde)
                _Gprime_scaled, _Stilde_scaled = self._batch_incremental_pca_torch(ajtj_scaled,
                                                                                    _Gprime_scaled,
                                                                                    _Stilde_scaled)
            else:
                # Keep the left singular vectors of view j
                As.append(A[:,:self.truncParam])
                Ts.append(T)
                Ts_unnorm.append(T_unnorm)
            
            if self.verbose:
                print ('Decomposed data matrix for view %d' % (i))
            
            
        if incremental:
            self.G        = _Gprime
            self.G_scaled = _Gprime_scaled
            
            self.lbda = _Stilde
            self.lbda_scaled = _Stilde_scaled
        else:
            # In practice M_tilde may be really big, so we would
            # like to perform this SVD incrementally, over examples.
            K_invSqrt = K_invSqrt.double()
            M_tilde = K_invSqrt.mm( torch.cat( [ torch.sqrt(w) * A.mm(T) for w, A, T in zip(self.W, As, Ts) ], dim=1).double() )
            
            Q, R = torch.qr( M_tilde, some=True)
            
            # Ignore right singular vectors
            U, lbda, V_toss = torch.svd(R, some=True, compute_uv=True)
            
            self.G = Q.mm(U[:,:self.k])
            self.lbda = lbda
            
            # Unnormalized version of G -> captures covariance between views
            M_tilde = K_invSqrt.mm( torch.cat( [ torch.sqrt(w) * A.mm(T) for w, A, T in zip(self.W, As, Ts_unnorm) ], dim=1).double() )
            Q, R = torch.qr( M_tilde, some=True)
            
            # Ignore right singular vectors
            U, lbda, V_toss = torch.svd(R, some=True, compute_uv=True)
            
            self.lbda_scaled = lbda
            self.G_scaled = self.G.mm(torch.diag(self.lbda_scaled[:self.k]))
            
            if self.verbose:
                print ('Decomposed M_tilde / solved for G')
        
        self.U = [] # Mapping from views to latent space
        self.U_unnorm = [] # Mapping, without normalizing variance
        #self._partUs = []
        
        # Get mapping to shared space
        for idx,(f, view) in enumerate(zip(self.F, views)):
            _ , R = torch.qr(view, some=True)
            Cjj_inv = torch.inverse( (R.T.mm(R) + eps * torch.eye( f )) )
            pinv = Cjj_inv.mm( view.T ).double()
            
            #self._partUs.append(pinv)
            self.U.append(pinv.mm( self.G ))
            self.U_unnorm.append(pinv.mm( self.G_scaled ))
            
            if self.verbose:
                print ('Solved for U in view %d' % (idx))
    
    @staticmethod
    def _batch_incremental_pca_torch(x, G, S):
        r = G.shape[1]
        b = x.shape[0]
        
        xh = G.T.mm(x)
        H  = x - G.mm(xh)
        J, W = torch.qr(H, some=False)
        
        # Q = np.bmat( [[torch.diag(S), xh], [torch.zeros((b,r), W]]] )
        Q = torch.cat([torch.cat([torch.diag(S), torch.zeros((b, r))], dim=0), torch.cat([xh, W], dim=0)], dim=1).double()
        
        G_new, St_new, Vtoss = torch.svd(Q, some=True, compute_uv=True)
        St_new = St_new[:r]
        G_new = torch.cat([G, J], dim=0).mm( G_new[:,:r] )
         
        return G_new, St_new  

    def reconstructionErr(self, views, missingData=None, G=None):
        pass

    def apply(self, views, K=None, scaleBySv=False):
        '''
        Extracts WGCCA embedding for new set of examples.  Maps each present view with
        $U_i$ and takes mean of embeddings.
        
        If scaleBySv is true, then does not normalize variance of each canonical
        direction.  This corresponds to GCCA-sv in "Learning iview embeddings
        of twitter users."  Applying WGCCA to a single view with scaleBySv set to true
        is equivalent to PCA.
        '''
        if self.backend is 'numpy':
            return self.apply_numpy(views, K=K, scaleBySv=scaleBySv)
        else : return self.apply_torch(views, K=K, scaleBySv=scaleBySv)

    def apply_numpy(self, views, K=None, scaleBySv=False):
                
        Us        = self.U_unnorm if scaleBySv else self.U
        projViews = []
        
        N = views[0].shape[0]
        
        if K is None:
            K = np.ones((N, self.V)) # Assume we have data for all views/examples
        
        for U, v in zip(Us, views):
            projViews.append( v.dot(U) )
        projViewsStacked = np.stack(projViews)
        
        # Get mean embedding from all views, weighting each view appropriately
        
        weighting = np.multiply(K, self.W) # How much to weight each example/view
        
        # If no views are present, embedding for that example will be NaN
        denom = weighting.sum(axis=1).reshape((N, 1))
        
        # Weight each view
        weightedViews = weighting.T.reshape((self.V, N, 1)) * projViewsStacked
        
        Gsum = weightedViews.sum(axis=0)
        Gprime = Gsum/denom
        
        Gprime = np.multiply(Gsum, 1./denom)
        
        return Gprime

    def apply_torch(self, views, K=None, scaleBySv=False):
                
        Us        = self.U_unnorm if scaleBySv else self.U
        projViews = []
        
        N = views[0].shape[0]
        
        if K is None:
            K = torch.ones((N, self.V)) # Assume we have data for all views/examples
        
        for U, v in zip(Us, views):
            projViews.append( v.mm(U) )
        projViewsStacked = torch.stack(projViews)
        
        # Get mean embedding from all views, weighting each view appropriately
        
        weighting = torch.mul(K, self.W) # How much to weight each example/view
        
        # If no views are present, embedding for that example will be NaN
        denom = weighting.sum(axis=1).reshape((N, 1))
        
        # Weight each view
        weightedViews = weighting.T.reshape((self.V, N, 1)) * projViewsStacked
        
        Gsum = weightedViews.sum(axis=0)
        Gprime = Gsum/denom
        
        Gprime = torch.mul(Gsum, 1./denom)
        
        return Gprime