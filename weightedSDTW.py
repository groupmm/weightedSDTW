########################################################
# Soft Dynamic Time Warping With Variable Step Weights #
#                                                      #
# Johannes Zeitler, 2024                               #
# johannes.zeitler@audiolabs-erlangen.de               #
########################################################

# Accompanying code for the paper
# Johannes Zeitler, Michael Krause, and Meinard Müller: "Soft Dynamic Time Warping With Variable Step Weights", International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2024, Seoul, Korea. 

# Based on the following repositories: 
# Kanru Hua: pytorch-softdtw (https://github.com/Sleepwalking/pytorch-softdtw)
# Mehran Maghoumi: pytorch-softdtw-cuda (https://github.com/Maghoumi/pytorch-softdtw-cuda)

#----------------------------------------------------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Johannes Zeitler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ----------------------------------------------------------------------------------------------------------------------
from functools import partial

import numpy as np
import torch
import torch.cuda
from numba import jit, prange
from torch.autograd import Function
from numba import cuda
import math


# ----------------------------------------------------------------------------------------------------------------------
#
# The following is based on the GPU implementation https://github.com/Maghoumi/pytorch-softdtw-cuda
# ----------------------------------------------------------------------------------------------------------------------

@cuda.jit
def compute_weightedSDTW_cuda(C, gamma, max_i, max_j, n_passes, D, weights):
    """
    :param seq_len: The length of the sequence (both inputs are assumed to be of the same size)
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # Compute I, J, the indices from [0, seq_len)

    # The row index is always the same as tid
    I = tid

    inv_gamma = 1.0 / gamma

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_passes):

        # The index is actually 'p - tid' but need to force it in-bounds
        J = max(0, min(p - tid, max_j - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == p and (I < max_i and J < max_j):
            d0 = - (C[b, i-1, j-1]*weights[2] +  D[b, i - 1, j - 1]) * inv_gamma
            d1 = - (C[b, i-1, j-1]*weights[0] +  D[b, i - 1, j]) * inv_gamma
            d2 = - (C[b, i-1, j-1]*weights[1] +  D[b, i, j - 1]) * inv_gamma
            dmax = max(max(d0, d1), d2)
            dsum = math.exp(d0 - dmax) + math.exp(d1 - dmax) + math.exp(d2 - dmax)
            D[b, i, j] = -gamma * (math.log(dsum) + dmax)

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_weightedSDTW_backward_cuda(C, D, inv_gamma, max_i, max_j, n_passes, E, H, weights):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    I = tid

    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i and J < max_j):

            if math.isinf(D[k, i, j]):
                D[k, i, j] = -math.inf

            # Don't compute if outside bandwidth
            #if not (abs(i - j) > bandwidth > 0):
            F_10 = math.exp((D[k, i + 1, j] - D[k, i, j] - weights[0]*C[k, i + 1, j]) * inv_gamma)
            F_01 = math.exp((D[k, i, j + 1] - D[k, i, j] - weights[1]*C[k, i, j + 1]) * inv_gamma)
            F_11 = math.exp((D[k, i + 1, j + 1] - D[k, i, j] - weights[2]*C[k, i + 1, j + 1]) * inv_gamma)
            E[k, i, j] = E[k, i + 1, j] * F_10 + E[k, i, j + 1] * F_01 + E[k, i + 1, j + 1] * F_11

            G = weights[0] * math.exp(-(D[k, i - 1, j] - D[k, i, j] + weights[0] * C[k, i, j]) * inv_gamma) + \
                          weights[1] * math.exp(-(D[k, i, j - 1] - D[k, i, j] + weights[1] * C[k, i, j]) * inv_gamma) + \
                          weights[2] * math.exp(-(D[k, i - 1, j - 1] - D[k, i, j] + weights[2] * C[k, i, j]) * inv_gamma)

            H[k, i, j] = E[k,i,j] * G

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
class _weightedSDTW_CUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """
    
    @staticmethod
    def forward(ctx, C, gamma, weights):
        dev = C.device
        dtype = C.dtype
        gamma = torch.cuda.FloatTensor([gamma])

        B = C.shape[0]
        N = C.shape[2]
        M = C.shape[1]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        # Prepare the output array
        D = torch.ones((B, M + 2, N + 2), device=dev, dtype=dtype) * math.inf
        D[:, 0, 0] = 0

        # Dun the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_weightedSDTW_cuda[B, threads_per_block](cuda.as_cuda_array(C.detach()),
                                                   gamma.item(), M, N, n_passes,
                                                   cuda.as_cuda_array(D),
                                                   weights)
        ctx.save_for_backward(C, D.clone(), gamma, weights)
        return D[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        C, D, gamma, weights = ctx.saved_tensors

        B = C.shape[0]
        N = C.shape[2]
        M = C.shape[1]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        C_ = torch.zeros((B, M + 2, N + 2), dtype=dtype, device=dev)
        C_[:, 1:M + 1, 1:N + 1] = C

        D[:, :, -1] = -math.inf
        D[:, -1, :] = -math.inf
        D[:, -1, -1] = D[:, -2, -2]

        E = torch.zeros((B, M + 2, N + 2), dtype=dtype, device=dev)
        H = torch.zeros((B, M + 2, N + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        # Grid and block sizes are set same as done above for the forward() call
        compute_weightedSDTW_backward_cuda[B, threads_per_block](cuda.as_cuda_array(C_),
                                                            cuda.as_cuda_array(D),
                                                            1.0 / gamma.item(), M, N, n_passes,
                                                            cuda.as_cuda_array(E),
                                                            cuda.as_cuda_array(H),
                                                            weights)
        E = E[:, 1:M + 1, 1:N + 1]
        _weightedSDTW_CUDA.E = E
        
        H = H[:, 1:M + 1, 1:N + 1]
        _weightedSDTW_CUDA.H = H
        
        return grad_output.view(-1, 1, 1).expand_as(H) * H, None, None, None


# ----------------------------------------------------------------------------------------------------------------------
#
# The following is based on the CPU implementation https://github.com/Sleepwalking/pytorch-softdtw
#
# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_weightedSDTW(C, gamma, weights):
    B = C.shape[0]
    N = C.shape[2]
    M = C.shape[1]
    D = np.ones((B, M + 2, N + 2)) * np.inf
    D[:, 0, 0] = 0
    for b in prange(B):
        for j in range(1, N + 1):
            for i in range(1, M + 1):

                d0 = - (C[b, i-1, j-1]*weights[2] +  D[b, i - 1, j - 1]) / gamma
                d1 = - (C[b, i-1, j-1]*weights[0] +  D[b, i - 1, j]) / gamma
                d2 = - (C[b, i-1, j-1]*weights[1] +  D[b, i, j - 1]) / gamma
                dmax = max(max(d0, d1), d2)
                dsum = np.exp(d0 - dmax) + np.exp(d1 - dmax) + np.exp(d2 - dmax)
                D[b, i, j] = -gamma * (np.log(dsum) + dmax)
                
    return D

# ----------------------------------------------------------------------------------------------------------------------
@jit(nopython=True, parallel=True)
def compute_weightedSDTW_backward(C_, D, gamma, weights):
    B = C_.shape[0]
    N = C_.shape[2]
    M = C_.shape[1]
    C = np.zeros((B, M + 2, N + 2))
    E = np.zeros((B, M + 2, N + 2))
    
    H = np.zeros((B, M+2, N+2))
    
    C[:, 1:M + 1, 1:N + 1] = C_
    E[:, -1, -1] = 1
    D[:, :, -1] = -np.inf
    D[:, -1, :] = -np.inf
    D[:, -1, -1] = D[:, -2, -2]
    for k in prange(B):
        for j in range(N, 0, -1):
            for i in range(M, 0, -1):

                if np.isinf(D[k, i, j]):
                    D[k, i, j] = -np.inf

                ########################################################
                F_10 = np.exp((D[k, i + 1, j] - D[k, i, j] - weights[0]*C[k, i + 1, j]) / gamma)
                F_01 = np.exp((D[k, i, j + 1] - D[k, i, j] - weights[1]*C[k, i, j + 1]) / gamma)
                F_11 = np.exp((D[k, i + 1, j + 1] - D[k, i, j] - weights[2]*C[k, i + 1, j + 1]) / gamma)
                E[k, i, j] = E[k, i + 1, j] * F_10 + E[k, i, j + 1] * F_01 + E[k, i + 1, j + 1] * F_11
                
                G = weights[0] * np.exp(-(D[k, i - 1, j] - D[k, i, j] + weights[0] * C[k, i, j]) / gamma) + \
                              weights[1] * np.exp(-(D[k, i, j - 1] - D[k, i, j] + weights[1] * C[k, i, j]) / gamma) + \
                              weights[2] * np.exp(-(D[k, i - 1, j - 1] - D[k, i, j] + weights[2] * C[k, i, j]) / gamma)
                
                H[k, i, j] = E[k,i,j] * G
                
    return E[:, 1:M + 1, 1:N + 1], H[:, 1:M + 1, 1:N + 1]

# ----------------------------------------------------------------------------------------------------------------------
class _weightedSDTW(Function):
    """
    CPU implementation based on https://github.com/Sleepwalking/pytorch-softdtw
    """
    
    e_matrix = None # store expected alignment matrix for analysis purposes
    H_matrix=None

    @staticmethod
    def forward(ctx, C, gamma, weights):
        dev = C.device
        dtype = C.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype)  # dtype fixed
        C_ = C.detach().cpu().numpy()
        g_ = gamma.item()
        D = torch.Tensor(compute_weightedSDTW(C_, g_, weights.cpu().numpy())).to(dev).type(dtype)
        ctx.save_for_backward(C, D, gamma, weights)
        
        return D[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        C, D, gamma, weights = ctx.saved_tensors
        C_ = C.detach().cpu().numpy()
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        E, H = compute_weightedSDTW_backward(C_, D_, g_, weights.cpu().numpy())
        
        E = torch.Tensor(E).to(dev).type(dtype)
        H = torch.Tensor(H).to(dev).type(dtype)
        
        _weightedSDTW.E = E
        _weightedSDTW.H = H
        return grad_output.view(-1, 1, 1).expand_as(H) * H, None, None, None

# ----------------------------------------------------------------------------------------------------------------------
#
# A wrapper around cpu and gpu implementations
# 
# ----------------------------------------------------------------------------------------------------------------------

class weightedSDTW(torch.nn.Module):
    """
    The soft CTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda, gamma=1.0, dist_func=None, weights=[1,1,1]):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(weightedSDTW, self).__init__()
        self.gamma = gamma
        self.use_cuda = use_cuda
        self.dtw_class = None
        self.weights = weights

        # Set the distance function
        if dist_func is not None:
            self.dist_func = dist_func
        else:
            self.dist_func = weightedSDTW._euclidean_dist_func

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
                print("weightedSDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
                use_cuda = False

        # Finally, return the correct function
        self.dtw_class = _weightedSDTW_CUDA if use_cuda else _weightedSDTW
        return _weightedSDTW_CUDA.apply if use_cuda else _weightedSDTW.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        N = x.size(1)
        M = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(1).expand(-1, M, N, d)
        y = y.unsqueeze(2).expand(-1, M, N, d)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """
        
        # Format input to the right shape
        X = torch.squeeze(X, 1)
        Y = torch.squeeze(Y, 1)
        num_frames = X.shape[1]
        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(X, Y)
        
        C = self.dist_func(X, Y)         
        return torch.mean(func_dtw(C, self.gamma, self.weights))/num_frames