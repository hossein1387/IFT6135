import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import numpy as np


def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(0) == 3
    t = torch.cat([w[-2:], w, w[:2]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c[1:-1]


class NTMMemory(nn.Module):
    """Memory bank for NTM."""
    def __init__(self, N, M):
        """Initialize the NTM Memory matrix.
        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.
        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        """
        super(NTMMemory, self).__init__()

        self.N = N
        self.M = M

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        self.register_buffer('mem_bias', Variable(torch.Tensor(N, M)))

        # Initialize memory bias
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform(self.mem_bias, -stdev, stdev)

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)

    def size(self):
        return self.N, self.M

    def read(self, w):
        """Read from memory (according to section 3.1)."""
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w, e, a):
        """write to memory (according to section 3.2)."""
        self.prev_mem = self.memory
        self.memory = Variable(torch.Tensor(self.batch_size, self.N, self.M))
        for b in range(self.batch_size):
            erase = torch.ger(w[b], e[b])
            add = torch.ger(w[b], a[b])
            self.memory[b] = self.prev_mem[b] * (1 - erase) + add

    def address(self, k, beta, g, s, Y, w_prev):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.
        :param k: The key vector.
        :param beta: The key strength (focus).
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param Y: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        """
        # Content focus
        wc = self._similarity(k, beta)

        # Location focus
        wg = self._interpolate(w_prev, wc, g)
        w = self._shift(wg, s)
        w = self._sharpen(w, Y)

        return w

    def _similarity(self, k, beta):
        k = k.view(self.batch_size, 1, -1)
        w = F.softmax(beta * F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    def _shift(self, wg, s):
        result = Variable(torch.zeros(wg.size()))
        for b in range(self.batch_size):
            result[b] = _convolve(wg[b], s[b])
        return result

    def _sharpen(self, w, Y):
        w = w ** Y
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w