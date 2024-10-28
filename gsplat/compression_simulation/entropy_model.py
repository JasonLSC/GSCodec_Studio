import torch
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np
from torch.distributions.uniform import Uniform
from torch import Tensor
from typing_extensions import List



class Entropy_factorized(nn.Module):
    def __init__(self, channel=32, init_scale=10, filters=(3, 3, 3), # (3, 3, 3)
                 likelihood_bound=1e-6, tail_mass=1e-9, optimize_integer_offset=True, Q=1):
        super(Entropy_factorized, self).__init__()
        self.filters = tuple(int(t) for t in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)
        self.optimize_integer_offset = bool(optimize_integer_offset)
        self.Q = Q
        if not 0 < self.tail_mass < 1:
            raise ValueError(
                "`tail_mass` must be between 0 and 1")
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1.0 / (len(self.filters) + 1))
        self._matrices = nn.ParameterList([])
        self._bias = nn.ParameterList([])
        self._factor = nn.ParameterList([])
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix = nn.Parameter(torch.FloatTensor(
                channel, filters[i + 1], filters[i]))
            self.matrix.data.fill_(init)
            self._matrices.append(self.matrix)
            self.bias = nn.Parameter(
                torch.FloatTensor(channel, filters[i + 1], 1))
            noise = np.random.uniform(-0.5, 0.5, self.bias.size())
            noise = torch.FloatTensor(noise)
            self.bias.data.copy_(noise)
            self._bias.append(self.bias)
            if i < len(self.filters):
                self.factor = nn.Parameter(
                    torch.FloatTensor(channel, filters[i + 1], 1))
                self.factor.data.fill_(0.0)
                self._factor.append(self.factor)
        
        # for jit
        self.register_buffer('filters_len', torch.tensor(len(self.filters)))
        self.register_buffer('factor_len', torch.tensor(len(self._factor)))

        self.likelihood_lower_bound = LowerBound(likelihood_bound)

    # default code
    def _logits_cumulative(self, logits, stop_gradient):
        import pdb; pdb.set_trace()
        for i in range(len(self.filters) + 1):
            matrix = nnf.softplus(self._matrices[i])
            if stop_gradient:
                matrix = matrix.detach()
            # print('dqnwdnqwdqwdqwf:', matrix.shape, logits.shape)
            logits = torch.matmul(matrix, logits)
            bias = self._bias[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias
            if i < len(self._factor):
                factor = nnf.tanh(self._factor[i])
                if stop_gradient:
                    factor = factor.detach()
                logits += factor * nnf.tanh(logits)
        return logits

    def forward(self, x, Q=None):
        # x: [N, C], quantized
        if Q is None:
            Q = self.Q
        else:
            Q = torch.tensor([Q], device=x.device)
        x = x.unsqueeze(1).permute((2, 1, 0)).contiguous()  # [C, 1, N]
        # print('dqwdqwdqwdqwfqwf:', x.shape, Q.shape)
        lower = self._logits_cumulative(x - 0.5*Q.detach(), stop_gradient=False)
        upper = self._logits_cumulative(x + 0.5*Q.detach(), stop_gradient=False)
        sign = -torch.sign(torch.add(lower, upper))
        sign = sign.detach()
        likelihood = torch.abs(
            nnf.sigmoid(sign * upper) - nnf.sigmoid(sign * lower))
        # likelihood = Low_bound.apply(likelihood)
        likelihood = self.likelihood_lower_bound(likelihood)
        bits = -torch.log2(likelihood)  # [C, 1, N]
        bits = bits.permute((2, 1, 0)).squeeze(1).contiguous()
        return bits

class Entropy_factorized_optimized(nn.Module):
    def __init__(self, channel=32, init_scale=10, filters=(3, 3, 3), # (3, 3, 3)
                 likelihood_bound=1e-6, tail_mass=1e-9, optimize_integer_offset=True, Q=1):
        super(Entropy_factorized_optimized, self).__init__()
        self.channel = channel
        self.filters = tuple(int(t) for t in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)
        self.optimize_integer_offset = bool(optimize_integer_offset)
        self.Q = Q
        if not 0 < self.tail_mass < 1:
            raise ValueError(
                "`tail_mass` must be between 0 and 1")
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1.0 / (len(self.filters) + 1))
        self._matrices = nn.ParameterList([])
        self._bias = nn.ParameterList([])
        self._factor = nn.ParameterList([])
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix = nn.Parameter(torch.FloatTensor(
                channel, filters[i + 1], filters[i]))
            self.matrix.data.fill_(init)
            self._matrices.append(self.matrix)
            self.bias = nn.Parameter(
                torch.FloatTensor(channel, filters[i + 1], 1))
            noise = np.random.uniform(-0.5, 0.5, self.bias.size())
            noise = torch.FloatTensor(noise)
            self.bias.data.copy_(noise)
            self._bias.append(self.bias)
            if i < len(self.filters):
                self.factor = nn.Parameter(
                    torch.FloatTensor(channel, filters[i + 1], 1))
                self.factor.data.fill_(0.0)
                self._factor.append(self.factor)
        
        # for jit
        self.register_buffer('filters_len', torch.tensor(len(self.filters)))
        self.register_buffer('factor_len', torch.tensor(len(self._factor)))

        self.likelihood_lower_bound = LowerBound(likelihood_bound)

    # default code
    def _logits_cumulative(self, logits, stop_gradient):
        import pdb; pdb.set_trace()
        for i in range(len(self.filters) + 1):
            matrix = nnf.softplus(self._matrices[i])
            if stop_gradient:
                matrix = matrix.detach()
            # print('dqnwdnqwdqwdqwf:', matrix.shape, logits.shape)
            logits = torch.matmul(matrix, logits)
            bias = self._bias[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias
            if i < len(self._factor):
                factor = nnf.tanh(self._factor[i])
                if stop_gradient:
                    factor = factor.detach()
                logits += factor * nnf.tanh(logits)
        return logits

    def forward(self, x, Q=None):
        # x: [N, C], quantized
        if Q is None:
            Q = self.Q
        else:
            Q = torch.tensor([Q], device=x.device)

        # [N, C] -> [C, 1, N], batch维度移到最后以提高内存访问效率
        x = x.t().unsqueeze(1)

        # 预计算公共部分
        half_Q = 0.5 * Q.detach()
        x_lower = x - half_Q
        x_upper = x + half_Q
        
        stacked_inputs = torch.cat([x_lower, x_upper], dim=0)  # [2C, 1, N]
        logits = stacked_inputs
        
        for i in range(len(self.filters) + 1):
            matrix = nnf.softplus(self._matrices[i])  # [C, filters[i+1], filters[i]]
            matrix = matrix.repeat(2, 1, 1)  # [2*C, filters[i+1], filters[i]]
            
            logits = torch.bmm(matrix, logits)  # [2*C, filters[i+1], N]
            logits = logits + self._bias[i].repeat(2, 1, 1) 
            
            if i < len(self._factor):
                factor = nnf.tanh(self._factor[i].repeat(2, 1, 1))
                logits += factor * nnf.tanh(logits)
        
        lower, upper = logits[0:self.channel], logits[self.channel:self.channel*2]

        sign = -(lower + upper).sign()
        likelihood = torch.abs(nnf.sigmoid(sign * upper) - nnf.sigmoid(sign * lower))
        likelihood = self.likelihood_lower_bound(likelihood)
        
        bits = -torch.log2(likelihood)
        return bits.permute(2, 1, 0).squeeze(1)


class Entropy_gaussian(nn.Module):
    def __init__(self, Q=1, likelihood_bound=1e-6):
        super(Entropy_gaussian, self).__init__()
        self.Q = Q
        self.likelihood_lower_bound = LowerBound(likelihood_bound)
    def forward(self, x, mean, scale, Q=None):
        if Q is None:
            Q = self.Q
        scale = torch.clamp(scale, min=1e-9)
        m1 = torch.distributions.normal.Normal(mean, scale)
        lower = m1.cdf(x - 0.5*Q)
        upper = m1.cdf(x + 0.5*Q)
        likelihood = torch.abs(upper - lower)
        # likelihood = Low_bound.apply(likelihood)
        likelihood = self.likelihood_lower_bound(likelihood)
        bits = -torch.log2(likelihood)
        return bits


# class Low_bound(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         ctx.save_for_backward(x)
#         x = torch.clamp(x, min=1e-6)
#         return x

#     @staticmethod
#     def backward(ctx, g):
#         x, = ctx.saved_tensors
#         grad1 = g.clone()
#         grad1[x < 1e-6] = 0
#         pass_through_if = np.logical_or(
#             x.cpu().numpy() >= 1e-6, g.cpu().numpy() < 0.0)
#         t = torch.Tensor(pass_through_if+0.0).cuda()
#         return grad1 * t
    

def lower_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.max(x, bound)


def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = (x >= bound) | (grad_output < 0)
    return pass_through_if * grad_output, None


class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)

class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)