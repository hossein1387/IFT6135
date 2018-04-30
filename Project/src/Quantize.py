import torch 
import math

def S(bits):
    return 2.0 ** (bits - 1)

def Shift(x):
    # import ipdb as pdb; pdb.set_trace()
    return 2 ** torch.round(torch.log(x) / math.log(2))

class WAGEClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        delta = 1. / S(bits)
        MAX = +1 - delta
        MIN = -1 + delta
        x = torch.clamp(x, min=MIN, max=MAX)
        return x
    
    @staticmethod
    def backward(ctx, dx):
        x, = ctx.saved_variables
        
        gt1  = x > (1 - delta)
        lsm1 = x < (-1 + delta)
        gi   = 1-gt1.float()-lsm1.float()
        return gi*dx
wage_clip = WAGEClip.apply


class WAGEWQuant(torch.autograd.Function):

def W(x, wbits, scale = 1.0):
    y = Q(wage_clip(x, wbits), wbits)
    # we scale W in QW rather than QA for simplicity
    if scale > 1.8:
      y = y/scale
    # if bitsG > 15:
      # when not quantize gradient, we should rescale the scale factor in backprop
      # otherwise the learning rate will have decay factor scale
      # x = x * scale
    return x + (y - x).detach()  # skip derivation of Quantize and Clip
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        delta = 1. / S(bits)
        MAX = +1 - delta
        MIN = -1 + delta
        x = torch.clamp(x, min=MIN, max=MAX)
        return x
    
    @staticmethod
    def backward(ctx, dx):
        x, = ctx.saved_variables
        
        gt1  = x > (1 - delta)
        lsm1 = x < (-1 + delta)
        gi   = 1-gt1.float()-lsm1.float()
        return gi*dx
w_quant = WAGEWQuant.apply


def Q(x, bits):
        SCALE = S(bits)
        return torch.round(x * SCALE) / SCALE

def E(x, bitsE):
    if bitsE > 15:
        return x
    else:
        xmax = torch.max(torch.abs(x))
        xmax_shift = Shift(xmax)
    return Q(C( x /xmax_shift, bitsE), bitsE)

def A(x, abits):
    x = C(x, abits)
    y = Q(x, abits)
    return x + (y - x).detach()  # skip derivation of Quantize, but keep Clip

def G(x):
  with tf.name_scope('QG'):
    if bitsG > 15:
      return x
    else:
      if x.name.lower().find('batchnorm') > -1:
        return x  # batch norm parameters, not quantize now

      xmax = tf.reduce_max(tf.abs(x))
      x = x / Shift(xmax)

      norm = Q(LR * x, bitsR)

      norm_sign = tf.sign(norm)
      norm_abs = tf.abs(norm)
      norm_int = tf.floor(norm_abs)
      norm_float = norm_abs - norm_int
      rand_float = tf.random_uniform(x.get_shape(), 0, 1)
      norm = norm_sign * ( norm_int + 0.5 * (tf.sign(norm_float - rand_float) + 1) )

      return norm / S(bitsG)


class BNNSign(torch.autograd.Function):
    """
    BinaryNet q = Sign(r) with gradient override.
    Equation (1) and (4) of https://arxiv.org/pdf/1602.02830.pdf
    """
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()
    
    @staticmethod
    def backward(ctx, dx):
        x, = ctx.saved_variables
        
        gt1  = x > +1
        lsm1 = x < -1
        gi   = 1-gt1.float()-lsm1.float()
        
        return gi*dx

bnn_sign = BNNSign.apply
