import torch 
import math

def S(bits):
    return 2.0 ** (bits - 1)

def Shift(x):
    # import ipdb as pdb; pdb.set_trace()
    return 2 ** torch.round(torch.log(x) / math.log(2))

def C(x, bits=32):
    if bits > 15 or bits == 1:
        delta = 0.
    else:
        delta = 1. / S(bits)
    MAX = +1 - delta
    MIN = -1 + delta
    x = torch.clamp(x, min=MIN, max=MAX)
    return x

def Q(x, bits):
    if bits > 15:
        return x
    elif bits == 1:  # BNN
        return torch.sign(x)
    else:
        SCALE = S(bits)
        return torch.round(x * SCALE) / SCALE

def E(x, bitsE):
    if bitsE > 15:
        return x
    else:
        xmax = torch.max(torch.abs(x))
        xmax_shift = Shift(xmax)
    return Q(C( x /xmax_shift, bitsE), bitsE)

def W(x,scale = 1.0):
    y = Q(C(x, bitsW), bitsW)
    # we scale W in QW rather than QA for simplicity
    if scale > 1.8:
      y = y/scale
    # if bitsG > 15:
      # when not quantize gradient, we should rescale the scale factor in backprop
      # otherwise the learning rate will have decay factor scale
      # x = x * scale
    return x + (y - x).detach()  # skip derivation of Quantize and Clip

def A(x, bitsA):
    x = C(x, bitsA)
    y = Q(x, bitsA)
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

