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

def get_scale(wbits, limit):
    scale = 1.0
    if wbits < 32:
      beta = 1.5
      Wm = beta / S(wbits)
      scale = 2 ** round(math.log(Wm / limit, 2.0))
      scale = scale if scale > 1 else 1.0
      limit = Wm if Wm > limit else limit
    return scale, limit


class WAGEWQuant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, wbits, n):
        factor=2.0
        # import ipdb as pdb; pdb.set_trace()
        limit = math.sqrt(1.3 * factor / n)
        scale, limit = get_scale(wbits, limit)
        ctx.save_for_backward(x)
        y = Q(C(x, wbits), wbits)
        # we scale W in QW rather than QA for simplicity
        if scale > 1.8:
          y = y/scale
        return x + (y - x)

    @staticmethod
    def backward(ctx, dx):
        rbits = 16
        gbits = 8
        LR = 1e-4
        x, = ctx.saved_variables
        if gbits > 15:
          return x
        xmax = torch.max(torch.abs(x))
        x = x / Shift(xmax)
        norm = Q(LR * x, rbits)
        norm_sign = torch.sign(norm)
        norm_abs = torch.abs(norm)
        norm_int = torch.floor(norm_abs)
        norm_float = norm_abs - norm_int
        rand_float = torch.randn(x.size())
        norm = norm_sign * ( norm_int + 0.5 * (torch.sign(norm_float - rand_float) + 1) )
        g =  norm / S(gbits)
        return dx*g, None, None
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
    return x + (y - x)  # skip derivation of Quantize, but keep Clip


class BNNSign(torch.autograd.Function):
    """
    BinaryNet q = Sign(r) with gradient override.
    Equation (1) and (4) of https://arxiv.org/pdf/1602.02830.pdf
    """
    
    @staticmethod
    def forward(ctx, x):
        import ipdb as pdb; pdb.set_trace()
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
