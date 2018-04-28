import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import Quantize

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.cnn1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.cnn2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc = nn.Linear(7*7*32, 10)
        self.batchnorm16 = nn.BatchNorm2d(16)
        self.batchnorm32 = nn.BatchNorm2d(32)
        self.maxpool2d = nn.MaxPool2d(2)

    def forward(self, x):
        layer1_out = self.maxpool2d(self._activation(self.batchnorm16(self.cnn1(x))))
        layer2_out = self.maxpool2d(self._activation(self.batchnorm32(self.cnn2(layer1_out))))
        out = layer2_out.view(layer2_out.size(0), -1)
        out = self.fc(out)
        return out

    def _QA(self, x):
        if self.config['abits'] <= 16:
            x = Quantize.A(x, self.config['abits'])
        return x

    def _QE(self, x):
        if self.config['ebits'] <= 16:
            x = Quantize.E(x, self.config['ebits'])
        return x

    def _activation(self, x):
        x = F.relu(x)
        x = self._QE(x)
        x = self._QA(x)
        return x
    def _conv(self, x, ksize, c_out, stride=1, padding='SAME', name='conv'):
        c_in = x.get_shape().as_list()[1]
        W = self._get_variable([ksize, ksize, c_in, c_out], name)
        x = torch.Conv2d(x, W, self._arr(stride), padding=padding, data_format='NCHW', name=name)
        return x

  def _get_variable(self, shape, name):
    print 'W:', self.W[-1].device, scope, shape,
    if Quantize.bitsW <= 16:
      # manually clip and quantize W if needed
      self.W_q_op.append(tf.assign(self.W[-1], Quantize.Q(self.W[-1], Quantize.bitsW)))
      self.W_clip_op.append(tf.assign(self.W[-1],Quantize.C(self.W[-1],Quantize.bitsW)))

        scale = Option.W_scale[len(self.W)-1]
        print 'Scale:%d' % scale
        self.W_q.append(Quantize.W(self.W[-1], scale))
        return self.W_q[-1]
    else:
        print ''
        return self.W[-1]
