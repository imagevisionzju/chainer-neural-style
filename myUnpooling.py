import numpy as np
import os
from chainer import cuda, Variable
from chainer import function
import collections

def makepair(x):
    if isinstance(x, collections.Iterable):
        return x
    return x, x

#hl: outh,outw = inh*stridey,inw * stridex
class myUnpooling2D(function.Function):

    """Unpooling over a set of 2d planes."""

    def __init__(self,stride):
        self.outh, self.outw = (None, None)
        self.sy, self.sx = makepair(stride)

    def check_type_forward(self, in_types):
        '''
        n_in = in_types.size()
        type_check.expect(n_in == 1)
        x_type = in_types[0]

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim == 4,
        )

        if self.outh is not None:
            expected_h = conv.get_conv_outsize(
                self.outh, self.kh, self.sy, self.ph, cover_all=self.cover_all)
            type_check.expect(x_type.shape[2] == expected_h)
        if self.outw is not None:
            expected_w = conv.get_conv_outsize(
                self.outw, self.kw, self.sx, self.pw, cover_all=self.cover_all)
            type_check.expect(x_type.shape[3] == expected_w)
        '''

    def forward(self, x):
        h, w = x[0].shape[2:]
        if self.outh is None:
            self.outh = self.sy * h
        if self.outw is None:
            self.outw = self.sx * w
        xp = cuda.get_array_module(*x)

        xt = x[0].reshape(x[0].shape[0:2] + (h,1,w,1))
        y = xp.tile(xt,(self.sy, 1, self.sx))
        y = y.reshape(x[0].shape[0:2] + (self.outh,self.outw))

        return y,

    def backward(self, x, gy):
        xp = cuda.get_array_module(*gy)
        b,c,h,w = gy[0].shape
        nshape = gy[0].shape[0:2] + (h/self.sy,self.sy,w/self.sx,self.sx)
        gx = gy[0].reshape(nshape)
        gx = xp.mean(gx,axis=(3,5))
        return gx,


def my_unpooling_2d(x,stride):
    """Inverse operation of pooling for 2d array.

    This function acts similarly to :class:`~functions.Deconvolution2D`, but
    it spreads input 2d array's value without any parameter instead of
    computing the inner products.

    Args:
        x (~chainer.Variable): Input variable.
        ksize (int or pair of ints): Size of pooling window. ``ksize=k`` and
            ``ksize=(k, k)`` are equivalent.
        stride (int, pair of ints or None): Stride of pooling applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent. If ``None`` is
            specified, then it uses same stride as the pooling window size.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return myUnpooling2D( stride)(x)
