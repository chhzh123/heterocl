import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np
from ..utils import *
from .op import *
from .nn import pad, get_pad_tuple, simplify

dtype = hcl.Float()

def if_mac(y, x, out_h, out_w, pad_top, pad_left, pad_down, pad_right):
    return tvm.all(x >= pad_left, x < out_w - pad_right, y >= pad_top, y < out_h - pad_down)

def conv2d_nchw(
        Input,
        Filter,
        strides=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        out_dtype=None,
        name='binary_conv2d'):
    if out_dtype is None or out_dtype == '':
        out_dtype = Input.dtype
    assert isinstance(strides, int) or len(strides) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify(
        (in_height -
         dilated_kernel_h +
         pad_top +
         pad_down) //
        stride_h +
        1)
    out_width = simplify(
        (in_width -
         dilated_kernel_w +
         pad_left +
         pad_right) //
        stride_w +
        1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad(Input, pad_before, pad_after, name="pad_temp")
    pad_in_height = in_height + pad_top + pad_down
    pad_in_width = in_width + pad_left + pad_right
    rc = hcl.reduce_axis(0, channel, name='rc')
    ry = hcl.reduce_axis(0, kernel_h, name='ry')
    rx = hcl.reduce_axis(0, kernel_w, name='rx')
    kernel_size = kernel_h * kernel_w
    out = hcl.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: hcl.sum(
            tvm.select(if_mac(yy+ry, xx+rx, pad_in_height, pad_in_width, pad_top, pad_left, pad_down, pad_right),
            ((1-(temp[nn, rc, yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w] ^
            Filter[ff, rc, ry, rx])) << 1) - 1, # xnor
            0), # neglect padding pixels in mac
            axis=[rc, ry, rx], dtype=out_dtype),
            name=name,
            dtype=out_dtype)
    return out