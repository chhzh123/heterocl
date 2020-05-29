import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np
from ..utils import *
from .op import *
from .nn import pad, get_pad_tuple, simplify

dtype = hcl.Float()
qtype_bit = hcl.UInt(1)

def if_mac(y, x, in_h, in_w, pad_top, pad_left, pad_down, pad_right):
    return tvm.all(x >= pad_left, x < in_w - pad_right, y >= pad_top, y < in_h - pad_down)

def flatten(data, name="flatten"):
    ishape = data.shape
    dim = 1
    for i in range(1, len(ishape)):
        dim = dim * ishape[i]
    oshape = (ishape[0], dim)

    def unwrap(idx, shape): # channel first
        index = [idx % shape[0], idx / (shape[0]*shape[1]), (idx / shape[0]) % shape[1]]
        return index

    return hcl.compute(oshape, lambda i,j: data[tuple([i] + unwrap(j,ishape[1:]))],
        name=name,attrs=OrderedDict([('app_name',tvm.make.StringImm('flatten'))]),
        dtype=data.dtype)

def dense(data, weight, bias=None, use_relu=False, name="binary_dense"):
    assert len(
        data.shape) == 2 and len(
        weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = hcl.reduce_axis(0, in_dim)
    var_w = np.sqrt(2. / in_dim) # predefined constant
    # var_w = 1
    attrs = OrderedDict([
        ('k', in_dim),
        ('j', out_dim),
        ('i', batch),
        ('app_name', tvm.make.StringImm('mm'))])
    if bias is None:
        matmul = hcl.compute((batch, out_dim), lambda i, j: sum(
            tvm.all(data[i, k] == weight[j, k]), axis=k)
            * 2 - in_dim,
            name=name+"_matmul",
            attrs=attrs) # Data type needs to be specified!
    else:
        matmul = hcl.compute((batch, out_dim), lambda i, j: (hcl.sum(
            tvm.all(data[i, k] == weight[j, k]), axis=k, dtype=bias.dtype, name=name+"_sum")
            * 2 - in_dim) * var_w + bias[j],
            name=(name+"_matmul" if use_relu else name),
            attrs=attrs,
            dtype=bias.dtype)
    if use_relu:
        matmul = hcl.compute(
            (batch, out_dim),
            lambda i, j: hcl.select(matmul[i, j] > 0, 1, 0),
            name=name,
            attrs=attrs,
            dtype=qtype_bit
        )
    return matmul

def _popcount(num,name="popcnt"):
    out = hcl.scalar(0, name=name)
    with hcl.for_(0, 32) as i:
        # Bit selection operation
        out.v += num[i]
    return out.v

def packed_dense(data, weight, bias=None, use_relu=False, name="packed_binary_dense"):
    assert len(
        data.shape) == 2 and len(
        weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    assert "int" in data.dtype, "data type should be int or unsigned int"
    bitwidth = int(data.dtype.split("int")[-1])
    batch, in_dim = data.shape # in_dim has been packed
    out_dim, _ = weight.shape # only packed axis 1
    k = hcl.reduce_axis(0, in_dim)
    var_w = np.sqrt(2. / in_dim) # predefined constant
    # var_w = 1
    attrs = OrderedDict([
        ('k', in_dim),
        ('j', out_dim),
        ('i', batch),
        ('app_name', tvm.make.StringImm('mm'))])
    xor = hcl.compute((batch, out_dim, in_dim), lambda i, j, u:
            data[i, u] ^ weight[j, u],
            name=name+"_xor",
            attrs=attrs)
    popcnt = hcl.compute(xor.shape, lambda i, j, u:
            _popcount(xor[i, j, u],name=name+"_popcnt"),
            name=name+"_popcount",
            dtype=data.dtype)
    matmul = hcl.compute((batch, out_dim), lambda i, j:
            in_dim * bitwidth - 2 * sum(popcnt[i, j, k], axis=k),
            name=name+"_matmul",
            attrs=attrs,
            dtype=data.dtype) # Data type needs to be specified!
    if bias is not None:
        matmul = hcl.compute((batch, out_dim), lambda i, j:
                matmul[i, j] * var_w + bias[j],
                name=(name+"_bias" if use_relu else name),
                attrs=attrs,
                dtype=bias.dtype)
    def genpack(i, j):
        out = hcl.scalar(0, name=name+"_pack", dtype=data.dtype)
        with hcl.for_(0, bitwidth) as k:
            out[0][(k+1) : k] = hcl.select(matmul[i, j*bitwidth+k] > 0, 1, 0)
        return out[0]
    if use_relu:
        matmul = hcl.compute(
            (batch, out_dim // bitwidth),
            genpack,
            name=name,
            attrs=attrs,
            dtype=data.dtype
        )
    return matmul

def conv2d_nchw(
        Input,
        Filter,
        strides=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        out_dtype=None,
        name='binary_conv2d'):
    if out_dtype is None or out_dtype == '':
        out_dtype = hcl.Int()
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
    temp = pad(Input, pad_before, pad_after, name=name+"_pad")
    pad_in_height = in_height + pad_top + pad_down
    pad_in_width = in_width + pad_left + pad_right
    rc = hcl.reduce_axis(0, channel, name='rc')
    ry = hcl.reduce_axis(0, kernel_h, name='ry')
    rx = hcl.reduce_axis(0, kernel_w, name='rx')
    kernel_size = kernel_h * kernel_w
    out = hcl.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: hcl.sum(
            hcl.select(
                if_mac(yy+ry, xx+rx, pad_in_height, pad_in_width, pad_top, pad_left, pad_down, pad_right), # neglect padding pixels in mac
                tvm.all(temp[nn, rc,
                            yy * stride_h + ry * dilation_h,
                            xx * stride_w + rx * dilation_w] ==
                         Filter[ff, rc, ry, rx])
                * 2 - 1, # xnor
                0),
            axis=[rc, ry, rx], dtype=out_dtype, name=name+"_sum"),
            name=name,
            dtype=out_dtype)
    return out

def max_pool2d_nchw(
        data,
        pooling=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        layout='NCHW',
        name='binary_max_pool2d'):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    max = hcl.reducer(
        hcl.min_value(data.dtype),
        lambda x, y: tvm.make.Max(x, y),
        data.dtype)
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, channel, height, width = data.shape
    if len(padding) == 4:
        pad_top = padding[0]
        pad_left = padding[1]
        pad_bottom = padding[2]
        pad_right = padding[3]
    else:
        pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (pooling_h, pooling_w))
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_bottom, pad_right]
    if (pad_top,pad_left,pad_bottom,pad_right) != (0,0,0,0):
        data = pad(data, pad_before, pad_after, pad_value=hcl.min_value(data.dtype),name=name+"_pad")
    out_height = simplify(
        (height - pooling_h + pad_top + pad_bottom) // stride_h + 1)
    out_width = simplify(
        (width - pooling_w + pad_left + pad_right) // stride_w + 1)
    dheight = hcl.reduce_axis(0, pooling_h)
    dwidth = hcl.reduce_axis(0, pooling_w)
    return hcl.compute(
        (batch, channel, out_height, out_width),
        lambda i, c, h, w: hcl.select(max(data[i, c, h *
                                    stride_h +
                                    dheight, w *
                                    stride_w +
                                    dwidth], axis=[dheight, dwidth]) > 0,
                                    1,
                                    0),
        name=name, dtype=qtype_bit,
        attrs=OrderedDict([
            ('out_img_w', out_width),
            ('out_img_h', out_height),
            ('in_num', channel),
            ('kernel_h', pooling[1]),
            ('kernel_w', pooling[0]),
            ('stride_h', stride[1]),
            ('stride_w', stride[0]),
            ('app_name', tvm.make.StringImm('max_pool'))]))

def packed_max_pool2d_nchw(
        data,
        pooling=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        layout='NCHW',
        name='packed_binary_max_pool2d'):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    assert pooling == [2,2], "only support [2,2] padding now"
    max = hcl.reducer(
        hcl.min_value(data.dtype),
        lambda x, y: tvm.make.Max(x, y),
        data.dtype)
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, channel, height, width = data.shape
    bitwidth = int(data.dtype.split("int")[-1])
    if len(padding) == 4:
        pad_top, pad_left, pad_bottom, pad_right = padding
    else:
        pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (pooling_h, pooling_w))
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_bottom, pad_right]
    if (pad_top,pad_left,pad_bottom,pad_right) != (0,0,0,0):
        data = pad(data, pad_before, pad_after, pad_value=hcl.min_value(data.dtype),name=name+"_pad")
    out_height = simplify(
        (height - pooling_h + pad_top + pad_bottom) // stride_h + 1)
    out_width = simplify(
        (width - pooling_w + pad_left + pad_right) // stride_w + 1)
    dheight = hcl.reduce_axis(0, pooling_h)
    dwidth = hcl.reduce_axis(0, pooling_w)
    maxpool = hcl.compute(
        (batch, channel, out_height, out_width),
        lambda i, c, h, w:
            data[i, c, h * stride_h, w * stride_w] |
            data[i, c, h * stride_h, w * stride_w+1] |
            data[i, c, h * stride_h+1, w * stride_w] |
            data[i, c, h * stride_h+1, w * stride_w+1],
        name=name, dtype=hcl.UInt(bitwidth),
        attrs=OrderedDict([
            ('out_img_w', out_width),
            ('out_img_h', out_height),
            ('in_num', channel),
            ('kernel_h', pooling[1]),
            ('kernel_w', pooling[0]),
            ('stride_h', stride[1]),
            ('stride_w', stride[0]),
            ('app_name', tvm.make.StringImm('max_pool'))]))
    # return maxpool
    return hcl.compute((batch, channel * bitwidth, out_height, out_width),
        lambda i, c, h, w:
            maxpool[i, c // bitwidth, h, w][c % bitwidth],
        name=name+"_unpack",
        dtype=qtype_bit)

def batch_norm(
        data,
        gamma,
        beta,
        moving_mean,
        moving_var,
        M0=1,
        axis=1,
        epsilon=10**-5,
        center=1,
        scale=1,
        name="batch_norm"):
    if axis < 0:
        axis = len(data.shape) - 1
    mred = []
    vred = []
    size = 1.0
    for i in range(len(data.shape)):
        if not i == axis:
            mred.append(hcl.reduce_axis(0, data.shape[i], "mred" + str(i)))
            vred.append(hcl.reduce_axis(0, data.shape[i], "vred" + str(i)))
            size = size * data.shape[i]
    new_shape = (data.shape[axis],)

    def insert_axis(axis, red, *indices):
        idx = []
        cur_red = 0
        for i in range(len(data.shape)):
            if i == axis:
                idx.append(indices[0])
            else:
                idx.append(red[cur_red])
                cur_red = cur_red + 1
        return tuple(idx)

    def get_axis(axis, *indices):
        indices = list(indices[0])
        return (indices[axis],)

    var_w = np.sqrt(2. / (9. * M0)) # predefined constant
    out = hcl.compute(data.shape, lambda *x: hcl.select(
                    (data[x] * var_w - moving_mean[get_axis(axis, x)]) /
                    (hcl.sqrt(moving_var[get_axis(axis, x)] + epsilon)) * gamma[get_axis(axis, x)]
                    + beta[get_axis(axis, x)] > 0,
                    1, # quantize
                    0), name=name, dtype=qtype_bit)
    return out, moving_mean, moving_var

def batch_norm_threshold(
        data,
        threshold,
        name="batch_norm_threshold"):
    batch, channel, out_height, out_width = data.shape
    bitwidth = channel # pack channels
    def genpack(i, c, h, w):
        out = hcl.scalar(0, name=name+"_pack", dtype=hcl.UInt(bitwidth))
        with hcl.for_(0, bitwidth) as k:
            out[0][(k+1) : k] = hcl.select(data[i, c*bitwidth+k, h, w] > threshold[c*bitwidth+k, h, w], 1, 0)
        return out[0]
    return hcl.compute((batch, channel//bitwidth, out_height, out_width),
                        genpack, name=name, dtype=hcl.UInt(bitwidth))