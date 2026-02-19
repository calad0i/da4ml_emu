import typing
from math import log2

import hls4ml.model.layers
import numpy as np
from da4ml.trace import FixedVariableArray
from da4ml.trace.ops import relu
from hls4ml.model.layers import Layer
from hls4ml.model.optimizer.passes.bit_exact import stride_arrs
from hls4ml.model.optimizer.passes.hgq_proxy_model import FixedPointQuantizer
from hls4ml.model.types import FixedPrecisionType

from ._base import BasicDispatcher, to_quantizer

T = typing.TypeVar('T', FixedVariableArray, np.ndarray)


def hls4ml_dense(
    w: np.ndarray,
    b: np.ndarray | int | None,
    inp: T,
    accum_p: FixedPrecisionType | None,
    optimize: bool,
) -> T:
    accum_q = to_quantizer(accum_p)
    b = 0 if b is None else b
    if optimize:
        assert accum_p is None or (str(accum_p.saturation_mode) == 'WRAP' and str(accum_p.rounding_mode) == 'TRN'), (
            'Impossible to have DA optimization while bit-exact to non-trivial accum mode'
        )
        return accum_q(inp @ w + b)
    out = accum_q(b)

    for i in range(w.shape[0]):
        _out = accum_q(inp[..., i, None] * w[i, :])
        out = accum_q(out + _out)
    return out


class _Dense(BasicDispatcher):
    layer_type = hls4ml.model.layers.Dense

    def call(self, layer: Layer, inputs: list[FixedVariableArray], optimize: bool) -> FixedVariableArray:
        assert len(inputs) == 1
        inp = inputs[0]
        w, b = self.get_quantized(layer, 'weight'), self.get_quantized(layer, 'bias')

        accum_p = layer.attributes.attributes['accum_t'].precision
        if layer.get_attr('strategy', None) == 'distributed_arithmetic':
            accum_p = None
        return hls4ml_dense(w, b, inp, accum_p, optimize)


class _ConvXD(BasicDispatcher):
    layer_type = hls4ml.model.layers.Conv1D | hls4ml.model.layers.Conv2D

    def call(self, layer: Layer, inputs: list[FixedVariableArray], optimize: bool) -> FixedVariableArray:
        assert len(inputs) == 1
        inp = inputs[0]

        w, b = self.get_quantized(layer, 'weight'), self.get_quantized(layer, 'bias')
        *px_in, ch_in, ch_out = w.shape
        pads = ((layer.attributes['pad_left'], layer.attributes['pad_right']),)
        if layer.class_name.endswith('2D'):
            pads = (layer.attributes['pad_top'], layer.attributes['pad_bottom']), *pads

        if layer.attributes.attributes['data_format'] == 'channels_first':
            inp = np.moveaxis(inp, 0, -1)  # type: ignore

        pads = pads + ((0, 0),)
        inp = np.pad(inp, pads, mode='constant', constant_values=0)  # type: ignore

        inp_col = np.lib.stride_tricks.sliding_window_view(  # type: ignore
            inp,  # type: ignore
            px_in,
            axis=tuple(range(len(px_in))),  # type: ignore
        )
        inp_col = np.moveaxis(inp_col, len(px_in), -1).reshape(*inp_col.shape[: len(px_in)], -1)

        layer.get_output_variable().__dict__

        inp_col = stride_arrs(layer, inp_col)[0]

        accum_p = layer.attributes.attributes['accum_t'].precision
        if layer.get_attr('strategy', None) == 'distributed_arithmetic':
            accum_p = None
        _w = w.reshape(-1, w.shape[-1])
        out = hls4ml_dense(_w, b, inp_col, accum_p, optimize)  # type: ignore

        if layer.attributes.attributes['data_format'] == 'channels_first':
            out = np.moveaxis(out, -1, 0)  # type: ignore

        return out  # type: ignore


class _Activation(BasicDispatcher):
    layer_type = hls4ml.model.layers.Activation

    def call(self, layer: Layer, inputs: list[FixedVariableArray], optimize: bool) -> FixedVariableArray:
        assert len(inputs) == 1
        inp = inputs[0]

        fn_name = layer.attributes['activation']
        if fn_name == 'relu':
            return relu(inp)
        elif fn_name == 'linear':
            return inp

        table_size = layer.attributes['table_size']
        _B = log2(table_size)
        B = int(_B)
        assert B == _B, 'table_size must be a power of 2'
        k = int(inp.kif[0].any())
        b = B - k
        match fn_name:
            case 'sigmoid':
                fn = lambda x: 1 / (1 + np.exp(-x))
                f = b - 4  # int data_round = in_data[j] * CONFIG_T::table_size / 16;
            case 'tanh':
                fn = np.tanh
                f = b - 3  # int data_round = in_data[j] * CONFIG_T::table_size / 8;
            case 'softplus':
                fn = lambda x: np.log(1 + np.exp(x))
                f = b - 4  # data_round = data[ii] * CONFIG_T::table_size / 16;
            case 'softsign':
                fn = lambda x: x / (1 + np.abs(x))
                f = b - 4  # data_round = data[ii] * CONFIG_T::table_size / 16;
            case 'elu':
                fn = lambda x: x if x >= 0 else np.exp(x) - 1
                b += 1  # lazy to isolate negative range
                f = b - 3
            case _:
                raise NotImplementedError(f'Activation function {fn_name} not implemented')

        i = b - f
        inp = inp.quantize(k, i, f)
        out = inp.apply(fn)

        table_p = layer.attributes['table_t'].precision
        out = to_quantizer(table_p)(out)
        return out


class _Softmax(BasicDispatcher):
    layer_type = hls4ml.model.layers.Softmax

    def call(self, layer: Layer, inputs: list[FixedVariableArray], optimize: bool) -> FixedVariableArray:
        assert len(inputs) == 1
        inp = inputs[0]

        impl = layer.attributes['implementation']

        if impl == 'argmax':
            return np.max(inp, axis=-1)  # type: ignore
        assert impl in ('stable', 'latency')

        _table_size = layer.attributes['table_size']
        exp_table_size = layer.attributes.get('exp_table_size', _table_size)
        inv_table_size = layer.attributes.get('inv_table_size', _table_size)
        _B_exp, _B_inv = log2(int(exp_table_size)), log2(int(inv_table_size))
        B_exp, B_inv = int(_B_exp), int(_B_inv)
        assert B_exp == _B_exp and B_inv == _B_inv, 'exp_table_size and inv_table_size must be powers of 2'

        exp_table_q = to_quantizer(layer.attributes['exp_table_t'].precision)
        inv_table_q = to_quantizer(layer.attributes['inv_table_t'].precision)

        accum_q = to_quantizer(layer.attributes['accum_t'].precision)
        exp_scale = layer.attributes.get('exp_scale', 1.0)

        if impl == 'stable':
            if 'inp_norm_t' in layer.attributes:
                inp_norm_p: FixedPrecisionType = layer.attributes['inp_norm_t'].precision
            else:
                input_t: FixedPrecisionType = layer.get_input_variable().type.precision
                inp_norm_p = FixedPrecisionType(input_t.width, input_t.integer, False)
            inp_norm_q = to_quantizer(inp_norm_p)
            _max = np.max(inp, axis=-1, keepdims=True)  # type: ignore
            inp_norm = inp_norm_q(_max - inp)
            k, I = inp_norm_p.signed, inp_norm_p.integer
            i = I - k
            f = B_exp - I
            _exp = inp_norm.quantize(k, i, f).apply(lambda x: np.exp(-x * exp_scale))
        else:
            inp_p: FixedPrecisionType = layer.get_input_variable().type.precision
            k, I = inp_p.signed, inp_p.integer
            i = I - k
            f = B_exp - I
            _exp = inp.quantize(k, i, f).apply(lambda x: np.exp(x * exp_scale))
        _exp = accum_q(exp_table_q(_exp))
        _sum = accum_q(np.sum(_exp, axis=-1, keepdims=True))

        inp_inp_t: FixedPrecisionType = layer.attributes['inv_inp_t'].precision
        k, I = inp_inp_t.signed, inp_inp_t.integer
        i = I - k
        f = B_inv - I
        inv = inv_table_q(_sum.quantize(k, i, f).apply(lambda x: 1 / (x + 1e-12)))
        return _exp * inv


class _EinsumDense(BasicDispatcher):
    layer_type = hls4ml.model.layers.EinsumDense

    def call(self, layer: Layer, inputs: list[FixedVariableArray], optimize: bool) -> FixedVariableArray:
        assert len(inputs) == 1
        accum_p: FixedPrecisionType = layer.attributes.attributes['accum_t'].precision
        assert str(accum_p.saturation_mode) == 'WRAP' and str(accum_p.rounding_mode) == 'TRN', (
            f'Non-trivial accum mode not supported for EinsumDense (layer name: {layer.name})'
        )
        out_interpert_shape = layer.attributes['out_interpert_shape']
        inp_tpose_idxs = layer.attributes['inp_tpose_idxs']
        out_tpose_idxs = layer.attributes['out_tpose_idxs']
        L0 = layer.attributes['n_free_data']
        L1 = layer.attributes['n_free_kernel']
        I = layer.attributes['n_inplace']
        C = layer.attributes['n_contract']

        input0 = inputs[0].transpose(inp_tpose_idxs).ravel()
        input1 = self.get_quantized(layer, 'weight').ravel()
        out_dtype = object if input0.dtype == object or input1.dtype == object else np.float64
        output = np.zeros(L0 * L1 * I, dtype=out_dtype)

        for l0 in range(L0):
            for i in range(I):
                A = input0[(i * L0 + l0) * C : (i * L0 + l0 + 1) * C]
                B = input1[i * L1 * C : (i + 1) * L1 * C].reshape((C, L1))
                _r = hls4ml_dense(B, None, A, accum_p, optimize=optimize)
                output[(i * L0 + l0) * L1 : (i * L0 + l0 + 1) * L1] = _r

        output = output + self.get_quantized(layer, 'bias').ravel()
        output = output.reshape(out_interpert_shape).transpose(out_tpose_idxs)
        output = FixedVariableArray(output, solver_options=input0.solver_options)
        output = to_quantizer(accum_p)(output)
        return output


class _Einsum(BasicDispatcher):
    layer_type = hls4ml.model.layers.Einsum

    def call(self, layer: Layer, inputs: list[FixedVariableArray], optimize: bool) -> FixedVariableArray:
        assert len(inputs) == 2
        equation = layer.attributes['equation']
        return np.einsum(equation, inputs[0], inputs[1])  # type: ignore


class _Reshape(BasicDispatcher):
    layer_type = hls4ml.model.layers.Reshape

    def call(self, layer: Layer, inputs: list[FixedVariableArray], optimize: bool) -> FixedVariableArray:
        assert len(inputs) == 1
        inp = inputs[0]
        return inp.reshape(layer.attributes['target_shape'])


class _BatchNorm(BasicDispatcher):
    layer_type = hls4ml.model.layers.BatchNormalization

    def call(self, layer: Layer, inputs: list[FixedVariableArray], optimize: bool) -> FixedVariableArray:
        assert len(inputs) == 1
        scale, bias = self.get_quantized(layer, 'scale'), self.get_quantized(layer, 'bias')
        accum_p = layer.attributes.attributes['accum_t'].precision
        return to_quantizer(accum_p)(inputs[0] * scale + bias)  # type: ignore


class _Quantizer(BasicDispatcher):
    layer_type = FixedPointQuantizer

    def call(self, layer: FixedPointQuantizer, inputs: list[FixedVariableArray], optimize: bool) -> FixedVariableArray:
        assert len(inputs) == 1
        inp = inputs[0]
        k, b, i = layer.mask_kbi
        k, i, f = k, b - k, b - i
        RND, SAT = layer.RND, layer.SAT
        k = k[0] if np.ndim(k) > 0 else k
        i = i[0] if np.ndim(i) > 0 else i
        f = f[0] if np.ndim(f) > 0 else f
        return inp.quantize(k, i, f, round_mode=RND, overflow_mode=SAT)
