from functools import singledispatch

import numpy as np
from da4ml.trace import FixedVariableArray
from da4ml.trace.ops import quantize
from hls4ml.backends.fpga.fpga_types import FixedPrecisionType
from hls4ml.model.layers import Layer


def to_quantizer(precision: FixedPrecisionType|None):
    if precision is None:
        return lambda x: x
    width = precision.width
    integer = precision.integer
    signed = precision.signed
    rounding = str(precision.rounding_mode)
    overflow = str(precision.saturation_mode)
    k, i, f = signed, integer - signed, width - integer

    return lambda x: quantize(x, k, i, f, overflow, rounding)


def apply_precision(precision: FixedPrecisionType, data: np.ndarray):
    return to_quantizer(precision)(data)


# from hls4ml.model.layers import Input, Dense


@singledispatch
def dispatch_layer(layer: Layer, inputs: list[FixedVariableArray], optimize: bool) -> FixedVariableArray:
    raise NotImplementedError(f'Layer type {type(layer)} not implemented in dispatcher')


class BasicDispatcherMeta(type):
    def __new__(cls, name, bases, attrs):
        r = super().__new__(cls, name, bases, attrs)
        if r.layer_type is not None:  # type: ignore
            dispatch_layer.register(r.layer_type, r().__call__)  # type: ignore
        return r


class BasicDispatcher(metaclass=BasicDispatcherMeta):
    layer_type = None

    def __call__(self, layer: Layer, inputs: list[FixedVariableArray], optimize: bool) -> FixedVariableArray:
        r = self.call(layer, inputs, optimize)
        output_precision: FixedPrecisionType = layer.get_output_variable().type.precision
        assert isinstance(output_precision, FixedPrecisionType), (
            f'Output precision of layer {layer.name} is not FixedPrecisionType'
        )
        assert not output_precision.saturation_bits, f'Saturation bits not supported in layer {layer.name}'
        k, B, I = output_precision.signed, output_precision.width, output_precision.integer
        i, f = I - k, B - I
        overflow_mode = str(output_precision.saturation_mode)
        round_mode = str(output_precision.rounding_mode)
        return r.quantize(k, i, f, round_mode=round_mode, overflow_mode=overflow_mode)

    def get_quantized(self, layer: Layer, key: str) -> np.ndarray:
        v = layer.attributes.attributes[key]
        precision: FixedPrecisionType = v.type.precision
        value = v.data
        q = to_quantizer(precision)
        return q(np.round(value, precision.fractional))  # first round is for hls4ml print artifact...

    def call(self, layer, inputs: list[FixedVariableArray], optimize: bool) -> FixedVariableArray:
        raise NotImplementedError(f'Dispatcher for layer {layer.name} ({layer.__class__}) not implemented')
