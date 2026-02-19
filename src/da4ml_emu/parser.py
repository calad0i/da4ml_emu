from collections.abc import Sequence

import hls4ml.model.layers
from da4ml.converter.plugin import DAISTracerPluginBase
from da4ml.trace import FixedVariableArray
from hls4ml.model.graph import ModelGraph
from hls4ml.model.layers import Layer
from hls4ml.model.types import FixedPrecisionType

from .layers import dispatch_layer


def _prec_to_kif(prec: FixedPrecisionType):
    return int(prec.signed), int(prec.integer - prec.signed), int(prec.fractional)


class hls4mlTracer(DAISTracerPluginBase):
    model: ModelGraph

    def __init__(
        self,
        *args,
        optimize: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.optimize = optimize

    def get_input_shapes(self):
        return [tuple(v.shape) for v in self.model.get_input_variables()]

    def _get_inputs(
        self,
        inputs: tuple[FixedVariableArray, ...] | FixedVariableArray | None,
        inputs_kif: Sequence[tuple[int, int, int]] | tuple[int, int, int] | None,
    ) -> tuple[FixedVariableArray, ...]:
        if inputs is None and inputs_kif is None:
            # inputs_kif = inputs_kif or [_prec_to_kif(v.type.precision) for v in self.model.get_input_variables()]
            inputs_kif = []
            for v in self.model.get_input_variables():
                prec: FixedPrecisionType = v.type.precision
                k, i, f = _prec_to_kif(prec)
                if prec.saturation_mode == 'SAT':
                    i = 24
                    print(f'WRAN: input saturation mode {prec.saturation_mode} cannot be perfectly bit-exact')
                if str(prec.rounding_mode) == 'RND':
                    f += 1
                elif str(prec.rounding_mode) != 'TRN':
                    f = 24
                    print(
                        f'WRAN: unsupported rounding mode {prec.rounding_mode} for input variable {v.name}, using 32 fractional bits'
                    )
                inputs_kif.append((k, i, f))

        return super()._get_inputs(inputs, inputs_kif)

    def apply_model(
        self,
        verbose: bool,
        inputs: tuple[FixedVariableArray, ...],
    ) -> tuple[dict[str, FixedVariableArray], list[str]]:
        graph: dict[str, Layer] = self.model.graph
        io_map: dict[str, tuple[Layer, list[str], list[str]]] = {}
        'name -> (layer, inputs, outputs)'
        for layer_name, layer in graph.items():
            io_map[layer_name] = (layer, layer.inputs, layer.outputs)

        tensor_map: dict[str, FixedVariableArray] = {}
        inp_tensor_outer: dict[str, FixedVariableArray] = {}

        for tensor, inp_var in zip(inputs, self.model.get_input_variables()):
            precision: FixedPrecisionType = inp_var.type.precision
            k, i, f = precision.signed, precision.integer - precision.signed, precision.fractional
            SAT, RND = str(precision.saturation_mode), str(precision.rounding_mode)
            inp_tensor_outer[inp_var.name] = tensor
            tensor_map[inp_var.name] = tensor.quantize(k, i, f, round_mode=RND, overflow_mode=SAT)

        for layer_name, (layer, _inp_names, _out_names) in io_map.items():
            if isinstance(layer, hls4ml.model.layers.Input):
                continue
            in_tensors = [tensor_map[inp_name] for inp_name in _inp_names]
            if verbose:
                print(f'Dispatching layer {layer_name}...', end='')
            out_tensors = dispatch_layer(layer, in_tensors, self.optimize)
            if verbose:
                print(' done.')
            assert len(_out_names) == 1, 'Only single output layers are supported'
            tensor_map[_out_names[0]] = out_tensors

        assert self.model.outputs is not None
        assert self.model.inputs is not None

        return tensor_map, self.model.outputs
