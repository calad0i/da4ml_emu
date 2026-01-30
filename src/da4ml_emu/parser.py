import hls4ml.model.layers
import numpy as np
from da4ml.trace import FixedVariableArray
from hls4ml.model.graph import ModelGraph
from hls4ml.model.layers import Layer
from hls4ml.model.types import FixedPrecisionType

from .layers import dispatch_layer


def trace_hls4ml_model(model: ModelGraph):
    graph: dict[str, Layer] = model.graph
    io_map: dict[str, tuple[Layer, list[str], list[str]]] = {}
    'name -> (layer, inputs, outputs)'
    for layer_name, layer in graph.items():
        io_map[layer_name] = (layer, layer.inputs, layer.outputs)

    tensor_map: dict[str, FixedVariableArray] = {}
    inp_tensor_outer: dict[str, FixedVariableArray] = {}

    for inp_var in model.get_input_variables():
        shape = inp_var.shape
        precision: FixedPrecisionType = inp_var.type.precision
        k, i, f = precision.signed, precision.integer - precision.signed, precision.fractional
        SAT, RND = str(precision.saturation_mode), str(precision.rounding_mode)
        k, i, f = (np.full(shape, x, dtype=np.int16) for x in (k, i, f))
        if RND == 'RND':
            _f = f + 1
        elif RND == 'TRN':
            _f = f
        else:
            raise ValueError(f'Unsupported rounding mode {RND} for input variable {inp_var.name}')
        if SAT == 'WRAP':
            _i = i
        elif SAT == 'SAT':
            _i = np.full_like(i, 32, dtype=np.int16)
            print(f'WRAN: input saturation mode {SAT} cannot be perfectly bit-exact, using 32 integer bits inputs')
        else:
            raise ValueError(f'Unsupported saturation mode {SAT} for input variable {inp_var.name}')

        t = FixedVariableArray.from_kif(k, _i, _f)
        inp_tensor_outer[inp_var.name] = t
        tensor_map[inp_var.name] = t.quantize(k, i, f, round_mode=RND, overflow_mode=SAT)

    inp_tensors = list(tensor_map.values())

    for layer_name, (layer, inputs, outputs) in io_map.items():
        if isinstance(layer, hls4ml.model.layers.Input):
            continue
        in_tensors = [tensor_map[inp_name] for inp_name in inputs]
        out_tensors = dispatch_layer(layer, in_tensors)
        assert len(outputs) == 1, 'Only single output layers are supported'
        tensor_map[outputs[0]] = out_tensors

    assert model.outputs is not None
    assert model.inputs is not None

    inp_tensors = [inp_tensor_outer[name] for name in model.inputs]
    out_tensors = [tensor_map[name] for name in model.outputs]

    return inp_tensors, out_tensors
