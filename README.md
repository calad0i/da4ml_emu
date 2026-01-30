# bit-exact DA4ML emulator for HLS4ML

Vivado/Vitis models only with only a subset of layers (dense/conv+some activations and softmax), but shall be pretty extendable. Bit-exact emulation performed in da4ml by DAIS interperter running with int64 arithmetic.

```python
from da4ml_emu import trace_hls4ml_model

inp, out = trace_hls4ml_model(model_hls)
comb = comb_trace(inp, out)
comb.predict(...) # DAIS interpreter called here
comb.save_binary('/tmp/1.bin') # save binary DAIS for use in cpp directly
```
