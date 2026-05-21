from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from colosseum_sdk.inference.base import InferenceEngine


@dataclass
class _StateBuffer:
    input_name: str
    output_name: str
    shape: list[int]
    data: np.ndarray  # zeroed, updated each infer()


class OnnxInferenceEngine(InferenceEngine):
    """ONNX Runtime inference engine. Port of arena/include/engines/OnnxInferenceEngine.h.

    Supports both stateless models (obs → actions) and stateful RMA models
    (obs + state_in... → actions + state_out...).

    State buffers are allocated at construction (zeroed) and automatically
    updated each infer() call. Call reset_state() at episode boundaries.

    I/O layout convention (must match the Python export):
      inputs:  obs, [state_in_0, state_in_1, ...]
      outputs: actions, [state_out_0, state_out_1, ...]
    """

    def __init__(self, model_path: str | Path) -> None:
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()

        # Primary obs input / action output (always first).
        self._input_name: str = inputs[0].name
        self._output_name: str = outputs[0].name
        self._input_dim: int = int(np.prod([d for d in inputs[0].shape if isinstance(d, int) and d > 0]))
        self._output_dim: int = int(np.prod([d for d in outputs[0].shape if isinstance(d, int) and d > 0]))

        # Recurrent state buffers for stateful (RMA) models.
        self._state_bufs: list[_StateBuffer] = []
        for inp, out in zip(inputs[1:], outputs[1:]):
            shape = [d for d in inp.shape if isinstance(d, int) and d > 0]
            self._state_bufs.append(_StateBuffer(
                input_name=inp.name,
                output_name=out.name,
                shape=shape,
                data=np.zeros(shape, dtype=np.float32),
            ))

    # ------------------------------------------------------------------

    def infer(self, obs: np.ndarray) -> np.ndarray:
        feed = {self._input_name: obs.reshape(1, -1).astype(np.float32)}
        for buf in self._state_bufs:
            feed[buf.input_name] = buf.data.reshape([1] + buf.shape)

        all_output_names = [self._output_name] + [b.output_name for b in self._state_bufs]
        results = self._session.run(all_output_names, feed)

        # Update recurrent state in-place.
        for buf, new_state in zip(self._state_bufs, results[1:]):
            buf.data[:] = new_state.reshape(buf.shape)

        return results[0].flatten().astype(np.float32)

    def reset_state(self) -> None:
        for buf in self._state_bufs:
            buf.data[:] = 0.0

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim
