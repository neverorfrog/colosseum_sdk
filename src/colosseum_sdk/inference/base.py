from abc import ABC, abstractmethod

import numpy as np


class InferenceEngine(ABC):
    """Abstract inference engine. Port of arena/include/engines/IInferenceEngine.h."""

    @abstractmethod
    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Run one forward pass. obs shape: (input_dim,). Returns (output_dim,)."""

    @abstractmethod
    def reset_state(self) -> None:
        """Zero recurrent state buffers. No-op for stateless models."""

    @property
    @abstractmethod
    def input_dim(self) -> int: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...
