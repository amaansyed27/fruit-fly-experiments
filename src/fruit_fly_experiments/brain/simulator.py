from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse

from .loader import ConnectomeGraph


@dataclass(frozen=True)
class StepResult:
    spikes: np.ndarray
    active_count: int
    step_index: int
    simulated_time: float
    latency_ms: float


def resolve_device(requested: str = "auto") -> str:
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    if requested == "cuda" and not cuda:
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return "cuda" if cuda else "cpu"


class BrainSimulator:
    """Sparse point-neuron simulation constrained by the MaleCNS wiring graph.

    The connectome is measured biology. Dynamics below are an engineering model,
    not a claim of physiological fidelity.
    """

    def __init__(
        self,
        graph: ConnectomeGraph,
        *,
        device: str = "auto",
        seed: int = 1,
        dt: float = 0.020,
        tau: float = 0.100,
        threshold: float = 1.0,
        gain: float = 3.0,
        tonic: float = 0.14,
        noise_std: float = 0.0,
    ) -> None:
        self.graph = graph
        self.device = resolve_device(device)
        self.seed = int(seed)
        self.dt = float(dt)
        self.decay = float(np.exp(-dt / tau))
        self.threshold = float(threshold)
        self.gain = float(gain)
        self.tonic = float(tonic)
        self.noise_std = float(noise_std)
        self.step_index = 0
        self.rng = np.random.default_rng(seed)
        self.n = graph.neurons.size
        self._last_spikes = np.zeros(self.n, dtype=np.float32)
        self._voltage_cpu = np.zeros(self.n, dtype=np.float32)
        self._matrix_cpu = graph.weights.astype(np.float32).tocsr()
        self._torch = None
        self._matrix_gpu = None
        self._voltage_gpu = None
        self._spikes_gpu = None
        if self.device == "cuda":
            self._init_cuda()

    def _init_cuda(self) -> None:
        import torch
        csr = self._matrix_cpu
        self._torch = torch
        crow = torch.from_numpy(csr.indptr.astype(np.int64, copy=False)).to("cuda")
        col = torch.from_numpy(csr.indices.astype(np.int64, copy=False)).to("cuda")
        val = torch.from_numpy(csr.data.astype(np.float32, copy=False)).to("cuda")
        self._matrix_gpu = torch.sparse_csr_tensor(crow, col, val, size=csr.shape, device="cuda")
        self._voltage_gpu = torch.zeros(self.n, dtype=torch.float32, device="cuda")
        self._spikes_gpu = torch.zeros(self.n, dtype=torch.float32, device="cuda")
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    @property
    def voltage(self) -> np.ndarray:
        if self.device == "cpu":
            return self._voltage_cpu.copy()
        return self._voltage_gpu.detach().cpu().numpy().copy()

    @property
    def last_spikes(self) -> np.ndarray:
        return self._last_spikes.copy()

    def reset(self) -> None:
        self.step_index = 0
        self.rng = np.random.default_rng(self.seed)
        self._last_spikes.fill(0)
        self._voltage_cpu.fill(0)
        if self.device == "cuda":
            self._voltage_gpu.zero_()
            self._spikes_gpu.zero_()

    def step(self, inject_indices: np.ndarray | None = None, inject_values: np.ndarray | float | None = None) -> StepResult:
        start = perf_counter()
        if self.device == "cpu":
            spikes = self._step_cpu(inject_indices, inject_values)
        else:
            spikes = self._step_cuda(inject_indices, inject_values)
        self.step_index += 1
        self._last_spikes = spikes.astype(np.float32, copy=False)
        return StepResult(
            spikes=np.flatnonzero(spikes > 0),
            active_count=int(np.count_nonzero(spikes)),
            step_index=self.step_index,
            simulated_time=self.step_index * self.dt,
            latency_ms=(perf_counter() - start) * 1000.0,
        )

    def _injection_vector(self, indices: np.ndarray | None, values: np.ndarray | float | None) -> np.ndarray:
        result = np.zeros(self.n, dtype=np.float32)
        if indices is None or len(indices) == 0:
            return result
        idx = np.asarray(indices, dtype=np.int64)
        if np.any((idx < 0) | (idx >= self.n)):
            raise IndexError("injection contains neuron index outside graph")
        if values is None:
            val = np.ones(len(idx), dtype=np.float32)
        elif np.isscalar(values):
            val = np.full(len(idx), float(values), dtype=np.float32)
        else:
            val = np.asarray(values, dtype=np.float32)
            if len(val) != len(idx):
                raise ValueError("inject_values must match inject_indices")
        np.add.at(result, idx, val)
        return result

    def _step_cpu(self, indices, values) -> np.ndarray:
        synaptic = self._matrix_cpu @ self._last_spikes
        injection = self._injection_vector(indices, values)
        noise = self.rng.normal(0.0, self.noise_std, self.n).astype(np.float32) if self.noise_std else 0.0
        self._voltage_cpu = self.decay * self._voltage_cpu + self.gain * synaptic + self.tonic + injection + noise
        spikes = self._voltage_cpu >= self.threshold
        self._voltage_cpu[spikes] = 0.0
        if not np.all(np.isfinite(self._voltage_cpu)):
            raise FloatingPointError("non-finite membrane voltage")
        return spikes.astype(np.float32)

    def _step_cuda(self, indices, values) -> np.ndarray:
        torch = self._torch
        synaptic = torch.mv(self._matrix_gpu, self._spikes_gpu)
        self._voltage_gpu.mul_(self.decay).add_(synaptic, alpha=self.gain).add_(self.tonic)
        if indices is not None and len(indices):
            idx = torch.as_tensor(np.asarray(indices, dtype=np.int64), device="cuda")
            if values is None:
                val = torch.ones(len(idx), device="cuda")
            elif np.isscalar(values):
                val = torch.full((len(idx),), float(values), device="cuda")
            else:
                val = torch.as_tensor(np.asarray(values, dtype=np.float32), device="cuda")
            self._voltage_gpu.index_add_(0, idx, val)
        if self.noise_std:
            self._voltage_gpu.add_(torch.randn_like(self._voltage_gpu), alpha=self.noise_std)
        spikes = self._voltage_gpu >= self.threshold
        self._spikes_gpu = spikes.to(torch.float32)
        self._voltage_gpu.masked_fill_(spikes, 0.0)
        return self._spikes_gpu.detach().cpu().numpy()
