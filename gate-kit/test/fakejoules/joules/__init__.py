"""A stand-in for joules in the kit's test only (test/test_kit.sh): a "GPU" that draws a fixed 300 W, so the energy
arithmetic runs end to end on a machine with no meter. Never on PYTHONPATH in a real run."""
import contextlib, time

__version__ = "fake"
W = 300.0


class _Phase:
    def __init__(self):
        self.t0 = time.perf_counter(); self.duration_s = 0.0; self.energy_j = 0.0; self.devices = {}


class Session:
    def __init__(self, gpus=(0,), cpu=False, interval=0.1):
        self.t0 = time.perf_counter()
        self.devices = [{"name": "fake-gpu0", "kind": "gpu", "method": "fake 300 W"}]
        self.notes = ["test stand-in: fixed 300 W, not a measurement"]

    def snapshot(self):
        return {"devices_j": {"fake-gpu0": W * (time.perf_counter() - self.t0)}}

    @contextlib.contextmanager
    def phase(self, name):
        ph = _Phase()
        try:
            yield ph
        finally:
            ph.duration_s = time.perf_counter() - ph.t0
            ph.energy_j = W * ph.duration_s
            ph.devices = {"fake-gpu0": ph.energy_j}

    def close(self):
        pass
