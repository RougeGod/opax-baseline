import csv
import os
from typing import Any, Dict


class CSVLogger:
    """Minimal wandb.log replacement: one CSV file per metric, with columns (x, y).

    x is taken from a step-like key in the logged dict (env_steps, learning_step,
    train_steps, step, in that order of preference) or, if none is present,
    from a per-metric row counter.
    """

    STEP_KEYS = ("env_steps", "learning_step", "train_steps", "step")

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._files = {}
        self._writers = {}
        self._counters = {}

    def _writer_for(self, metric: str):
        if metric not in self._writers:
            path = os.path.join(self.log_dir, f"{metric}.csv")
            is_new = not os.path.exists(path)
            f = open(path, "a", newline="")
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["x", "y"])
            self._files[metric] = f
            self._writers[metric] = writer
            self._counters[metric] = 0
        return self._writers[metric]

    def log(self, data: Dict[str, Any]):
        x = next((data[key] for key in self.STEP_KEYS if key in data), None)
        for key, value in data.items():
            if key in self.STEP_KEYS:
                continue
            writer = self._writer_for(key)
            writer.writerow([x if x is not None else self._counters[key], value])
            self._files[key].flush()
            self._counters[key] += 1

    def close(self):
        for f in self._files.values():
            f.close()
