"""Optional W&B tracking; datasets and checkpoints are kept on the developer host."""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any


def default_monitor_config() -> str | None:
    explicit = os.environ.get("ACPV_WANDB_CONFIG")
    if explicit:
        return explicit
    candidate = Path(__file__).resolve().parents[4] / "configs/wandb_monitor.json"
    return str(candidate) if candidate.is_file() else None


def configure_paths(workspace: str | Path) -> dict[str, Path]:
    root = Path(workspace).resolve()
    paths = {
        "WANDB_DIR": root,
        "WANDB_CONFIG_DIR": root / "configs/wandb",
        "WANDB_CACHE_DIR": root / "cache/wandb",
        "WANDB_DATA_DIR": root / "cache/wandb/staging",
        "WANDB_CREDENTIALS_FILE": root / "configs/wandb/credentials.json",
        "NETRC": root / "configs/wandb/netrc",
        "TMPDIR": root / "tmp",
    }
    for key, path in paths.items():
        (path.parent if key in {"NETRC", "WANDB_CREDENTIALS_FILE"} else path).mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    paths["WANDB_CONFIG_DIR"].chmod(0o700)
    return paths


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class WandbMonitor:
    """Log real metrics supplied by training or validation; never invent validation scores."""

    def __init__(self, options: dict[str, Any] | None = None, *, config: dict[str, Any] | None = None,
                 output_dir: str | Path | None = None, resume: bool = False,
                 mode: str | None = None, run_name: str | None = None,
                 job_type: str = "training", collect_system_metrics: bool = True):
        self.options = dict(options or {})
        self.mode = mode or self.options.get("mode", "disabled")
        if not self.options.get("enabled", False):
            self.mode = "disabled"
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("W&B mode must be online, offline or disabled")
        self.config = dict(config or {})
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.resume = resume
        self.run_name = run_name
        self.job_type = job_type
        self.collect_system_metrics = collect_system_metrics
        self.run = None
        self.logger = logging.getLogger("training")
        self.last_step = -1
        self.nonfinite_values = 0
        self.record: dict[str, Any] = {}
        self.state_path = self.output_dir / "wandb_run.json" if self.output_dir else None

    @classmethod
    def from_file(cls, filename: str | None, **kwargs: Any) -> "WandbMonitor":
        options = json.loads(Path(filename).read_text()) if filename else {}
        return cls(options, **kwargs)

    @property
    def log_every(self) -> int:
        return max(1, int(self.options.get("log_every", 20)))

    def __enter__(self) -> "WandbMonitor":
        if self.mode == "disabled":
            return self
        paths = configure_paths(self.options["workspace"])
        import wandb

        project = self.options.get("project", "acpv-net-irsamap8")
        entity = self.options.get("entity") or None
        previous = None
        if self.resume and self.mode == "online" and self.state_path and self.state_path.is_file():
            previous = json.loads(self.state_path.read_text())
            if previous.get("project") != project or previous.get("entity") != entity:
                raise ValueError("W&B resume project/entity differs from the previous run")
            if previous.get("mode") != "online":
                previous = None  # An offline run is a separate local segment.
        settings = wandb.Settings(
            disable_git=True, disable_code=True, save_code=False, console=self.options.get("console", "off"),
            credentials_file=str(paths["WANDB_CREDENTIALS_FILE"]), init_timeout=30,
            x_stats_gpu_device_ids=self.options.get("gpu_device_ids", [0]),
            x_disable_stats=not self.collect_system_metrics,
        )
        self.run = wandb.init(
            project=project, entity=entity, name=self.run_name,
            mode=self.mode, dir=str(paths["WANDB_DIR"]), config=self.config,
            job_type=self.job_type, settings=settings,
            id=previous.get("run_id") if previous else None,
            resume="allow" if previous else None,
        )
        self.run.define_metric("step")
        for prefix in ["train/*", "val/*", "test/*", "runtime/*", "storage/*", "checkpoint/*", "health/*", "setup/*"]:
            self.run.define_metric(prefix, step_metric="step")
        self.run.define_metric("val/mIoU", step_metric="step", summary="max")
        url = self.run.url if self.mode == "online" else None
        self.record = {
            "run_id": self.run.id, "entity": entity, "project": project,
            "mode": self.mode, "url": url, "run_dir": self.run.dir,
            "status": "running", "job_type": self.job_type,
            "images_uploaded": False, "checkpoints_uploaded": False,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._save_state()
        self.logger.info("W&B mode=%s; project=%s; dashboard=%s", self.mode, project, url or "offline local recording")
        return self

    def _save_state(self) -> None:
        if self.state_path and self.record:
            atomic_json(self.state_path, self.record)

    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        if self.run is None:
            return
        if step < self.last_step:
            raise ValueError("The training step must not move backwards")
        clean: dict[str, Any] = {"step": int(step)}
        for key, value in metrics.items():
            if hasattr(value, "detach"):
                value = value.detach().item()
            elif hasattr(value, "item"):
                value = value.item()
            if not isinstance(value, (int, float, bool)):
                raise TypeError(f"Metric {key} must be a numeric scalar")
            if isinstance(value, float) and not math.isfinite(value):
                self.nonfinite_values += 1
                clean[key] = None
                self.logger.error("Nonfinite metric detected: %s", key)
            else:
                clean[key] = value
        clean["health/nonfinite_metric_count"] = self.nonfinite_values
        self.run.log(clean)
        self.last_step = step
        self.record.update(last_step=step, last_metrics=clean,
                           last_update_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self._save_state()

    def log_validation(self, metrics: dict[str, Any], *, step: int) -> None:
        self.log({f"val/{k}": v for k, v in metrics.items()}, step=step)

    def log_checkpoint(self, filename: str | Path, *, step: int) -> None:
        """Record file size and step, without uploading model weights."""
        p = Path(filename)
        self.log({"checkpoint/iteration": step, "checkpoint/file_bytes": p.stat().st_size}, step=step)
        if self.run is not None:
            self.run.summary["last_checkpoint_name"] = p.name

    def log_predictions(self, images: list[Any], truth: list[Any], predictions: list[Any],
                        class_names: list[str], *, step: int) -> None:
        if self.run is None or not self.options.get("log_prediction_images", False):
            return
        import wandb
        limit = min(8, max(0, int(self.options.get("max_prediction_images", 8))))
        samples = [wandb.Image(image, masks={
            "ground_truth": {"mask_data": target, "class_labels": dict(enumerate(class_names))},
            "prediction": {"mask_data": prediction, "class_labels": dict(enumerate(class_names))},
        }) for image, target, prediction in list(zip(images, truth, predictions))[:limit]]
        self.run.log({"step": step, "val/predictions": samples})
        self.record["images_uploaded"] = self.mode == "online" and bool(samples)
        self._save_state()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self.run is not None:
            status = "failed" if exc_type else "completed"
            try:
                self.run.summary["status"] = status
                if exc_type:
                    self.run.summary["exception_type"] = exc_type.__name__
                self.run.finish(exit_code=1 if exc_type else 0)
            except Exception as finish_error:
                self.record["monitoring_finalize_error_type"] = type(finish_error).__name__
                self.logger.warning("W&B finalization failed (%s); local training logs remain available.",
                                    type(finish_error).__name__)
            finally:
                self.record.update(status=status, finished_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                self._save_state()
        return False
