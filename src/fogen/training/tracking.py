"""Experiment tracking — keeps logging logic out of the training loop.

Usage in train.py:
    tracker = create_tracker(cfg, out, args.mlflow, not args.no_wandb)
    # then just: tracker.log_step(), tracker.log_probes(), etc.
"""

import json
from pathlib import Path


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = str(v)
        else:
            out[key] = v
    return out


class MLflowTracker:

    def __init__(self, cfg, out_dir, run_name):
        self._active = False
        self._out = Path(out_dir)
        try:
            import mlflow
            self._mlflow = mlflow

            experiment = cfg.get("experiment", cfg.get("wandb_project", "fogen-phase"))
            mlflow.set_experiment(experiment)
            self._run = mlflow.start_run(
                run_name=run_name, log_system_metrics=True)
            mlflow.log_params(_flatten(cfg))
            self._log_dataset(cfg)
            self._active = True
        except Exception as e:
            print(f"mlflow disabled: {e}")

    def _log_dataset(self, cfg):
        shard_dir = Path(cfg["data"]["shard_dir"])
        manifest_path = shard_dir / "manifest.json"
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text())
        import numpy as np
        dataset = self._mlflow.data.from_numpy(
            features=np.empty(0),
            source=str(shard_dir),
            name=shard_dir.parent.name,
        )
        self._mlflow.log_input(dataset, context="training")
        self._mlflow.set_tags({
            "dataset.total_tokens": manifest.get("total_tokens"),
            "dataset.n_shards": len(manifest.get("shards", [])),
            "dataset.max_tokens": cfg["data"].get("max_tokens", "all"),
            "dataset.tokenizer": cfg["data"]["tokenizer_dir"],
            "dataset.shard_dir": str(shard_dir),
        })

    def log_step(self, step, rec, execution_metrics=None, guard_record=None,
                 muon_lr=None, adamw_lr=None):
        if not self._active:
            return
        metrics = {
            "train/loss": rec["loss"],
            "train/tok_s": rec["tok_s"],
            "train/step": step,
        }
        if muon_lr is not None:
            metrics["lr/muon"] = muon_lr
        if adamw_lr is not None:
            metrics["lr/adamw"] = adamw_lr
        if execution_metrics is not None:
            metrics.update({f"execution/{k}": v.item()
                            for k, v in execution_metrics.items()})
        if guard_record is not None:
            for k, v in guard_record.items():
                metrics[f"guard/{k}"] = int(v) if isinstance(v, bool) else float(v)
        self._mlflow.log_metrics(metrics, step=step)

    def log_probes(self, step, aggs):
        if not self._active:
            return
        probe_metrics = {}
        for a in aggs:
            prefix = f"probe/{a['probe']}/{a['split']}"
            probe_metrics[f"{prefix}/acc"] = a["argmax_acc"]
            probe_metrics[f"{prefix}/logprob_diff"] = a["logprob_diff"]
        self._mlflow.log_metrics(probe_metrics, step=step)

    def log_checkpoint(self, step):
        if not self._active:
            return
        self._mlflow.log_metrics({"train/step": step}, step=step)
        self._mlflow.log_artifact(str(self._out / "config_used.yaml"))
        self._mlflow.log_artifact(str(self._out / "train_log.jsonl"))
        self._mlflow.log_artifact(str(self._out / "probe_log.jsonl"))
        ckpt_path = self._out / "ckpts" / f"step{step:06d}.safetensors"
        if ckpt_path.exists():
            self._mlflow.log_artifact(str(ckpt_path), artifact_path="ckpts")

    def finish(self):
        if not self._active:
            return
        self._mlflow.log_artifact(str(self._out / "train_log.jsonl"))
        self._mlflow.log_artifact(str(self._out / "probe_log.jsonl"))
        self._mlflow.log_artifact(str(self._out / "config_used.yaml"))
        self._mlflow.end_run()


class WandbTracker:

    def __init__(self, cfg, run_name):
        self._run = None
        try:
            import wandb
            project = cfg.get("experiment", cfg.get("wandb_project", "fogen-phase"))
            self._run = wandb.init(
                project=project, name=run_name, config=cfg)
        except Exception as e:
            print(f"wandb disabled: {e}")

    def log_step(self, step, rec, execution_metrics=None, guard_record=None,
                 muon_lr=None, adamw_lr=None):
        if not self._run:
            return
        metrics = {"train/loss": rec["loss"], "train/tok_s": rec["tok_s"]}
        if muon_lr is not None:
            metrics["lr/muon"] = muon_lr
        if adamw_lr is not None:
            metrics["lr/adamw"] = adamw_lr
        if execution_metrics is not None:
            metrics.update({f"execution/{k}": v.item()
                            for k, v in execution_metrics.items()})
        if guard_record is not None:
            for k, v in guard_record.items():
                metrics[f"guard/{k}"] = int(v) if isinstance(v, bool) else float(v)
        self._run.log(metrics, step=step)

    def log_probes(self, step, aggs):
        if not self._run:
            return
        probe_metrics = {}
        for a in aggs:
            prefix = f"probe/{a['probe']}/{a['split']}"
            probe_metrics[f"{prefix}/acc"] = a["argmax_acc"]
            probe_metrics[f"{prefix}/logprob_diff"] = a["logprob_diff"]
        self._run.log(probe_metrics | {"step": step}, step=step)

    def log_checkpoint(self, step):
        pass

    def finish(self):
        if self._run:
            self._run.finish()


class CompositeTracker:
    """Fans out calls to multiple trackers."""

    def __init__(self, trackers):
        self._trackers = trackers

    def log_step(self, *a, **kw):
        for t in self._trackers:
            t.log_step(*a, **kw)

    def log_probes(self, *a, **kw):
        for t in self._trackers:
            t.log_probes(*a, **kw)

    def log_checkpoint(self, *a, **kw):
        for t in self._trackers:
            t.log_checkpoint(*a, **kw)

    def finish(self):
        for t in self._trackers:
            t.finish()


class NoopTracker:

    def log_step(self, *a, **kw): pass
    def log_probes(self, *a, **kw): pass
    def log_checkpoint(self, *a, **kw): pass
    def finish(self): pass


def create_tracker(cfg, out_dir, use_mlflow=False, use_wandb=False):
    trackers = []
    if use_mlflow:
        trackers.append(MLflowTracker(cfg, out_dir, Path(out_dir).name))
    if use_wandb:
        trackers.append(WandbTracker(cfg, Path(out_dir).name))
    if not trackers:
        return NoopTracker()
    if len(trackers) == 1:
        return trackers[0]
    return CompositeTracker(trackers)
