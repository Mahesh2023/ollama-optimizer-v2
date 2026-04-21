"""MLflow integration for model registry and experiment tracking."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    MLFLOW_AVAILABLE = False


class ModelRegistry:
    """Thin wrapper around MLflow for Ollama model configs."""

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "ollama-optimizer",
    ) -> None:
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("Install mlflow: pip install 'ollama-optimizer[llmops]'")

        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()

    def log_benchmark_run(
        self,
        model_name: str,
        tuning_config: dict[str, Any],
        metrics: dict[str, float],
        modelfile_content: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Log a benchmark + tuning run. Returns run_id."""
        with mlflow.start_run(run_name=f"tune-{model_name}") as run:
            mlflow.log_params(tuning_config)
            mlflow.log_metrics(metrics)
            if modelfile_content:
                mlflow.log_text(modelfile_content, "Modelfile")
            if tags:
                mlflow.set_tags(tags)
            return run.info.run_id

    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "Modelfile",
    ) -> str:
        """Register model version from run. Returns version."""
        registered_name = f"ollama-{model_name}-optimized"
        try:
            result = mlflow.register_model(
                model_uri=f"runs:/{run_id}/{artifact_path}",
                name=registered_name,
            )
            return result.version
        except Exception as e:  # noqa: BLE001
            logger.warning("Model registration failed: %s", e)
            return ""

    def promote_to_production(self, model_name: str, version: str) -> None:
        """Transition model version to Production stage."""
        self.client.transition_model_version_stage(
            name=f"ollama-{model_name}-optimized",
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )

    def get_production_config(self, model_name: str) -> dict[str, Any] | None:
        """Fetch production-promoted config params."""
        registered_name = f"ollama-{model_name}-optimized"
        try:
            versions = self.client.get_latest_versions(registered_name, stages=["Production"])
            if not versions:
                return None
            run = self.client.get_run(versions[0].run_id)
            return dict(run.data.params)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch production config: %s", e)
            return None

    def compare_runs(self, run_ids: list[str]) -> list[dict[str, Any]]:
        """Compare metrics across runs."""
        return [
            {
                "run_id": rid,
                "params": dict(self.client.get_run(rid).data.params),
                "metrics": dict(self.client.get_run(rid).data.metrics),
            }
            for rid in run_ids
        ]
