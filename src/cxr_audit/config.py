"""Configuration models and loaders with dual Pydantic v1/v2 support."""

from pathlib import Path
from typing import Any, Literal
import json, yaml, pydantic
from pydantic import BaseModel

LABEL_COLS: list[str] = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
]

MODEL_NAMES: list[str] = ["densenet121", "convnextv2_tiny", "swinb_lora"]
IS_PYDANTIC_V2 = pydantic.__version__.startswith("2")

class ThresholdRule(BaseModel):
    fail_below: float | None = None
    warn_below: float | None = None
    fail_above: float | None = None
    warn_above: float | None = None

class GateCriteria(BaseModel):
    min_macro_auroc: ThresholdRule
    min_critical_recall: ThresholdRule
    max_adebayo_spearman_rho: ThresholdRule
    max_type_b_fp_delta: ThresholdRule
    max_critical_blind_fn_rate: ThresholdRule

def _validate_severity_tiers_data(tiers: dict[str, list[str]]) -> None:
    all_tiered = []
    for tier in ["Critical", "Moderate", "Mild"]:
        if tier not in tiers:
            raise ValueError(f"Missing required severity tier: {tier}")
        all_tiered.extend(tiers[tier])
    if len(all_tiered) != len(set(all_tiered)):
        raise ValueError("Duplicate classes across severity tiers.")
    if set(all_tiered) != set(LABEL_COLS):
        raise ValueError("Severity tiers must partition all 14 canonical classes.")

def _validate_thresholds_data(data: dict[str, dict[str, float]]) -> None:
    for model in MODEL_NAMES:
        if model not in data:
            raise ValueError(f"Missing model: {model}")
        if set(LABEL_COLS) - set(data[model].keys()):
            raise ValueError(f"Model {model} missing class thresholds.")

if IS_PYDANTIC_V2:
    from pydantic import RootModel, model_validator
    class GatePolicyConfig(BaseModel):
        severity_tiers: dict[Literal["Critical", "Moderate", "Mild"], list[str]]
        criteria: GateCriteria
        @model_validator(mode="after")
        def check_tiers(self) -> "GatePolicyConfig":
            _validate_severity_tiers_data(self.severity_tiers)
            return self

    class ThresholdsConfig(RootModel[dict[str, dict[str, float]]]):
        @model_validator(mode="after")
        def check_thresholds(self) -> "ThresholdsConfig":
            _validate_thresholds_data(self.root)
            return self
else:
    from pydantic import root_validator
    class GatePolicyConfig(BaseModel):
        severity_tiers: dict[Literal["Critical", "Moderate", "Mild"], list[str]]
        criteria: GateCriteria
        @root_validator
        def check_tiers(cls, values: dict[str, Any]) -> dict[str, Any]:
            if "severity_tiers" in values:
                _validate_severity_tiers_data(values["severity_tiers"])
            return values
        @classmethod
        def model_validate(cls, obj: Any) -> "GatePolicyConfig":
            return cls.parse_obj(obj)

    class ThresholdsConfig(BaseModel):
        __root__: dict[str, dict[str, float]]
        @root_validator
        def check_thresholds(cls, values: dict[str, Any]) -> dict[str, Any]:
            if "__root__" in values:
                _validate_thresholds_data(values["__root__"])
            return values
        @property
        def root(self) -> dict[str, dict[str, float]]:
            return self.__root__
        @classmethod
        def model_validate(cls, obj: Any) -> "ThresholdsConfig":
            return cls.parse_obj(obj)

def load_gate_policy(path: Path | str = Path("configs/gate_policy.yaml")) -> GatePolicyConfig:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return GatePolicyConfig.model_validate(yaml.safe_load(f))

def load_thresholds(path: Path | str = Path("configs/thresholds.json")) -> dict[str, dict[str, float]]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return ThresholdsConfig.model_validate(json.load(f)).root
