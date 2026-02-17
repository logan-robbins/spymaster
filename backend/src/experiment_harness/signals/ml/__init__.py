"""ML walk-forward signal implementations.

Each module defines a single MLSignal subclass and registers it
with the signal registry at import time.
"""
from __future__ import annotations

from src.experiment_harness.signals import register_signal
from src.experiment_harness.signals.ml.gbm_mf import GBMMFSignal
from src.experiment_harness.signals.ml.knn_cl import KNNCLSignal
from src.experiment_harness.signals.ml.lsvm_der import LSVMDERSignal
from src.experiment_harness.signals.ml.pca_ad import PCAADSignal
from src.experiment_harness.signals.ml.svm_sp import SVMSPSignal
from src.experiment_harness.signals.ml.xgb_snap import XGBSnapSignal

register_signal("svm_sp", SVMSPSignal)
register_signal("gbm_mf", GBMMFSignal)
register_signal("knn_cl", KNNCLSignal)
register_signal("lsvm_der", LSVMDERSignal)
register_signal("xgb_snap", XGBSnapSignal)
register_signal("pca_ad", PCAADSignal)

__all__: list[str] = [
    "SVMSPSignal",
    "GBMMFSignal",
    "KNNCLSignal",
    "LSVMDERSignal",
    "XGBSnapSignal",
    "PCAADSignal",
]
