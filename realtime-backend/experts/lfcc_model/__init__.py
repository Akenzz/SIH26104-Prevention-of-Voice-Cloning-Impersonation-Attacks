"""Vendored LFCC-LCNN architecture for the Expert-3 adapter.

These modules are byte-for-byte copies of the trainer's architecture:
    lfcc-detector/models/lfcc_lcnn.py -> lfcc_lcnn.py
    lfcc-detector/models/features.py  -> features.py

They are vendored (not imported from the sibling package) so the realtime
backend runs standalone, even if ``lfcc-detector/`` is not checked out next to
it. The published checkpoint
``sarosh22/lfcc-lcnn-asvspoof19/best_lfcc_lcnn.pth`` loads into
``LFCCLCNNWithFeatureExtraction(sample_rate=16000, n_lfcc=20, with_deltas=True,
embedding_dim=128)`` with zero missing/unexpected keys.

If the trainer changes the architecture, re-vendor these two files and confirm
``load_state_dict`` still matches the new checkpoint.
"""

from .lfcc_lcnn import LFCCLCNN, LFCCLCNNWithFeatureExtraction

__all__ = ["LFCCLCNN", "LFCCLCNNWithFeatureExtraction"]
