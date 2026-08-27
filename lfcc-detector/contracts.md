# Model Interface Contract

**Version:** 1.0  
**Date:** 2026-08-26  
**Owner:** Task D (LFCC-LCNN Expert)

## Audio Contract

### Input Specification
- **Format:** Mono audio, 16 kHz sample rate
- **Data type:** `float32` in range `[-1.0, 1.0]`
- **Window size:** 4.0 seconds (64,000 samples at 16kHz)
- **Hop size:** 1.0 second (for streaming context)

### Input Handling
```python
# Expected input shape
audio_window: np.ndarray or torch.Tensor
    shape: (64000,) for single sample
    shape: (batch_size, 64000) for batch
    dtype: float32
    range: [-1.0, 1.0]
```

### Edge Cases
- **Too short:** Pad with zeros to 64,000 samples
- **Too long:** Truncate to first 64,000 samples
- **Silence/clipped audio:** Return `logit=0.0, confidence='uncertain'` (handled by backend policy)

---

## Model Interface

### Primary Method: `forward()`

```python
def forward(audio_window: Union[np.ndarray, torch.Tensor]) -> Tuple[float, np.ndarray]:
    """
    Score a single audio window for synthetic speech detection.
    
    Args:
        audio_window: Audio samples, shape (samples,) or (batch, samples)
                     Expected: 16kHz mono, ~4 seconds (64000 samples)
    
    Returns:
        logit (float): Spoof score. Higher = more evidence of synthetic speech.
                      Typical range: [-5.0, +5.0] but unbounded.
                      Direction: positive = spoof, negative = bonafide
        
        embedding (np.ndarray): 128-dimensional feature vector from penultimate layer.
                               Used by fusion gate to combine with other experts.
                               Shape: (128,)
    
    Raises:
        ValueError: If audio_window is wrong shape or dtype
    """
    pass
```

### Contract Guarantees
1. **Logit direction is fixed:** Higher values mean more spoof evidence
2. **Embedding dimension is fixed:** Always 128-dim, never changes
3. **Thread-safe:** Multiple calls can happen concurrently (model is in eval mode)
4. **Deterministic:** Same input → same output (no dropout at inference)
5. **Model version is tracked:** Checkpoint filename includes version/date

---

## Model Metadata

### Required Attributes
```python
class LFCCLCNNDetector:
    model_name: str = "LFCC-LCNN"
    model_version: str  # e.g. "v1.0_20260826"
    checkpoint_path: str
    device: torch.device
```

### Version String Format
`{architecture}_{date}_{dataset_hash}`

Example: `lfcc_lcnn_v1_20260826_asvspoof19`

---

## Integration with Backend

Backend will call your model like this:

```python
# One-time setup
from models.detector import LFCCLCNNDetector
detector = LFCCLCNNDetector(checkpoint_path='checkpoints/best_model.pth')

# Per-window inference (called every 1 second in streaming mode)
logit, embedding = detector.forward(audio_window)

# Backend then:
# 1. Combines logit with other experts via fusion gate
# 2. Applies calibration (logit -> probability)
# 3. Temporal smoothing (EMA over windows)
# 4. Thresholding (low/uncertain/high risk bands)
```

---

## Calibration (Handled by Backend, Not Your Model)

Your model outputs **uncalibrated logits**. The backend is responsible for:
- Temperature scaling or isotonic regression (fitted on dev set)
- Converting logit → probability
- Setting thresholds for risk bands

**You do not need to output probabilities.** Just logits.

---

## Performance Requirements

### Latency Targets
- **Inference time:** < 500ms per 4-second window on GPU
- **Inference time:** < 2000ms per 4-second window on CPU
- **Memory:** < 2GB VRAM for model + single batch

### Measured On
- GPU: NVIDIA RTX 3060 or equivalent
- CPU: Intel i5 10th gen or equivalent

---

## Testing Contract Compliance

```python
# Test script to verify your model follows the contract
def test_model_contract(detector):
    # Test 1: Correct output types
    audio = np.random.randn(64000).astype(np.float32) * 0.1
    logit, embedding = detector.forward(audio)
    
    assert isinstance(logit, float), "logit must be float"
    assert isinstance(embedding, np.ndarray), "embedding must be numpy array"
    assert embedding.shape == (128,), f"embedding must be 128-dim, got {embedding.shape}"
    
    # Test 2: Logit direction (bonafide should be negative on average)
    # Load known bonafide sample, check logit < 0
    
    # Test 3: Deterministic
    logit2, _ = detector.forward(audio)
    assert abs(logit - logit2) < 1e-6, "Model must be deterministic"
    
    # Test 4: Batch handling
    audio_batch = np.random.randn(8, 64000).astype(np.float32) * 0.1
    # Should handle or raise clear error
    
    print("✓ All contract tests passed")
```

Run this before integration: `python evaluation/test_contract.py`

---

## Change Log

| Date | Version | Change |
|---|---|---|
| 2026-08-26 | 1.0 | Initial contract definition |
