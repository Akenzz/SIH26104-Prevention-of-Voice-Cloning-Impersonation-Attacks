import 'dart:math' as math;
import 'dart:typed_data';

/// Real-time voice acoustic conditioning and background noise suppressor:
/// 1. 2nd-order Butterworth High-Pass Filter (120 Hz cutoff) to eliminate fan motor rumble,
///    wind turbulence, and 50/60Hz AC mains hum.
/// 2. Hysteresis Voice Activity Noise Gate (VAD) with adaptive noise floor tracking,
///    fast attack, 280ms hold time, and raised-cosine smooth release fade.
class AudioDenoiseProcessor {
  AudioDenoiseProcessor({
    this.highPassCutoffHz = 120.0,
    this.sampleRate = 16000,
    this.speechThresholdRms = 0.012, // ~ -38.4 dBFS
    this.holdTimeMs = 280,
    this.releaseTimeMs = 45,
    this.enabled = true,
  }) {
    _initFilter();
    _holdSamplesTotal = (holdTimeMs * sampleRate / 1000).toInt();
    _releaseSamplesTotal = (releaseTimeMs * sampleRate / 1000).toInt();
  }

  final double highPassCutoffHz;
  final int sampleRate;
  double speechThresholdRms;
  final int holdTimeMs;
  final int releaseTimeMs;
  bool enabled;

  late int _holdSamplesTotal;
  late int _releaseSamplesTotal;

  // Biquad HPF coefficients
  double _b0 = 1.0;
  double _b1 = -2.0;
  double _b2 = 1.0;
  double _a1 = 0.0;
  double _a2 = 0.0;

  // Biquad filter state registers
  double _x1 = 0.0;
  double _x2 = 0.0;
  double _y1 = 0.0;
  double _y2 = 0.0;

  // Adaptive noise gate state
  double _ambientNoiseFloor = 0.005;
  int _holdSamplesRemaining = 0;
  double _currentGain = 0.0; // [0.0 = fully closed/silent, 1.0 = fully open]
  bool _isSpeechActive = false;

  bool get isSpeechActive => _isSpeechActive;
  double get ambientNoiseFloor => _ambientNoiseFloor;
  double get currentGain => _currentGain;

  void _initFilter() {
    final w0 = 2.0 * math.pi * highPassCutoffHz / sampleRate;
    final alpha = math.sin(w0) / (2.0 * 0.7071067811865475); // Q = 1 / sqrt(2)
    final cosw0 = math.cos(w0);

    final a0 = 1.0 + alpha;
    _b0 = (1.0 + cosw0) / (2.0 * a0);
    _b1 = -(1.0 + cosw0) / a0;
    _b2 = (1.0 + cosw0) / (2.0 * a0);
    _a1 = (-2.0 * cosw0) / a0;
    _a2 = (1.0 - alpha) / a0;
  }

  /// Reset all internal filter buffers and gate state.
  void reset() {
    _x1 = 0.0;
    _x2 = 0.0;
    _y1 = 0.0;
    _y2 = 0.0;
    _holdSamplesRemaining = 0;
    _currentGain = 0.0;
    _isSpeechActive = false;
    _ambientNoiseFloor = 0.005;
  }

  /// Process an incoming 16-bit PCM mono byte buffer.
  /// Applies high-pass filter and adaptive noise gating.
  /// Returns a cleaned 16-bit PCM mono Uint8List.
  Uint8List processPcmChunk(Uint8List inputPcm) {
    if (!enabled || inputPcm.length < 2) return inputPcm;

    final sampleCount = inputPcm.length ~/ 2;
    final byteData = ByteData.sublistView(inputPcm);
    final outputSamples = Int16List(sampleCount);

    // Step 1: Apply 2nd-order High-Pass Filter (strips sub-120Hz fan/AC rumble)
    final filtered = Float64List(sampleCount);
    var sumSquares = 0.0;

    for (var i = 0; i < sampleCount; i++) {
      final x0 = byteData.getInt16(i * 2, Endian.little) / 32768.0;
      final y0 = _b0 * x0 + _b1 * _x1 + _b2 * _x2 - _a1 * _y1 - _a2 * _y2;

      _x2 = _x1;
      _x1 = x0;
      _y2 = _y1;
      _y1 = y0;

      filtered[i] = y0;
      sumSquares += y0 * y0;
    }

    final chunkRms = math.sqrt(sumSquares / sampleCount);

    // Step 2: Adaptive Voice Activity Energy Evaluation
    // Maintain rolling estimate of stationary room background noise floor
    if (chunkRms < speechThresholdRms * 1.5) {
      _ambientNoiseFloor = (0.96 * _ambientNoiseFloor) + (0.04 * chunkRms);
      if (_ambientNoiseFloor < 0.001) _ambientNoiseFloor = 0.001;
    }

    // Dynamic threshold: at least speechThresholdRms, and at least 2.2x ambient noise
    final activeThreshold = math.max(speechThresholdRms, _ambientNoiseFloor * 2.2);

    final hasSpeechEnergy = chunkRms >= activeThreshold;

    if (hasSpeechEnergy) {
      _isSpeechActive = true;
      _holdSamplesRemaining = _holdSamplesTotal;
    } else {
      if (_holdSamplesRemaining > 0) {
        _holdSamplesRemaining -= sampleCount;
        if (_holdSamplesRemaining <= 0) {
          _holdSamplesRemaining = 0;
          _isSpeechActive = false;
        } else {
          _isSpeechActive = true; // In hold period, gate remains open
        }
      } else {
        _isSpeechActive = false;
      }
    }

    // Step 3: Apply Soft-Gating with smooth sample-by-sample gain interpolation
    // Target gain: 1.0 if speech active or in hold, 0.0 if background noise/silence
    final targetGain = _isSpeechActive ? 1.0 : 0.0;
    final gainStep = (_releaseSamplesTotal > 0) ? (1.0 / _releaseSamplesTotal) : 0.05;

    for (var i = 0; i < sampleCount; i++) {
      if (_currentGain < targetGain) {
        // Fast attack (~10-15ms)
        _currentGain = math.min(1.0, _currentGain + 0.015);
      } else if (_currentGain > targetGain) {
        // Smooth release (~45ms)
        _currentGain = math.max(0.0, _currentGain - gainStep);
      }

      // Smooth cosine taper when gain is near zero
      final effectiveGain = _currentGain > 0.01
          ? (0.5 * (1.0 - math.cos(math.pi * _currentGain)))
          : 0.0;

      final processedSample = (filtered[i] * effectiveGain * 32767.0).round().clamp(-32768, 32767);
      outputSamples[i] = processedSample;
    }

    return Uint8List.view(outputSamples.buffer);
  }
}
