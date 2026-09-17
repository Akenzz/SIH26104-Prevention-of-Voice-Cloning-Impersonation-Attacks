import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:record/record.dart';

import 'audio_denoise_processor.dart';
import 'pcm_resampler.dart';

enum AudioCaptureStatus {
  idle,
  requestingPermission,
  permissionDenied,
  recording,
  paused,
  error,
}

/// Service managing microphone capture in 16-bit PCM mono format at 16kHz.
///
/// Features hardware echo cancellation (`echoCancel`) and noise suppression (`noiseSuppress`)
/// enabled by default, configured with Android `voiceCommunication` source and telephony DSP.
/// Also applies real-time `AudioDenoiseProcessor` (2nd order 120Hz Butterworth High-Pass Filter
/// + adaptive hysteresis noise gate) to eliminate fan hum and silence false-positive AI detection.
class AudioCaptureService {
  AudioCaptureService({
    AudioRecorder? recorder,
    bool echoCancel = true,
    bool noiseSuppress = true,
    bool autoGain = false,
    AudioDenoiseProcessor? denoiser,
  })  : _recorderInstance = recorder,
        _echoCancel = echoCancel,
        _noiseSuppress = noiseSuppress,
        _autoGain = autoGain,
        _denoiser = denoiser ?? AudioDenoiseProcessor();

  AudioRecorder? _recorderInstance;
  AudioRecorder get _recorder => _recorderInstance ??= AudioRecorder();
  StreamSubscription<Uint8List>? _recordSubscription;
  final AudioDenoiseProcessor _denoiser;

  final _pcmChunkController = StreamController<Uint8List>.broadcast();
  final _audioLevelController = StreamController<double>.broadcast();
  final _statusController = StreamController<AudioCaptureStatus>.broadcast();

  static const double _emaAlpha = 0.3;

  AudioCaptureStatus _status = AudioCaptureStatus.idle;
  String? _errorMessage;
  double _currentLevel = 0.0;

  bool _echoCancel;
  bool _noiseSuppress;
  bool _autoGain;

  Stream<Uint8List> get pcmStream => _pcmChunkController.stream;
  Stream<double> get audioLevelStream => _audioLevelController.stream;
  Stream<AudioCaptureStatus> get statusStream => _statusController.stream;

  AudioCaptureStatus get status => _status;
  bool get isRecording => _status == AudioCaptureStatus.recording;
  String? get errorMessage => _errorMessage;
  double get currentLevel => _currentLevel;

  bool get echoCancel => _echoCancel;
  bool get noiseSuppress => _noiseSuppress;
  bool get autoGain => _autoGain;
  AudioDenoiseProcessor get denoiser => _denoiser;
  bool get denoiserEnabled => _denoiser.enabled;
  set denoiserEnabled(bool value) => _denoiser.enabled = value;

  /// Configure hardware acoustic processing toggles and software noise suppression.
  void setAcousticProcessing({
    bool? echoCancel,
    bool? noiseSuppress,
    bool? autoGain,
    bool? denoiserEnabled,
  }) {
    if (echoCancel != null) _echoCancel = echoCancel;
    if (noiseSuppress != null) _noiseSuppress = noiseSuppress;
    if (autoGain != null) _autoGain = autoGain;
    if (denoiserEnabled != null) _denoiser.enabled = denoiserEnabled;
  }

  void _setStatus(AudioCaptureStatus status, [String? error]) {
    _status = status;
    _errorMessage = error;
    _statusController.add(status);
  }

  /// Check whether microphone permission has been granted.
  Future<bool> hasPermission() async {
    try {
      return await _recorder.hasPermission();
    } catch (e) {
      debugPrint('[AudioCaptureService] hasPermission error: $e');
      return false;
    }
  }

  /// Start streaming microphone audio in 16-bit PCM mono.
  Future<bool> startStream({
    int sampleRate = 16000,
    bool? echoCancel,
    bool? noiseSuppress,
    bool? autoGain,
  }) async {
    if (isRecording) return true;

    _setStatus(AudioCaptureStatus.requestingPermission);

    final granted = await hasPermission();
    if (!granted) {
      _setStatus(
        AudioCaptureStatus.permissionDenied,
        'Microphone permission is required to capture speech for live call integrity analysis.',
      );
      return false;
    }

    final useEchoCancel = echoCancel ?? _echoCancel;
    final useNoiseSuppress = noiseSuppress ?? _noiseSuppress;
    final useAutoGain = autoGain ?? _autoGain;

    try {
      // Hardware echo cancellation and telephony noise suppression configured for voice communications
      // to eliminate acoustic feedback loops and engage hardware noise suppression.
      final recordConfig = RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
        autoGain: useAutoGain,
        echoCancel: useEchoCancel,
        noiseSuppress: useNoiseSuppress,
        androidConfig: const AndroidRecordConfig(
          audioSource: AndroidAudioSource.voiceCommunication,
          audioManagerMode: AudioManagerMode.modeInCommunication,
        ),
      );

      final stream = await _recorder.startStream(recordConfig);
      _setStatus(AudioCaptureStatus.recording);

      _recordSubscription?.cancel();
      _recordSubscription = stream.listen(
        (data) {
          if (data.isEmpty) return;

          // Resample and ensure mono 16k if device emitted different rate
          final resampled = PcmResampler.processTo16kMono(
            inputBytes: data,
            inputSampleRate: sampleRate,
            inputChannels: 1,
            targetSampleRate: 16000,
          );

          // Apply real-time acoustic high-pass filter (120Hz) and adaptive hysteresis noise gate
          final processed = _denoiser.processPcmChunk(resampled);

          // Calculate RMS level for visual VU meter with EMA smoothing
          final rawLevel = PcmResampler.calculateRmsLevel(processed);
          _currentLevel = (_emaAlpha * rawLevel) + ((1.0 - _emaAlpha) * _currentLevel);
          if (_currentLevel < 0.005) {
            _currentLevel = 0.0;
          }
          _audioLevelController.add(_currentLevel);

          _pcmChunkController.add(processed);
        },
        onError: (Object error) {
          debugPrint('[AudioCaptureService] Stream error: $error');
          _setStatus(AudioCaptureStatus.error, error.toString());
        },
        onDone: () {
          debugPrint('[AudioCaptureService] Stream completed');
          _setStatus(AudioCaptureStatus.idle);
        },
        cancelOnError: false,
      );

      return true;
    } catch (error) {
      debugPrint(
        '[AudioCaptureService] Failed to start capture stream: $error',
      );
      _setStatus(
        AudioCaptureStatus.error,
        'Failed to initialize audio input device: $error',
      );
      return false;
    }
  }

  /// Stop capturing microphone audio.
  Future<void> stopStream() async {
    try {
      await _recordSubscription?.cancel();
      _recordSubscription = null;
      if (_recorderInstance != null && await _recorder.isRecording()) {
        await _recorder.stop();
      }
    } catch (e) {
      debugPrint('[AudioCaptureService] Stop error: $e');
    } finally {
      _denoiser.reset();
      _currentLevel = 0.0;
      _audioLevelController.add(0.0);
      _setStatus(AudioCaptureStatus.idle);
    }
  }

  /// Dispose resources.
  Future<void> dispose() async {
    await stopStream();
    if (_recorderInstance != null) {
      try {
        await _recorder.dispose();
      } catch (_) {}
    }
    await _pcmChunkController.close();
    await _audioLevelController.close();
    await _statusController.close();
  }
}
