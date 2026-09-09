import 'dart:async';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/foundation.dart';

import 'pcm_resampler.dart';

/// Service managing real-time PCM audio playback through the device speaker
/// with a ~300ms jitter buffer to ensure smooth acoustic reproduction.
class AudioPlaybackService {
  AudioPlaybackService({AudioPlayer? player}) : _playerInstance = player {
    if (player != null) {
      _setupPlayer(player);
    }
  }

  AudioPlayer? _playerInstance;
  bool _playerConfigured = false;
  AudioPlayer get _player {
    final p = _playerInstance ??= AudioPlayer();
    if (!_playerConfigured) {
      _playerConfigured = true;
      _setupPlayer(p);
    }
    return p;
  }

  final _playbackLevelController = StreamController<double>.broadcast();
  final _isPlayingController = StreamController<bool>.broadcast();

  // ~300ms jitter buffer at 16kHz 16-bit mono:
  // 16000 samples/sec * 2 bytes/sample * 0.300 sec = 9600 bytes.
  static const int _jitterBufferTargetBytes = 9600;

  // Immediate playback threshold (~100ms) when onPlayerComplete fires
  static const int _minPlayChunkBytes = 3200;

  // Smoothing factor for VU meter exponential moving average
  static const double _emaAlpha = 0.3;

  final List<int> _jitterBuffer = [];
  bool _isPlaying = false;
  bool _isMuted = false;
  double _volume = 1.0;
  double _currentLevel = 0.0;
  Timer? _drainTimer;
  Timer? _decayTimer;
  bool _isDisposed = false;

  Stream<double> get playbackLevelStream => _playbackLevelController.stream;
  Stream<bool> get isPlayingStream => _isPlayingController.stream;

  bool get isPlaying => _isPlaying;
  bool get isMuted => _isMuted;
  double get volume => _volume;
  double get currentLevel => _currentLevel;

  void _setupPlayer(AudioPlayer player) {
    // Critical Routing Configuration:
    // Route to loud speaker (music/media stream) instead of quiet phone call earpiece!
    try {
      player
          .setAudioContext(
            AudioContext(
              iOS: AudioContextIOS(
                category: AVAudioSessionCategory.playAndRecord,
                options: const {
                  AVAudioSessionOptions.defaultToSpeaker,
                  AVAudioSessionOptions.allowBluetooth,
                  AVAudioSessionOptions.allowBluetoothA2DP,
                },
              ),
              android: const AudioContextAndroid(
                isSpeakerphoneOn: true,
                stayAwake: true,
                contentType: AndroidContentType.music,
                usageType: AndroidUsageType.media,
                audioFocus: AndroidAudioFocus.gain,
              ),
            ),
          )
          .catchError((Object e) {
            debugPrint(
              '[AudioPlaybackService] AudioContext configuration note: $e',
            );
          });
    } catch (e) {
      debugPrint('[AudioPlaybackService] AudioContext configuration note: $e');
    }

    player.onPlayerComplete.listen((_) {
      _isPlaying = false;
      _isPlayingController.add(false);
      _checkAndPlayNextChunk();
    });
  }

  /// Ingest an incoming PCM chunk from WebSocket stream.
  void ingestPcmChunk(Uint8List pcmChunk) {
    if (_isDisposed || _isMuted || pcmChunk.isEmpty) return;

    // Cancel decay timer since fresh audio packets have arrived
    _decayTimer?.cancel();
    _decayTimer = null;

    // Calculate incoming playback level for visual VU meter with EMA smoothing
    final rawLevel = PcmResampler.calculateRmsLevel(pcmChunk);
    _currentLevel = (_emaAlpha * rawLevel) + ((1.0 - _emaAlpha) * _currentLevel);
    if (_currentLevel < 0.005) _currentLevel = 0.0;
    _playbackLevelController.add(_currentLevel);

    // Accumulate incoming PCM chunks into jitter buffer
    _jitterBuffer.addAll(pcmChunk);

    // Safety cap jitter buffer (e.g. 2 seconds = 64,000 bytes) to prevent unbounded memory
    if (_jitterBuffer.length > 64000) {
      _jitterBuffer.removeRange(0, _jitterBuffer.length - 64000);
    }

    // If we've reached the ~300ms jitter queue target and not currently playing, drain
    if (_jitterBuffer.length >= _jitterBufferTargetBytes && !_isPlaying) {
      _drainTimer?.cancel();
      _drainTimer = null;
      _playQueuedBuffer();
    } else if (!_isPlaying) {
      // Set a deadline timer (320ms) to ensure small chunks aren't stranded
      _drainTimer ??= Timer(const Duration(milliseconds: 320), () {
        _drainTimer = null;
        if (!_isPlaying && _jitterBuffer.isNotEmpty) {
          _playQueuedBuffer();
        }
      });
    }
  }

  Future<void> _playQueuedBuffer() async {
    if (_isDisposed || _isMuted || _isPlaying || _jitterBuffer.isEmpty) return;
    if (_jitterBuffer.length < 320) return; // Too short to play

    // Take current buffer
    final pcmBytes = Uint8List.fromList(_jitterBuffer);
    _jitterBuffer.clear();

    final wavData = PcmResampler.createWavContainer(
      pcmData: pcmBytes,
      sampleRate: 16000,
      channels: 1,
      bitsPerSample: 16,
    );

    try {
      _isPlaying = true;
      _isPlayingController.add(true);
      await _player.setVolume(_volume);
      await _player.play(BytesSource(wavData));
    } catch (e) {
      debugPrint('[AudioPlaybackService] Playback chunk error: $e');
      _isPlaying = false;
      _isPlayingController.add(false);
      _checkAndPlayNextChunk();
    }
  }

  void _checkAndPlayNextChunk() {
    if (_isDisposed || _isMuted) return;

    if (_jitterBuffer.length >= _minPlayChunkBytes) {
      // Immediately play back accumulated audio without waiting for a new trigger
      _drainTimer?.cancel();
      _drainTimer = null;
      _playQueuedBuffer();
    } else if (_jitterBuffer.isNotEmpty) {
      // Small buffer left (< 100ms), wait briefly for more chunks or drain
      _drainTimer?.cancel();
      _drainTimer = Timer(const Duration(milliseconds: 100), () {
        _drainTimer = null;
        if (!_isPlaying && _jitterBuffer.isNotEmpty) {
          _playQueuedBuffer();
        } else if (!_isPlaying && _jitterBuffer.isEmpty) {
          _startDecay();
        }
      });
      _startDecay();
    } else {
      // Temporarily empty buffer: decay RMS level smoothly instead of abrupt 0.0
      _startDecay();
    }
  }

  void _startDecay() {
    _decayTimer?.cancel();
    _decayTimer = Timer.periodic(const Duration(milliseconds: 50), (timer) {
      _currentLevel *= 0.75;
      if (_currentLevel < 0.01) {
        _currentLevel = 0.0;
        timer.cancel();
        _decayTimer = null;
      }
      _playbackLevelController.add(_currentLevel);
    });
  }

  /// Toggle speaker mute.
  Future<void> toggleMute() async {
    _isMuted = !_isMuted;
    if (_isMuted) {
      _drainTimer?.cancel();
      _drainTimer = null;
      _decayTimer?.cancel();
      _decayTimer = null;
      _jitterBuffer.clear();
      await _player.stop();
      _isPlaying = false;
      _isPlayingController.add(false);
      _currentLevel = 0.0;
      _playbackLevelController.add(0.0);
    }
  }

  /// Adjust playback volume (0.0 to 1.0).
  Future<void> setVolume(double vol) async {
    _volume = vol.clamp(0.0, 1.0);
    if (!_isMuted) {
      await _player.setVolume(_volume);
    }
  }

  /// Clear the buffer and stop current playback.
  Future<void> stop() async {
    _drainTimer?.cancel();
    _drainTimer = null;
    _decayTimer?.cancel();
    _decayTimer = null;
    _jitterBuffer.clear();
    if (_playerInstance != null) {
      try {
        await _playerInstance!.stop();
      } catch (_) {}
    }
    _isPlaying = false;
    _currentLevel = 0.0;
    _playbackLevelController.add(0.0);
    _isPlayingController.add(false);
  }

  /// Dispose player resources.
  Future<void> dispose() async {
    _isDisposed = true;
    await stop();
    if (_playerInstance != null) {
      try {
        await _playerInstance!.dispose();
      } catch (_) {}
    }
    await _playbackLevelController.close();
    await _isPlayingController.close();
  }
}
