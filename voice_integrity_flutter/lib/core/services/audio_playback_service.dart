import 'dart:async';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/foundation.dart';

import 'pcm_resampler.dart';

/// Service managing real-time PCM audio playback through the device speaker
/// with a ~120ms jitter buffer to ensure smooth acoustic reproduction.
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

  // 120ms jitter buffer at 16kHz 16-bit mono:
  // 16000 samples/sec * 2 bytes/sample * 0.120 sec = 3840 bytes.
  static const int _jitterBufferTargetBytes = 3840;

  final List<int> _jitterBuffer = [];
  bool _isPlaying = false;
  bool _isMuted = false;
  double _volume = 1.0;
  double _currentLevel = 0.0;
  Timer? _drainTimer;
  bool _isDisposed = false;

  Stream<double> get playbackLevelStream => _playbackLevelController.stream;
  Stream<bool> get isPlayingStream => _isPlayingController.stream;

  bool get isPlaying => _isPlaying;
  bool get isMuted => _isMuted;
  double get volume => _volume;
  double get currentLevel => _currentLevel;

  void _setupPlayer(AudioPlayer player) {
    // Critical Red Team Mitigation:
    // Route to loud speaker instead of quiet phone earpiece!
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
                contentType: AndroidContentType.speech,
                usageType: AndroidUsageType.voiceCommunication,
                audioFocus: AndroidAudioFocus.gainTransientMayDuck,
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

    // Calculate incoming playback level for visual VU meter
    final level = PcmResampler.calculateRmsLevel(pcmChunk);
    _currentLevel = level;
    _playbackLevelController.add(level);

    _jitterBuffer.addAll(pcmChunk);

    // If we've reached the ~120ms jitter queue target and not currently playing, drain
    if (_jitterBuffer.length >= _jitterBufferTargetBytes && !_isPlaying) {
      _drainTimer?.cancel();
      _playQueuedBuffer();
    } else if (!_isPlaying) {
      // Set a short deadline timer (140ms) to ensure small chunks aren't stranded
      _drainTimer ??= Timer(const Duration(milliseconds: 140), () {
        _drainTimer = null;
        if (!_isPlaying && _jitterBuffer.isNotEmpty) {
          _playQueuedBuffer();
        }
      });
    }
  }

  Future<void> _playQueuedBuffer() async {
    if (_isDisposed || _isMuted || _jitterBuffer.isEmpty) return;

    // Take current buffer
    final pcmBytes = Uint8List.fromList(_jitterBuffer);
    _jitterBuffer.clear();

    if (pcmBytes.length < 320) return; // Too short to play

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
    }
  }

  void _checkAndPlayNextChunk() {
    if (_jitterBuffer.isNotEmpty && !_isMuted && !_isDisposed) {
      _playQueuedBuffer();
    } else {
      _currentLevel = 0.0;
      _playbackLevelController.add(0.0);
    }
  }

  /// Toggle speaker mute.
  Future<void> toggleMute() async {
    _isMuted = !_isMuted;
    if (_isMuted) {
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
