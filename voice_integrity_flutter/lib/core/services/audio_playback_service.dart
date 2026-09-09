import 'dart:async';
import 'dart:math' as math;

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
  Timer? _testDelayTimer;
  Completer<void>? _testDelayCompleter;
  bool _isDisposed = false;
  bool _isTestingSound = false;

  Stream<double> get playbackLevelStream => _playbackLevelController.stream;
  Stream<bool> get isPlayingStream => _isPlayingController.stream;

  bool get isPlaying => _isPlaying;
  bool get isMuted => _isMuted;
  bool get isTestingSound => _isTestingSound;
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
    _testDelayTimer?.cancel();
    _testDelayTimer = null;
    if (_testDelayCompleter?.isCompleted == false) {
      _testDelayCompleter?.complete();
    }
    _testDelayCompleter = null;
    _isTestingSound = false;
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

  /// Generates a pleasant, clear 3-note harmonic chime (C5: 523.25Hz for 160ms,
  /// E5: 659.25Hz for 160ms, G5: 783.99Hz for 320ms) at 16000Hz 16-bit mono signed
  /// PCM with a smooth attack/decay envelope on each note to eliminate any clicks or pops.
  static Uint8List generateChimePcm({int sampleRate = 16000}) {
    // 3 notes: C5 (523.25Hz, 160ms), E5 (659.25Hz, 160ms), G5 (783.99Hz, 320ms)
    final notes = [
      (frequency: 523.25, durationMs: 160, isFinal: false),
      (frequency: 659.25, durationMs: 160, isFinal: false),
      (frequency: 783.99, durationMs: 320, isFinal: true),
    ];

    final totalSamples = notes.fold<int>(
      0,
      (sum, note) => sum + (note.durationMs * sampleRate ~/ 1000),
    );

    final pcmBytes = Uint8List(totalSamples * 2);
    final byteData = ByteData.view(pcmBytes.buffer);

    var sampleOffset = 0;
    for (final note in notes) {
      final noteSamples = note.durationMs * sampleRate ~/ 1000;
      final freq = note.frequency;
      final isFinal = note.isFinal;

      // Attack: 15ms (smooth cosine rise from 0 to 1)
      final attackSamples =
          (0.015 * sampleRate).round().clamp(1, noteSamples ~/ 2);
      // Decay: 20ms for interim notes, 120ms gentle fade for final chime note
      final decaySamples = isFinal
          ? (0.120 * sampleRate).round().clamp(1, noteSamples ~/ 2)
          : (0.020 * sampleRate).round().clamp(1, noteSamples ~/ 2);

      for (var i = 0; i < noteSamples; i++) {
        final t = i / sampleRate;

        // Rich harmonic chime timbre: fundamental + 2nd and 3rd harmonics
        final harmonicWave =
            (math.sin(2 * math.pi * freq * t) +
                0.25 * math.sin(4 * math.pi * freq * t) +
                0.08 * math.sin(6 * math.pi * freq * t)) /
            1.33;

        // Smooth envelope to eliminate any clicks or pops
        double envelope;
        if (i < attackSamples) {
          envelope = 0.5 * (1.0 - math.cos(math.pi * i / attackSamples));
        } else if (i >= noteSamples - decaySamples) {
          final decayProgress =
              (i - (noteSamples - decaySamples)) / decaySamples;
          final base = isFinal ? 0.7 : 1.0;
          envelope = base * 0.5 * (1.0 + math.cos(math.pi * decayProgress));
        } else if (isFinal) {
          final sustainProgress =
              (i - attackSamples) /
              (noteSamples - decaySamples - attackSamples);
          envelope = 1.0 - 0.3 * sustainProgress;
        } else {
          envelope = 1.0;
        }

        const maxAmplitude = 18000.0;
        final sampleVal = (harmonicWave * envelope * maxAmplitude)
            .round()
            .clamp(-32768, 32767);

        byteData.setInt16((sampleOffset + i) * 2, sampleVal, Endian.little);
      }
      sampleOffset += noteSamples;
    }

    return pcmBytes;
  }

  /// Plays a pleasant, clear 3-note harmonic chime through the exact playback pipeline
  /// so users can verify speaker output and VU meter functionality.
  ///
  /// - If the player is disposed, returns immediately.
  /// - If [_isMuted], unmutes first so the sound is audible.
  /// - Ingests the PCM into [ingestPcmChunk] in 100ms (3200 bytes) slices,
  ///   flowing through the exact jitter buffer, WAV encapsulation, AudioPlayer speaker
  ///   playback, and VU meter level stream that live call audio uses.
  /// - Returns when the sound has been dispatched.
  Future<void> playTestSound({
    Duration chunkDelay = const Duration(milliseconds: 100),
  }) async {
    if (_isDisposed || _isTestingSound) return;
    _isTestingSound = true;

    try {
      if (_isMuted) {
        await toggleMute();
      }
      if (_isDisposed) return;

      final pcm = generateChimePcm();
      const chunkSize = 3200; // 100ms at 16kHz 16-bit mono

      for (var offset = 0; offset < pcm.length; offset += chunkSize) {
        if (_isDisposed) break;

        final end = (offset + chunkSize < pcm.length)
            ? offset + chunkSize
            : pcm.length;
        final slice = Uint8List.sublistView(pcm, offset, end);
        ingestPcmChunk(slice);

        if (offset + chunkSize < pcm.length && chunkDelay > Duration.zero) {
          final completer = Completer<void>();
          _testDelayCompleter = completer;
          _testDelayTimer = Timer(chunkDelay, () {
            _testDelayTimer = null;
            if (!completer.isCompleted) completer.complete();
          });
          await completer.future;
          _testDelayCompleter = null;
        }
      }
      // Brief yield to ensure all queued stream events are delivered
      await Future<void>.microtask(() {});
    } finally {
      _isTestingSound = false;
    }
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
