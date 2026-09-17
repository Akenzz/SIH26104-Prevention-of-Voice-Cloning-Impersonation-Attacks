import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/foundation.dart';

import 'pcm_resampler.dart';

/// Service managing real-time PCM audio playback through the device speaker
/// using a continuous local loopback HTTP streaming server and a smooth jitter buffer.
///
/// Eliminates the native audio pipeline restarts and 50-100ms silence gaps caused by
/// repeatedly creating WAV containers and calling `play(BytesSource)` for each chunk.
class AudioPlaybackService {
  AudioPlaybackService({
    AudioPlayer? player,
    bool? enableStreamingServer,
  }) : _playerInstance = player,
       _fallbackToBytesSource = enableStreamingServer == false {
    if (player != null) {
      _playerConfigured = true;
      _setupPlayer(player);
    }
  }

  AudioPlayer? _playerInstance;
  bool _playerConfigured = false;
  StreamSubscription<void>? _playerCompleteSub;
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

  // ~200-300ms jitter buffer at 16kHz 16-bit mono:
  // 16000 samples/sec * 2 bytes/sample * 0.200 sec = 6400 bytes.
  static const int _jitterBufferTargetBytes = 6400;

  // Immediate playback threshold (~100ms)
  static const int _minPlayChunkBytes = 3200;

  // Smoothing factor for VU meter exponential moving average
  static const double _emaAlpha = 0.3;

  // Local loopback streaming HTTP server
  HttpServer? _server;
  int? _serverPort;
  bool _serverStarting = false;
  bool _serverFailed = false;
  bool _fallbackToBytesSource;
  bool _isHttpStreaming = false;
  HttpResponse? _activeResponse;

  final List<int> _jitterBuffer = [];
  bool _isPlaying = false;
  bool _isMuted = false;
  bool _needsFadeIn = true;
  double _volume = 1.0;
  double _currentLevel = 0.0;
  Timer? _drainTimer;
  Timer? _decayTimer;
  Timer? _underrunTimer;
  Timer? _testDelayTimer;
  Completer<void>? _testDelayCompleter;
  bool _isDisposed = false;
  bool _isTestingSound = false;

  Stream<double> get playbackLevelStream => _playbackLevelController.stream;
  Stream<double> get speakerAudioLevelStream => _playbackLevelController.stream;
  Stream<bool> get isPlayingStream => _isPlayingController.stream;

  bool get isPlaying => _isPlaying;
  bool get isMuted => _isMuted;
  bool get isTestingSound => _isTestingSound;
  bool get isHttpStreaming => _isHttpStreaming;
  bool get fallbackToBytesSource => _fallbackToBytesSource;
  int? get loopbackPort => _serverPort;
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

    _playerCompleteSub = player.onPlayerComplete.listen((_) {
      _isPlaying = false;
      _isHttpStreaming = false;
      _activeResponse = null;
      _needsFadeIn = true;
      if (!_isDisposed && !_isPlayingController.isClosed) {
        _isPlayingController.add(false);
      }
      _checkAndPlayNextChunk();
    });
  }

  /// Ensure the local loopback HTTP streaming server is bound and listening.
  Future<bool> _ensureServerStarted() async {
    if (_server != null) return true;
    if (_serverStarting) {
      while (_serverStarting) {
        await Future<void>.delayed(const Duration(milliseconds: 10));
      }
      return _server != null;
    }
    if (_serverFailed) return false;

    _serverStarting = true;
    try {
      _server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      _serverPort = _server!.port;
      _server!.listen(
        _handleHttpRequest,
        onError: (Object e) {
          debugPrint('[AudioPlaybackService] Loopback server error: $e');
        },
      );
      _serverStarting = false;
      return true;
    } catch (e) {
      debugPrint('[AudioPlaybackService] Failed to bind loopback server: $e');
      _serverFailed = true;
      _serverStarting = false;
      return false;
    }
  }

  /// Handle incoming HTTP request from the audio player to stream WAV data.
  Future<void> _handleHttpRequest(HttpRequest request) async {
    if (_isDisposed) {
      request.response.statusCode = HttpStatus.serviceUnavailable;
      await request.response.close();
      return;
    }

    if (request.uri.path != '/stream.wav') {
      request.response.statusCode = HttpStatus.notFound;
      await request.response.close();
      return;
    }

    // Clean up any stale active response
    if (_activeResponse != null) {
      try {
        await _activeResponse!.close();
      } catch (_) {}
      _activeResponse = null;
    }

    final response = request.response;
    _activeResponse = response;
    _isHttpStreaming = true;

    response.headers.contentType = ContentType('audio', 'x-wav');
    response.headers.chunkedTransferEncoding = true;
    response.headers.set('Accept-Ranges', 'none');
    response.headers.set('Cache-Control', 'no-cache, no-store, must-revalidate');

    // Endless 44-byte WAV header with 0x7FFFFFFF data size
    final header = createStreamingWavHeader(
      sampleRate: 16000,
      channels: 1,
      bitsPerSample: 16,
    );
    response.add(header);

    // Drain currently buffered jitter data into the HTTP response stream immediately
    if (_jitterBuffer.isNotEmpty) {
      final initialBytes = Uint8List.fromList(_jitterBuffer);
      _jitterBuffer.clear();
      if (_needsFadeIn) {
        applyRaisedCosineFadeIn(initialBytes, fadeSamples: 160);
        _needsFadeIn = false;
      }
      response.add(initialBytes);
    }

    try {
      await response.flush();
    } catch (_) {}

    // Listen for client disconnect or stream closure
    response.done.then((_) {
      if (_activeResponse == response) {
        _activeResponse = null;
        _isHttpStreaming = false;
      }
    }).catchError((_) {
      if (_activeResponse == response) {
        _activeResponse = null;
        _isHttpStreaming = false;
      }
    });
  }

  /// Creates a streaming 44-byte RIFF WAV header with 0x7FFFFFFF size for continuous playback.
  static Uint8List createStreamingWavHeader({
    int sampleRate = 16000,
    int channels = 1,
    int bitsPerSample = 16,
  }) {
    final byteRate = sampleRate * channels * (bitsPerSample ~/ 8);
    final blockAlign = channels * (bitsPerSample ~/ 8);
    const maxLen = 0x7FFFFFFF;

    final header = Uint8List(44);
    final b = ByteData.view(header.buffer);

    // RIFF header
    b.setUint8(0, 0x52); // 'R'
    b.setUint8(1, 0x49); // 'I'
    b.setUint8(2, 0x46); // 'F'
    b.setUint8(3, 0x46); // 'F'
    b.setUint32(4, maxLen, Endian.little);
    b.setUint8(8, 0x57); // 'W'
    b.setUint8(9, 0x41); // 'A'
    b.setUint8(10, 0x56); // 'V'
    b.setUint8(11, 0x45); // 'E'

    // fmt subchunk
    b.setUint8(12, 0x66); // 'f'
    b.setUint8(13, 0x6D); // 'm'
    b.setUint8(14, 0x74); // 't'
    b.setUint8(15, 0x20); // ' '
    b.setUint32(16, 16, Endian.little); // Subchunk1Size (16 for PCM)
    b.setUint16(20, 1, Endian.little); // AudioFormat (1 for PCM)
    b.setUint16(22, channels, Endian.little);
    b.setUint32(24, sampleRate, Endian.little);
    b.setUint32(28, byteRate, Endian.little);
    b.setUint16(32, blockAlign, Endian.little);
    b.setUint16(34, bitsPerSample, Endian.little);

    // data subchunk
    b.setUint8(36, 0x64); // 'd'
    b.setUint8(37, 0x61); // 'a'
    b.setUint8(38, 0x74); // 't'
    b.setUint8(39, 0x61); // 'a'
    b.setUint32(40, maxLen, Endian.little);

    return header;
  }

  /// Applies a smooth raised-cosine fade-out to the last [fadeSamples] (default 160 = 10ms at 16k).
  static void applyRaisedCosineFadeOut(Uint8List pcm, {int fadeSamples = 160}) {
    final totalSamples = pcm.length ~/ 2;
    if (totalSamples < 2) return;
    final actualFade = fadeSamples.clamp(1, totalSamples);
    final startIndex = totalSamples - actualFade;
    final byteData = ByteData.sublistView(pcm);
    for (var i = 0; i < actualFade; i++) {
      final sampleIdx = startIndex + i;
      final originalVal = byteData.getInt16(sampleIdx * 2, Endian.little);
      final gain = 0.5 * (1.0 + math.cos(math.pi * i / actualFade));
      final fadedVal = (originalVal * gain).round().clamp(-32768, 32767);
      byteData.setInt16(sampleIdx * 2, fadedVal, Endian.little);
    }
  }

  /// Applies a smooth raised-cosine fade-in to the first [fadeSamples] (default 160 = 10ms at 16k).
  static void applyRaisedCosineFadeIn(Uint8List pcm, {int fadeSamples = 160}) {
    final totalSamples = pcm.length ~/ 2;
    if (totalSamples < 2) return;
    final actualFade = fadeSamples.clamp(1, totalSamples);
    final byteData = ByteData.sublistView(pcm);
    for (var i = 0; i < actualFade; i++) {
      final originalVal = byteData.getInt16(i * 2, Endian.little);
      final gain = 0.5 * (1.0 - math.cos(math.pi * i / actualFade));
      final fadedVal = (originalVal * gain).round().clamp(-32768, 32767);
      byteData.setInt16(i * 2, fadedVal, Endian.little);
    }
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

    _resetUnderrunTimer();

    // If HTTP streaming response is actively listening, pipe chunk directly
    if (_isHttpStreaming && _activeResponse != null) {
      final chunkToSend = Uint8List.fromList(pcmChunk);
      if (_needsFadeIn) {
        applyRaisedCosineFadeIn(chunkToSend, fadeSamples: 160);
        _needsFadeIn = false;
      }
      try {
        _activeResponse!.add(chunkToSend);
        _activeResponse!.flush().catchError((_) {});
      } catch (e) {
        debugPrint('[AudioPlaybackService] Error streaming chunk: $e');
        _activeResponse = null;
        _isHttpStreaming = false;
      }
      return;
    }

    // Accumulate incoming PCM chunks into jitter buffer
    _jitterBuffer.addAll(pcmChunk);

    // Safety cap jitter buffer (2 seconds = 64,000 bytes) to prevent unbounded memory.
    // Truncation MUST drop an even number of bytes to preserve 16-bit mono sample boundaries!
    if (_jitterBuffer.length > 64000) {
      final dropCount = _jitterBuffer.length - 64000;
      final alignedDropCount = dropCount + (dropCount % 2);
      _jitterBuffer.removeRange(0, alignedDropCount);
    }

    // If we've reached the ~200-300ms jitter queue target and not currently playing, start
    if (_jitterBuffer.length >= _jitterBufferTargetBytes && !_isPlaying) {
      _drainTimer?.cancel();
      _drainTimer = null;
      _startPlayback();
    } else if (!_isPlaying) {
      // Set a deadline timer (250ms) to ensure small chunks aren't stranded
      _drainTimer ??= Timer(const Duration(milliseconds: 250), () {
        _drainTimer = null;
        if (!_isPlaying && _jitterBuffer.isNotEmpty) {
          _startPlayback();
        }
      });
    }
  }

  void _resetUnderrunTimer() {
    _underrunTimer?.cancel();
    _underrunTimer = Timer(const Duration(milliseconds: 300), () {
      _underrunTimer = null;
      // Mark for smooth raised-cosine fade-in when stream resumes
      _needsFadeIn = true;
      if (!_isDisposed) {
        _startDecay();
      }
    });
  }

  /// Start playback through continuous loopback HTTP streaming or fallback.
  Future<void> _startPlayback() async {
    if (_isDisposed || _isMuted || _isPlaying || _jitterBuffer.isEmpty) return;

    if (_fallbackToBytesSource) {
      await _playQueuedBufferBytesSource();
      return;
    }

    final serverReady = await _ensureServerStarted();
    if (!serverReady) {
      _fallbackToBytesSource = true;
      await _playQueuedBufferBytesSource();
      return;
    }

    try {
      _isPlaying = true;
      _isPlayingController.add(true);
      await _player.setVolume(_volume);
      final streamUrl = 'http://127.0.0.1:$_serverPort/stream.wav';
      await _player.play(UrlSource(streamUrl));
    } catch (e) {
      debugPrint('[AudioPlaybackService] Play stream error: $e, falling back to BytesSource');
      _fallbackToBytesSource = true;
      _isPlaying = false;
      _isHttpStreaming = false;
      _activeResponse = null;
      await _playQueuedBufferBytesSource();
    }
  }

  /// Graceful fallback for mock/test environments or when loopback server cannot bind.
  Future<void> _playQueuedBufferBytesSource() async {
    if (_isDisposed || _isMuted || _isPlaying || _jitterBuffer.isEmpty) return;
    if (_jitterBuffer.length < 320) return; // Too short to play

    final pcmBytes = Uint8List.fromList(_jitterBuffer);
    _jitterBuffer.clear();

    if (_needsFadeIn) {
      applyRaisedCosineFadeIn(pcmBytes, fadeSamples: 160);
      _needsFadeIn = false;
    }

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
      if (!_isDisposed && !_isPlayingController.isClosed) {
        _isPlayingController.add(false);
      }
      _checkAndPlayNextChunk();
    }
  }

  void _checkAndPlayNextChunk() {
    if (_isDisposed || _isMuted) return;

    if (_jitterBuffer.length >= _minPlayChunkBytes) {
      _drainTimer?.cancel();
      _drainTimer = null;
      _startPlayback();
    } else if (_jitterBuffer.isNotEmpty) {
      _drainTimer?.cancel();
      _drainTimer = Timer(const Duration(milliseconds: 100), () {
        _drainTimer = null;
        if (!_isPlaying && _jitterBuffer.isNotEmpty) {
          _startPlayback();
        } else if (!_isPlaying && _jitterBuffer.isEmpty) {
          _startDecay();
        }
      });
      _startDecay();
    } else {
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
      await stop();
    } else {
      _needsFadeIn = true;
    }
  }

  /// Explicitly set speaker mute state.
  Future<void> setMuted(bool muted) async {
    if (_isMuted == muted) return;
    await toggleMute();
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
    _underrunTimer?.cancel();
    _underrunTimer = null;
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
    _needsFadeIn = true;

    if (_activeResponse != null) {
      try {
        await _activeResponse!.close();
      } catch (_) {}
      _activeResponse = null;
    }
    _isHttpStreaming = false;

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
        if (_isDisposed || !_isTestingSound) break;

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
      await Future<void>.microtask(() {});
    } finally {
      _isTestingSound = false;
    }
  }

  /// Dispose player and server resources.
  Future<void> dispose() async {
    _isDisposed = true;
    await _playerCompleteSub?.cancel();
    _playerCompleteSub = null;
    await stop();
    if (_server != null) {
      try {
        await _server!.close(force: true);
      } catch (_) {}
      _server = null;
    }
    if (_playerInstance != null) {
      try {
        await _playerInstance!.dispose();
      } catch (_) {}
    }
    await _playbackLevelController.close();
    await _isPlayingController.close();
  }
}
