import 'dart:async';
import 'dart:typed_data';

import 'package:audioplayers_platform_interface/audioplayers_platform_interface.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:voice_integrity_flutter/core/models/live_call_models.dart';
import 'package:voice_integrity_flutter/core/services/audio_capture_service.dart';
import 'package:voice_integrity_flutter/core/services/audio_playback_service.dart';
import 'package:voice_integrity_flutter/core/services/live_call_service.dart';
import 'package:voice_integrity_flutter/core/services/pcm_resampler.dart';
import 'package:voice_integrity_flutter/features/live_call/live_call_screen.dart';

class MockAudioplayersPlatform extends AudioplayersPlatformInterface {
  final Map<String, StreamController<AudioEvent>> _eventControllers = {};

  StreamController<AudioEvent> _getOrCreateController(String playerId) {
    return _eventControllers.putIfAbsent(
      playerId,
      () => StreamController<AudioEvent>.broadcast(),
    );
  }

  @override
  Future<void> create(String playerId) async {
    _getOrCreateController(playerId);
  }

  @override
  Future<void> dispose(String playerId) async {
    final controller = _eventControllers.remove(playerId);
    await controller?.close();
  }

  @override
  Future<void> emitError(String playerId, String code, String message) async {}

  @override
  Future<void> emitLog(String playerId, String message) async {}

  @override
  Future<int?> getCurrentPosition(String playerId) async => 0;

  @override
  Future<int?> getDuration(String playerId) async => 0;

  @override
  Future<void> pause(String playerId) async {}

  @override
  Future<void> release(String playerId) async {}

  @override
  Future<void> resume(String playerId) async {}

  @override
  Future<void> seek(String playerId, Duration position) async {}

  @override
  Future<void> setAudioContext(
    String playerId,
    AudioContext audioContext,
  ) async {}

  @override
  Future<void> setBalance(String playerId, double balance) async {}

  @override
  Future<void> setPlaybackRate(String playerId, double playbackRate) async {}

  @override
  Future<void> setPlayerMode(String playerId, PlayerMode playerMode) async {}

  @override
  Future<void> setReleaseMode(String playerId, ReleaseMode releaseMode) async {}

  @override
  Future<void> setSourceBytes(
    String playerId,
    Uint8List bytes, {
    String? mimeType,
  }) async {
    _getOrCreateController(playerId).add(
      const AudioEvent(eventType: AudioEventType.prepared, isPrepared: true),
    );
  }

  @override
  Future<void> setSourceUrl(
    String playerId,
    String url, {
    bool? isLocal,
    String? mimeType,
  }) async {
    _getOrCreateController(playerId).add(
      const AudioEvent(eventType: AudioEventType.prepared, isPrepared: true),
    );
  }

  @override
  Future<void> setVolume(String playerId, double volume) async {}

  @override
  Future<void> stop(String playerId) async {}

  @override
  Stream<AudioEvent> getEventStream(String playerId) {
    return _getOrCreateController(playerId).stream;
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUpAll(() {
    AudioplayersPlatformInterface.instance = MockAudioplayersPlatform();

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('com.llfbandit.record/messages'),
          (MethodCall methodCall) async {
            if (methodCall.method == 'create') return 'mock-id';
            if (methodCall.method == 'hasPermission') return true;
            if (methodCall.method == 'isRecording') return false;
            if (methodCall.method == 'start') return null;
            if (methodCall.method == 'stop') return null;
            if (methodCall.method == 'dispose') return null;
            return null;
          },
        );
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('xyz.luan/audioplayers.global'),
          (MethodCall methodCall) async => null,
        );
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          const MethodChannel('xyz.luan/audioplayers'),
          (MethodCall methodCall) async => null,
        );
  });

  group('LiveCallModels', () {
    test('LiveRiskLevel parses wire backend states correctly', () {
      expect(LiveRiskLevel.fromBackendString('low'), LiveRiskLevel.low);
      expect(
        LiveRiskLevel.fromBackendString('uncertain'),
        LiveRiskLevel.uncertain,
      );
      expect(LiveRiskLevel.fromBackendString('high'), LiveRiskLevel.high);
      expect(
        LiveRiskLevel.fromBackendString('collecting'),
        LiveRiskLevel.unavailable,
      );
      expect(
        LiveRiskLevel.fromBackendString('unavailable'),
        LiveRiskLevel.unavailable,
      );
      expect(
        LiveRiskLevel.fromBackendString('transitioning'),
        LiveRiskLevel.transitioning,
      );
      expect(
        LiveRiskLevel.fromBackendString('unknown_noise'),
        LiveRiskLevel.unavailable,
      );
    });

    test('LiveRiskAssessment parses score JSON correctly', () {
      final json = {
        'type': 'score',
        'risk_state': 'high',
        'smoothed_probability': 0.892,
        'fused_logit': 2.45,
        'recommended_action': 'Pause call immediately and verify caller.',
        'reasoning':
            'Spectral centroid and LFCC energy distributions match synthetic vocoder.',
        'window_index': 4,
        'latency_ms': 35.5,
        'is_spoofed': true,
        'scores': {
          'wavlm': {
            'probability': 0.85,
            'risk_state': 'high',
            'logit': 2.1,
            'label': 'WavLM Base+',
          },
          'hybrid': {
            'probability': 0.92,
            'risk_state': 'high',
            'logit': 2.8,
            'label': 'LFCC-LCNN Hybrid',
          },
          'ssl': {
            'probability': 0.88,
            'risk_state': 'high',
            'logit': 2.3,
            'label': 'TakHemlata SSL',
          },
        },
      };

      final assessment = LiveRiskAssessment.fromJson(json);

      expect(assessment.riskLevel, LiveRiskLevel.high);
      expect(assessment.probability, closeTo(0.892, 0.001));
      expect(assessment.fusedLogit, closeTo(2.45, 0.001));
      expect(assessment.windowIndex, 4);
      expect(assessment.latencyMs, 35.5);
      expect(assessment.isSpoofedByCaller, isTrue);
      expect(assessment.expertScores.length, 3);
      expect(assessment.expertScores['wavlm']?.probability, 0.85);
      expect(assessment.expertScores['hybrid']?.probability, 0.92);
      expect(assessment.expertScores['ssl']?.probability, 0.88);
    });

    test(
      'LiveRiskAssessment transitioning factory initializes amber buffer state',
      () {
        final trans = LiveRiskAssessment.transitioning(isSpoofTarget: true);
        expect(trans.riskLevel, LiveRiskLevel.transitioning);
        expect(trans.reasoning, contains('Analyzing Buffer'));
        expect(trans.isSpoofedByCaller, isTrue);
      },
    );

    test('CallStats copyWith updates properties', () {
      const stats = CallStats();
      expect(stats.packetsSent, 0);

      final updated = stats.copyWith(
        duration: const Duration(seconds: 45),
        spoofDuration: const Duration(seconds: 12),
        packetsSent: 150,
        bytesTransferred: 48000,
        audioLevel: 0.75,
      );

      expect(updated.duration.inSeconds, 45);
      expect(updated.spoofDuration.inSeconds, 12);
      expect(updated.packetsSent, 150);
      expect(updated.bytesTransferred, 48000);
      expect(updated.audioLevel, 0.75);
    });
  });

  group('PcmResampler', () {
    test('generates valid RIFF-WAV header container', () {
      final pcm = Uint8List(3200); // 100ms of 16k 16-bit mono
      final wav = PcmResampler.createWavContainer(
        pcmData: pcm,
        sampleRate: 16000,
        channels: 1,
        bitsPerSample: 16,
      );

      expect(wav.length, 44 + 3200);

      // Check RIFF magic bytes 'RIFF'
      expect(wav[0], 0x52); // R
      expect(wav[1], 0x49); // I
      expect(wav[2], 0x46); // F
      expect(wav[3], 0x46); // F

      // Check WAVE magic bytes 'WAVE'
      expect(wav[8], 0x57); // W
      expect(wav[9], 0x41); // A
      expect(wav[10], 0x56); // V
      expect(wav[11], 0x45); // E
    });

    test('returns exact bytes when already 16kHz mono', () {
      final pcm = Uint8List.fromList([0x01, 0x02, 0x03, 0x04]);
      final out = PcmResampler.processTo16kMono(
        inputBytes: pcm,
        inputSampleRate: 16000,
        inputChannels: 1,
      );
      expect(out, equals(pcm));
    });

    test('resamples 48kHz mono to 16kHz mono by 3:1 ratio', () {
      // 480 samples at 48kHz = 10ms -> should become 160 samples (320 bytes) at 16kHz
      final pcm48k = Uint8List(480 * 2);
      final byteData = ByteData.view(pcm48k.buffer);
      for (var i = 0; i < 480; i++) {
        byteData.setInt16(i * 2, (i % 100) * 100, Endian.little);
      }

      final resampled = PcmResampler.processTo16kMono(
        inputBytes: pcm48k,
        inputSampleRate: 48000,
        inputChannels: 1,
        targetSampleRate: 16000,
      );

      expect(resampled.length, 160 * 2);
    });

    test('calculates RMS level in 0.0 to 1.0 range', () {
      final silence = Uint8List(320);
      expect(PcmResampler.calculateRmsLevel(silence), 0.0);

      final loud = Uint8List(320);
      final b = ByteData.view(loud.buffer);
      for (var i = 0; i < 160; i++) {
        b.setInt16(i * 2, 20000, Endian.little);
      }
      final level = PcmResampler.calculateRmsLevel(loud);
      expect(level, greaterThan(0.0));
      expect(level, lessThanOrEqualTo(1.0));
    });
  });

  group('LiveCallService', () {
    test('initializes in Caller mode with default endpoint', () {
      final service = LiveCallService();
      expect(service.mode, CallMode.caller);
      expect(service.serverHost, 'codequantum.in');
      expect(service.serverPort, 443);
      expect(service.isSpoofActive, isFalse);
      expect(service.spoofSpeaker, 'Teammate 3');
      expect(service.isConnected, isFalse);
      expect(service.webSocketUrl, 'wss://codequantum.in/relay/ws/caller');
    });

    test('builds WebSocket URLs correctly for different hosts, ports, and protocols', () {
      final service = LiveCallService();

      // Direct port 8001
      service.setServerEndpoint('codequantum.in', 8001);
      expect(service.webSocketUrl, 'ws://codequantum.in:8001/ws/caller');

      // Receiver mode on port 8001
      service.setMode(CallMode.receiver);
      expect(service.webSocketUrl, 'ws://codequantum.in:8001/ws/receiver');
      service.setMode(CallMode.caller);

      // Port 443 with default codequantum.in relay path
      service.setServerEndpoint('codequantum.in', 443);
      expect(service.webSocketUrl, 'wss://codequantum.in/relay/ws/caller');

      // Explicit https:// URL with subpath
      service.setServerEndpoint('https://codequantum.in/relay', 8001);
      expect(service.webSocketUrl, 'wss://codequantum.in/relay/ws/caller');

      // Explicit wss:// URL
      service.setServerEndpoint('wss://codequantum.in/relay', 443);
      expect(service.webSocketUrl, 'wss://codequantum.in/relay/ws/caller');

      // Custom LAN host with port
      service.setServerEndpoint('192.168.1.50:8001', 8001);
      expect(service.webSocketUrl, 'ws://192.168.1.50:8001/ws/caller');
    });

    test('switches mode cleanly', () {
      final service = LiveCallService();
      service.setMode(CallMode.receiver);
      expect(service.mode, CallMode.receiver);

      service.setMode(CallMode.caller);
      expect(service.mode, CallMode.caller);
    });

    test('toggleSpoof switches spoof active state and speaker', () {
      final service = LiveCallService();
      service.toggleSpoof();
      expect(service.isSpoofActive, isTrue);

      service.toggleSpoof(speaker: 'Teammate 3');
      expect(service.isSpoofActive, isFalse);
      expect(service.spoofSpeaker, 'Teammate 3');
    });

    test('simulation mode produces active session and assessment', () async {
      final service = LiveCallService();
      service.setSimulationMode(true);
      service.setMode(CallMode.receiver);

      await service.connect();
      expect(service.isConnected, isTrue);
      expect(service.reasoningLogs, isNotEmpty);

      await service.disconnect();
      expect(service.isConnected, isFalse);
    });

    test('playTestSound delegates to playbackService and excites VU stream', () async {
      final service = LiveCallService();
      final levels = <double>[];
      final sub = service.playbackService.playbackLevelStream.listen(levels.add);

      await service.playTestSound(chunkDelay: Duration.zero);

      expect(levels, isNotEmpty);
      expect(levels.any((lvl) => lvl > 0.0), isTrue);

      await sub.cancel();
      await service.dispose();
    });

    test('heartbeatInterval is 15 seconds for NAT and reverse proxy keep-alive', () {
      expect(LiveCallService.heartbeatInterval, const Duration(seconds: 15));
      expect(LiveCallService.reconnectBackoffDelays, [
        const Duration(seconds: 1),
        const Duration(seconds: 2),
        const Duration(seconds: 4),
        const Duration(seconds: 8),
      ]);
    });

    test('unexpected disconnection triggers auto-reconnect with exponential backoff', () async {
      final service = LiveCallService();
      service.setSimulationMode(true);
      await service.connect();
      expect(service.isConnected, isTrue);

      // Trigger unexpected drop while simulation mode is disabled to test reconnect flow
      service.setSimulationMode(false);
      service.triggerUnexpectedDisconnectionForTesting('Connection reset by peer');

      expect(service.isReconnecting, isTrue);
      expect(service.reconnectAttempts, 1);
      expect(service.connectionState, CallConnectionState.reconnecting);
      expect(
        service.reasoningLogs.any((log) => log.text.contains('Reconnecting in 1s')),
        isTrue,
      );

      // Trigger remaining attempts up to max
      service.triggerUnexpectedDisconnectionForTesting('Timeout');
      expect(service.reconnectAttempts, 2);
      service.triggerUnexpectedDisconnectionForTesting('Timeout');
      expect(service.reconnectAttempts, 3);
      service.triggerUnexpectedDisconnectionForTesting('Timeout');
      expect(service.reconnectAttempts, 4);

      // Fifth unexpected drop exceeds max attempts and transitions to error
      service.triggerUnexpectedDisconnectionForTesting('Timeout');
      expect(service.connectionState, CallConnectionState.error);
      expect(
        service.reasoningLogs.any((log) => log.text.contains('Auto-reconnect failed')),
        isTrue,
      );

      await service.dispose();
    });
  });

  group('AudioPlaybackService', () {
    test('ingestPcmChunk applies EMA smoothing to audio level', () async {
      final service = AudioPlaybackService();

      // Create a loud 16-bit PCM chunk (3200 bytes = 100ms)
      final loud = Uint8List(3200);
      final b = ByteData.view(loud.buffer);
      for (var i = 0; i < 1600; i++) {
        b.setInt16(i * 2, 24000, Endian.little);
      }

      service.ingestPcmChunk(loud);
      expect(service.currentLevel, greaterThan(0.0));
      final level1 = service.currentLevel;

      // Second ingest should smoothly track level via EMA
      service.ingestPcmChunk(loud);
      expect(service.currentLevel, greaterThan(0.0));
      expect(service.currentLevel, greaterThanOrEqualTo(level1));

      await service.stop();
      expect(service.currentLevel, 0.0);
      await service.dispose();
    });

    test('toggleMute stops playback and zeroes acoustic level', () async {
      final service = AudioPlaybackService();
      final pcm = Uint8List(3200);
      final b = ByteData.view(pcm.buffer);
      for (var i = 0; i < 1600; i++) {
        b.setInt16(i * 2, 18000, Endian.little);
      }

      service.ingestPcmChunk(pcm);
      expect(service.currentLevel, greaterThan(0.0));

      await service.toggleMute();
      expect(service.isMuted, isTrue);
      expect(service.currentLevel, 0.0);

      await service.dispose();
    });

    test('generateChimePcm creates pleasant 3-note chime with smooth envelopes', () {
      final pcm = AudioPlaybackService.generateChimePcm();
      // 160ms (C5) + 160ms (E5) + 320ms (G5) = 640ms = 10,240 samples = 20,480 bytes
      expect(pcm.length, 20480);

      final b = ByteData.view(pcm.buffer);
      // First and last sample smoothly ramp from/to zero to prevent clicks or pops
      expect(b.getInt16(0, Endian.little).abs(), lessThan(50));
      expect(b.getInt16(pcm.length - 2, Endian.little).abs(), lessThan(50));

      // Peak sample is audible and properly clamped within 16-bit range
      var peak = 0;
      for (var i = 0; i < pcm.length; i += 2) {
        final sample = b.getInt16(i, Endian.little).abs();
        if (sample > peak) peak = sample;
      }
      expect(peak, greaterThan(8000));
      expect(peak, lessThanOrEqualTo(32767));
    });

    test('playTestSound can be called without errors and excites VU meter stream', () async {
      final service = AudioPlaybackService();
      final levels = <double>[];
      final sub = service.playbackLevelStream.listen(levels.add);

      await service.playTestSound(chunkDelay: Duration.zero);

      expect(levels, isNotEmpty);
      expect(levels.any((l) => l > 0.0), isTrue);
      expect(service.currentLevel, greaterThan(0.0));

      await sub.cancel();
      await service.dispose();
    });

    test('playTestSound unmutes if muted before playing', () async {
      final service = AudioPlaybackService();
      await service.toggleMute();
      expect(service.isMuted, isTrue);

      final levels = <double>[];
      final sub = service.playbackLevelStream.listen(levels.add);

      await service.playTestSound(chunkDelay: Duration.zero);

      // Verify unmuted
      expect(service.isMuted, isFalse);
      expect(levels.any((l) => l > 0.0), isTrue);

      await sub.cancel();
      await service.dispose();
    });

    test('playTestSound returns early if disposed', () async {
      final service = AudioPlaybackService();
      await service.dispose();

      // Should complete without error
      await expectLater(
        service.playTestSound(chunkDelay: Duration.zero),
        completes,
      );
    });

    test('playTestSound with default delay runs and excites stream', () async {
      final service = AudioPlaybackService();
      final levels = <double>[];
      final sub = service.playbackLevelStream.listen(levels.add);

      await service.playTestSound();

      expect(levels, isNotEmpty);
      expect(levels.any((l) => l > 0.0), isTrue);

      await sub.cancel();
      await service.dispose();
    });

    test('createStreamingWavHeader generates 44-byte endless WAV header with 0x7FFFFFFF size', () {
      final header = AudioPlaybackService.createStreamingWavHeader(
        sampleRate: 16000,
        channels: 1,
        bitsPerSample: 16,
      );

      expect(header.length, 44);
      // RIFF header
      expect(header[0], 0x52); // 'R'
      expect(header[1], 0x49); // 'I'
      expect(header[2], 0x46); // 'F'
      expect(header[3], 0x46); // 'F'

      final b = ByteData.view(header.buffer);
      // RIFF chunk size 0x7FFFFFFF
      expect(b.getUint32(4, Endian.little), 0x7FFFFFFF);
      // WAVE fmt
      expect(b.getUint16(20, Endian.little), 1); // PCM
      expect(b.getUint16(22, Endian.little), 1); // mono
      expect(b.getUint32(24, Endian.little), 16000); // 16kHz
      expect(b.getUint32(28, Endian.little), 32000); // 32000 bytes/sec
      // data chunk size 0x7FFFFFFF
      expect(b.getUint32(40, Endian.little), 0x7FFFFFFF);
    });

    test('raised-cosine windowing fades audio smoothly on underruns and resumes', () {
      final pcm = Uint8List(320); // 160 samples (10ms)
      final b = ByteData.view(pcm.buffer);
      for (var i = 0; i < 160; i++) {
        b.setInt16(i * 2, 20000, Endian.little);
      }

      // Test fade-in: sample 0 starts at ~0 and rises smoothly to ~20000
      AudioPlaybackService.applyRaisedCosineFadeIn(pcm, fadeSamples: 160);
      expect(b.getInt16(0, Endian.little).abs(), lessThan(50));
      expect(b.getInt16(158 * 2, Endian.little), closeTo(20000, 500));

      // Reset
      for (var i = 0; i < 160; i++) {
        b.setInt16(i * 2, 20000, Endian.little);
      }

      // Test fade-out: sample starts high and ends at ~0
      AudioPlaybackService.applyRaisedCosineFadeOut(pcm, fadeSamples: 160);
      expect(b.getInt16(0, Endian.little), closeTo(20000, 500));
      expect(b.getInt16(159 * 2, Endian.little).abs(), lessThan(50));
    });
  });

  group('AudioCaptureService Hardware Acoustic Processing', () {
    test('enables hardware echo cancellation and noise suppression by default', () {
      final capture = AudioCaptureService();
      expect(capture.echoCancel, isTrue);
      expect(capture.noiseSuppress, isTrue);
      expect(capture.autoGain, isFalse);
    });

    test('allows easy runtime toggling via setAcousticProcessing', () {
      final capture = AudioCaptureService(echoCancel: true, noiseSuppress: true);
      capture.setAcousticProcessing(echoCancel: false, noiseSuppress: false, autoGain: true);
      expect(capture.echoCancel, isFalse);
      expect(capture.noiseSuppress, isFalse);
      expect(capture.autoGain, isTrue);
    });
  });

  group('SpeakerPlaybackCard Receiver UI', () {
    testWidgets('renders Test Speaker button and shows visual feedback on tap', (tester) async {
      final service = LiveCallService();
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SingleChildScrollView(
              child: SpeakerPlaybackCard(
                service: service,
                stats: const CallStats(),
              ),
            ),
          ),
        ),
      );
      await tester.pump();

      // Verify Test Speaker button is rendered in the speaker playback card
      expect(find.text('Test Speaker'), findsOneWidget);
      expect(find.byIcon(Icons.music_note_rounded), findsOneWidget);

      // Tap Test Speaker button
      await tester.tap(find.text('Test Speaker'));
      await tester.pump();

      // Clean up service and unmount
      await service.playbackService.stop();
      await tester.pumpWidget(const SizedBox());
      await service.dispose();
      await tester.pump();
    });
  });

  group('Session Audio Recording (Volatile Judge Proof)', () {
    test('clearSessionRecording purges buffer and notifies listeners', () async {
      final service = LiveCallService();
      expect(service.hasSessionRecording, isFalse);
      expect(service.lastSessionWavBytes, isNull);
      expect(service.lastSessionDuration, isNull);

      final events = <Uint8List?>[];
      final sub = service.sessionRecordingStream.listen(events.add);

      service.clearSessionRecording();
      await Future<void>.delayed(Duration.zero);
      expect(events, contains(null));
      expect(service.hasSessionRecording, isFalse);

      await sub.cancel();
      await service.dispose();
    });

    testWidgets('renders PostCallSessionVerificationCard when recording is present and dismisses on discard', (tester) async {
      tester.view.physicalSize = const Size(1200, 1000);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);

      final service = LiveCallService();

      // Synthesize a valid 16kHz mono WAV in memory (0.5 seconds = 16000 bytes PCM)
      final dummyPcm = Uint8List(16000);
      final wav = PcmResampler.createWavContainer(
        pcmData: dummyPcm,
        sampleRate: 16000,
        channels: 1,
        bitsPerSample: 16,
      );

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SingleChildScrollView(
              child: LiveCallScreen(service: service),
            ),
          ),
        ),
      );
      await tester.pump(const Duration(milliseconds: 50));

      // Initially disconnected with no audio buffer, card is absent
      expect(find.text('Audible Session Replay'), findsNothing);

      // Supply session recording to simulate call completion
      service.setSessionRecordingForTesting(wav, const Duration(seconds: 1));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // Card must be visible with play and discard actions
      expect(find.text('Audible Session Replay'), findsOneWidget);
      expect(find.text('Tap play to listen'), findsOneWidget);
      expect(find.text('Discard'), findsOneWidget);
      expect(find.byIcon(Icons.play_arrow_rounded), findsOneWidget);

      // Tap Discard
      await tester.tap(find.text('Discard'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // Audio recording is purged and card is gone
      expect(service.hasSessionRecording, isFalse);
      expect(find.text('Audible Session Replay'), findsNothing);

      // Clean up
      await tester.pumpWidget(const SizedBox());
      await service.dispose();
      await tester.pump(const Duration(milliseconds: 50));
    });
  });

  group('Live Dual-Model Probability Telemetry', () {
    testWidgets('shows both WavLM Base+ and LFCC-LCNN probabilities live on call screen', (tester) async {
      tester.view.physicalSize = const Size(1200, 1000);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);

      final service = LiveCallService();
      service.setSimulationMode(true);
      service.setMode(CallMode.receiver);

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SingleChildScrollView(
              child: LiveCallScreen(service: service),
            ),
          ),
        ),
      );
      await tester.pump(const Duration(milliseconds: 50));

      // Initially collapsed by default per Apple Progressive Disclosure architecture
      expect(find.text('Forensic Evidence'), findsOneWidget);
      expect(find.text('2 Models Gated'), findsOneWidget);
      expect(find.text('WavLM Base+'), findsNothing);

      // Tap expansion card to disclose ML model telemetry (Level 1 Disclosure)
      await tester.tap(find.text('Forensic Evidence'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 300));

      // Once expanded, both model cards appear on screen with STANDBY
      expect(find.text('WavLM Base+'), findsWidgets);
      expect(find.text('LFCC-LCNN'), findsWidgets);
      expect(find.text('SSL LATENT'), findsWidgets);
      expect(find.text('SPECTRAL'), findsWidgets);
      expect(find.text('STANDBY'), findsWidgets);

      // Transition to connected state with authentic human phonation scores
      service.setConnectionStateForTesting(CallConnectionState.connected);
      service.setAssessmentForTesting(
        LiveRiskAssessment(
          windowIndex: 12,
          riskLevel: LiveRiskLevel.low,
          probability: 0.124,
          latencyMs: 32.0,
          recommendedAction: 'Continue conversation',
          reasoning: 'Both neural acoustic models verify authentic human phonation.',
          timestamp: DateTime.now(),
          expertScores: const {
            'wavlm': LiveExpertScore(
              name: 'wavlm',
              label: 'WavLM Base+',
              probability: 0.124,
              riskLevel: LiveRiskLevel.low,
            ),
            'hybrid': LiveExpertScore(
              name: 'hybrid',
              label: 'LFCC-LCNN',
              probability: 0.089,
              riskLevel: LiveRiskLevel.low,
            ),
          },
        ),
      );
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // Both models now display active inference header, individual probabilities, and human pills
      expect(find.text('DUAL-MODEL INFERENCE ACTIVE'), findsWidgets);
      expect(find.text('WavLM Base+'), findsWidgets);
      expect(find.text('LFCC-LCNN'), findsWidgets);
      expect(find.text('12.4%'), findsWidgets);
      expect(find.text('8.9%'), findsWidgets);
      expect(find.text('spoof'), findsWidgets);
      expect(find.text('HUMAN'), findsWidgets);

      // Now inject synthetic impersonation / clone attack scores
      service.setAssessmentForTesting(
        LiveRiskAssessment(
          windowIndex: 18,
          riskLevel: LiveRiskLevel.high,
          probability: 0.942,
          latencyMs: 35.0,
          recommendedAction: 'TERMINATE CALL IMMEDIATELY',
          reasoning: 'Synthetic pitch trajectory and phase vocoder artifacts detected.',
          timestamp: DateTime.now(),
          expertScores: const {
            'wavlm': LiveExpertScore(
              name: 'wavlm',
              label: 'WavLM Base+',
              probability: 0.942,
              riskLevel: LiveRiskLevel.high,
            ),
            'hybrid': LiveExpertScore(
              name: 'hybrid',
              label: 'LFCC-LCNN',
              probability: 0.915,
              riskLevel: LiveRiskLevel.high,
            ),
          },
        ),
      );
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      // Active header indicates clone detection and displays elevated model probabilities
      expect(find.text('DUAL-EXPERT DETECTED CLONE'), findsWidgets);
      expect(find.text('94.2%'), findsWidgets);
      expect(find.text('91.5%'), findsWidgets);
      expect(find.text('SPOOF'), findsWidgets);

      // Clean up
      await tester.pumpWidget(const SizedBox());
      await service.dispose();
      await tester.pump(const Duration(milliseconds: 50));
    });

    testWidgets('dynamic caller identity and persona rack decouple from hardcoded values', (tester) async {
      tester.view.physicalSize = const Size(1200, 1000);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);

      final service = LiveCallService();
      service.setSimulationMode(true);
      service.setMode(CallMode.receiver);

      // Set custom caller identity
      service.setCallerIdentity('Executive Board Member', 'Audit Committee · Direct SIP Secure');

      expect(service.callerName, equals('Executive Board Member'));
      expect(service.callerSubtitle, equals('Audit Committee · Direct SIP Secure'));
      expect(LiveCallService.availablePersonas.length, greaterThanOrEqualTo(5));

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SingleChildScrollView(
              child: LiveCallScreen(service: service),
            ),
          ),
        ),
      );
      await tester.pump(const Duration(milliseconds: 50));

      // Disconnected / standby defaults
      expect(find.text('Voice Integrity Sentinel'), findsOneWidget);
      expect(find.text('Line Standby · Awaiting Audio Connection'), findsOneWidget);
      expect(find.text('Sarah Jenkins'), findsNothing);

      // Now set connection state to connected
      service.setConnectionStateForTesting(CallConnectionState.connected);
      await tester.pump(const Duration(milliseconds: 50));

      // Connected view displays dynamic caller name and subtitle
      expect(find.text('Executive Board Member'), findsOneWidget);
      expect(find.text('Audit Committee · Direct SIP Secure'), findsOneWidget);
      expect(find.text('Sarah Jenkins'), findsNothing);

      // Verify GlanceableTrustShield with null prob doesn't show fake 98.4%
      service.setAssessmentForTesting(
        LiveRiskAssessment(
          windowIndex: 1,
          riskLevel: LiveRiskLevel.low,
          probability: null, // null probability test
          latencyMs: 30.0,
          recommendedAction: 'OK',
          reasoning: 'Authentic baseline',
          timestamp: DateTime.now(),
        ),
      );
      await tester.pump(const Duration(milliseconds: 50));

      expect(find.text('VERIFIED HUMAN SPEECH · SECURE'), findsOneWidget);
      expect(find.textContaining('98.4'), findsNothing);

      // Verify audioLevelNotifier updates without stats stream thrash
      var statsEmitted = false;
      final statsSub = service.statsStream.listen((_) => statsEmitted = true);
      service.audioLevelNotifier.value = 0.75;
      expect(service.currentAudioLevel, equals(0.75));
      await tester.pump(const Duration(milliseconds: 50));
      expect(statsEmitted, isFalse, reason: 'audioLevelNotifier must not emit on statsStream');
      statsSub.cancel();

      // Clean up
      await tester.pumpWidget(const SizedBox());
      await service.dispose();
      await tester.pump(const Duration(milliseconds: 50));
    });
  });

  group('Apple HIG Progressive Disclosure & Overflow Resilience', () {
    testWidgets('top role selector provides 1-glance compact pill with Attacker and Receiver modes', (tester) async {
      tester.view.physicalSize = const Size(800, 1000);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);

      final service = LiveCallService();
      service.setSimulationMode(true);
      service.setMode(CallMode.receiver);

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SingleChildScrollView(
              child: LiveCallScreen(service: service),
            ),
          ),
        ),
      );
      await tester.pump(const Duration(milliseconds: 50));

      // Unmistakable 1-glance labels
      expect(find.text('⚡ Attacker'), findsOneWidget);
      expect(find.text('🛡️ Receiver'), findsOneWidget);

      // Tap Attacker segment to switch to Caller mode
      await tester.tap(find.text('⚡ Attacker'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 150));

      expect(service.mode, equals(CallMode.caller));
      expect(find.textContaining('TALK AS'), findsOneWidget);

      // Tap Receiver segment to switch back
      await tester.tap(find.text('🛡️ Receiver'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 150));

      expect(service.mode, equals(CallMode.receiver));

      await tester.pumpWidget(const SizedBox());
      await service.dispose();
      await tester.pump(const Duration(milliseconds: 50));
    });

    testWidgets('implements two-level progressive disclosure without first-glance information overload', (tester) async {
      tester.view.physicalSize = const Size(800, 1200);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);

      final service = LiveCallService();
      service.setSimulationMode(true);
      service.setMode(CallMode.receiver);

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SingleChildScrollView(
              child: LiveCallScreen(service: service),
            ),
          ),
        ),
      );
      await tester.pump(const Duration(milliseconds: 50));

      // Level 0 (First Glance): No deep technical telemetry visible
      expect(find.text('WavLM Base+'), findsNothing);
      expect(find.text('LFCC-LCNN'), findsNothing);
      expect(find.text('JITTER QUEUE: ~120 - 300 ms TARGET'), findsNothing);
      expect(find.text('FORENSIC AUDIO NARRATION LOG'), findsNothing);

      // Level 1 Disclosure: Tap 'Forensic Evidence' card
      expect(find.text('Forensic Evidence'), findsOneWidget);
      expect(find.text('2 Models Gated'), findsOneWidget);
      await tester.tap(find.text('Forensic Evidence'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 300));

      // Level 1 reveals dual-model telemetry and Advanced Diagnostics card
      expect(find.text('WavLM Base+'), findsWidgets);
      expect(find.text('LFCC-LCNN'), findsWidgets);
      expect(find.text('Advanced Audio Diagnostics'), findsOneWidget);
      expect(find.text('Deep Trace'), findsOneWidget);

      // But Level 2 internals (Jitter queue & narration log) are still collapsed
      expect(find.text('JITTER QUEUE: ~120 - 300 ms TARGET'), findsNothing);
      expect(find.text('FORENSIC AUDIO NARRATION LOG'), findsNothing);

      // Level 2 Disclosure: Tap 'Advanced Audio Diagnostics' card
      await tester.tap(find.text('Advanced Audio Diagnostics'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 300));

      // Level 2 reveals deep diagnostics
      expect(find.text('JITTER QUEUE: ~120 - 300 ms TARGET'), findsOneWidget);
      expect(find.text('BUFFER STABLE · 0 DROPS'), findsOneWidget);
      expect(find.text('FORENSIC AUDIO NARRATION LOG'), findsOneWidget);
      expect(find.text('Test Speaker'), findsOneWidget);

      await tester.pumpWidget(const SizedBox());
      await service.dispose();
      await tester.pump(const Duration(milliseconds: 50));
    });

    testWidgets('renders with zero RenderFlex overflow on ultra-narrow 320px screen in both modes', (tester) async {
      // 320px width represents narrowest mobile screens (iPhone SE 1st gen)
      tester.view.physicalSize = const Size(320, 900);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);

      final service = LiveCallService();
      service.setSimulationMode(true);
      service.setMode(CallMode.receiver);

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SingleChildScrollView(
              child: LiveCallScreen(service: service),
            ),
          ),
        ),
      );
      await tester.pump(const Duration(milliseconds: 50));

      // Expand Level 1 and Level 2 to test all overflow-prone widgets simultaneously
      await tester.ensureVisible(find.text('Forensic Evidence'));
      await tester.tap(find.text('Forensic Evidence'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 300));

      await tester.ensureVisible(find.text('Advanced Audio Diagnostics'));
      await tester.tap(find.text('Advanced Audio Diagnostics'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 300));

      // Switch to Caller mode to test VU meter and persona selector on 320px
      await tester.ensureVisible(find.text('⚡ Attacker'));
      await tester.tap(find.text('⚡ Attacker'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 150));

      expect(service.mode, equals(CallMode.caller));
      expect(find.text('MIC CAPTURE CHANNEL'), findsOneWidget);

      // No assertion failure or RenderFlex overflow occurred
      expect(tester.takeException(), isNull);

      await tester.pumpWidget(const SizedBox());
      await service.dispose();
      await tester.pump(const Duration(milliseconds: 50));
    });
  });
}
