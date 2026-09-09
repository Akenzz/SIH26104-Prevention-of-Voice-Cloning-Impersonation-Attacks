import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:voice_integrity_flutter/core/models/live_call_models.dart';
import 'package:voice_integrity_flutter/core/services/audio_playback_service.dart';
import 'package:voice_integrity_flutter/core/services/live_call_service.dart';
import 'package:voice_integrity_flutter/core/services/pcm_resampler.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUpAll(() {
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
      expect(service.serverHost, '127.0.0.1');
      expect(service.serverPort, 8001);
      expect(service.isSpoofActive, isFalse);
      expect(service.spoofSpeaker, 'Teammate 3');
      expect(service.isConnected, isFalse);
      expect(service.webSocketUrl, 'ws://127.0.0.1:8001/ws/caller');
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
  });
}
