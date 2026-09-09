import 'package:flutter_test/flutter_test.dart';
import 'package:voice_integrity_flutter/core/models/voice_models.dart';
import 'package:voice_integrity_flutter/core/services/voice_integrity_api.dart';

void main() {
  group('risk contract', () {
    test('maps every supported backend risk state', () {
      expect(riskFromWire('collecting'), RiskState.collecting);
      expect(riskFromWire('low'), RiskState.low);
      expect(riskFromWire('uncertain'), RiskState.uncertain);
      expect(riskFromWire('high'), RiskState.high);
      expect(riskFromWire('silence'), RiskState.unavailable);
    });

    test('showcase report exposes model evidence and a valid timeline', () {
      final report = AnalysisReport.demo();

      expect(report.isDemo, isTrue);
      expect(report.experts, hasLength(2));
      expect(report.windows, isNotEmpty);
      expect(
        report.windows.every((window) => window.probability != null),
        isTrue,
      );
      expect(report.action, isNotEmpty);
    });

    test('health payload preserves the active runtime information', () {
      final health = BackendHealth.fromJson({
        'expert_details': [
          {
            'name': 'hybrid',
            'label': 'Expert 2 · LFCC-LCNN Hybrid',
            'calibrator_version': 'platt-hybrid-v1',
          },
        ],
        'decision_expert': 'hybrid',
        'decision_label': 'LFCC-LCNN Hybrid',
        'fusion_mode': 'single',
        'window_sec': 4.0,
        'hop_sec': 0.5,
        'target_sample_rate': 16000,
      });

      expect(health.decisionExpert, 'hybrid');
      expect(health.experts.single.modelVersion, 'platt-hybrid-v1');
      expect(health.windowSeconds, 4);
      expect(health.sampleRate, 16000);
    });
  });

  group('VoiceIntegrityApi endpoint and URI resolution', () {
    test('defaultEndpoint is hardcoded to Cloud VPS', () {
      expect(VoiceIntegrityApi.defaultEndpoint, 'https://codequantum.in/sih');
    });

    test('constructor defaults to defaultEndpoint when no baseUrl provided', () {
      final defaultApi = VoiceIntegrityApi();
      expect(defaultApi.baseUri.toString(), 'https://codequantum.in/sih');

      final emptyApi = VoiceIntegrityApi('');
      expect(emptyApi.baseUri.toString(), 'https://codequantum.in/sih');

      final whitespaceApi = VoiceIntegrityApi('   ');
      expect(whitespaceApi.baseUri.toString(), 'https://codequantum.in/sih');
    });

    test('normalizes trailing slashes and whitespace in baseUrl', () {
      expect(
        VoiceIntegrityApi.normalise('https://codequantum.in/sih/'),
        'https://codequantum.in/sih',
      );
      expect(
        VoiceIntegrityApi.normalise('https://codequantum.in/sih///'),
        'https://codequantum.in/sih',
      );
      expect(
        VoiceIntegrityApi.normalise('  http://127.0.0.1:8000/  '),
        'http://127.0.0.1:8000',
      );

      final trailingSlashApi = VoiceIntegrityApi('https://codequantum.in/sih///');
      expect(trailingSlashApi.baseUri.toString(), 'https://codequantum.in/sih');
    });

    test('builds correct URIs with subpath for default Cloud VPS endpoint', () {
      final api = VoiceIntegrityApi();

      expect(api.uri('/health').toString(), 'https://codequantum.in/sih/health');
      expect(
        api.uri('/predict-file').toString(),
        'https://codequantum.in/sih/predict-file',
      );
      expect(
        api.buildUri('/health').toString(),
        'https://codequantum.in/sih/health',
      );
      expect(
        api.buildUri('/predict-file').toString(),
        'https://codequantum.in/sih/predict-file',
      );

      // Also handles paths passed without leading slash
      expect(api.uri('health').toString(), 'https://codequantum.in/sih/health');
      expect(
        api.uri('predict-file').toString(),
        'https://codequantum.in/sih/predict-file',
      );
    });

    test(
      'builds correct URIs for localhost and root endpoints with or without trailing slash',
      () {
        final localhostApi = VoiceIntegrityApi('http://127.0.0.1:8000');
        expect(
          localhostApi.uri('/health').toString(),
          'http://127.0.0.1:8000/health',
        );
        expect(
          localhostApi.uri('/predict-file').toString(),
          'http://127.0.0.1:8000/predict-file',
        );

        final localhostSlashApi = VoiceIntegrityApi('http://127.0.0.1:8000/');
        expect(
          localhostSlashApi.uri('/health').toString(),
          'http://127.0.0.1:8000/health',
        );
        expect(
          localhostSlashApi.uri('/predict-file').toString(),
          'http://127.0.0.1:8000/predict-file',
        );

        final cloudWithSlashApi = VoiceIntegrityApi('https://codequantum.in/sih/');
        expect(
          cloudWithSlashApi.uri('/health').toString(),
          'https://codequantum.in/sih/health',
        );
        expect(
          cloudWithSlashApi.uri('/predict-file').toString(),
          'https://codequantum.in/sih/predict-file',
        );
      },
    );
  });
}
