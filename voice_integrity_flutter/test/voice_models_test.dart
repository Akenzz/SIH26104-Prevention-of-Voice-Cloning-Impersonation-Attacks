import 'package:flutter_test/flutter_test.dart';
import 'package:voice_integrity_flutter/core/models/voice_models.dart';

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
      expect(report.experts, hasLength(3));
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
}
