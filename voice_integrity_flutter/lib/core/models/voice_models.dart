import 'dart:math' as math;

enum RiskState { collecting, low, uncertain, high, unavailable }

extension RiskStateX on RiskState {
  String get wireName => switch (this) {
    RiskState.collecting => 'collecting',
    RiskState.low => 'low',
    RiskState.uncertain => 'uncertain',
    RiskState.high => 'high',
    RiskState.unavailable => 'unavailable',
  };

  String get label => switch (this) {
    RiskState.collecting => 'Collecting signal',
    RiskState.low => 'Low synthetic risk',
    RiskState.uncertain => 'Review needed',
    RiskState.high => 'High synthetic risk',
    RiskState.unavailable => 'Signal unavailable',
  };

  String get shortLabel => switch (this) {
    RiskState.collecting => 'Collecting',
    RiskState.low => 'Low',
    RiskState.uncertain => 'Review',
    RiskState.high => 'High',
    RiskState.unavailable => 'Unavailable',
  };

  String get action => switch (this) {
    RiskState.collecting =>
      'Keep listening. A second clean audio window is needed before a decision.',
    RiskState.low =>
      'Continue the interaction; still use out-of-band verification for sensitive actions.',
    RiskState.uncertain =>
      'Pause the request and verify through a known callback or MFA.',
    RiskState.high =>
      'Do not rely on the call alone. Verify through a known callback number.',
    RiskState.unavailable =>
      'Do not treat missing, clipped, silent, or dropped audio as a safe result.',
  };
}

RiskState riskFromWire(Object? value) => switch ('$value'.toLowerCase()) {
  'collecting' => RiskState.collecting,
  'low' => RiskState.low,
  'uncertain' => RiskState.uncertain,
  'high' => RiskState.high,
  _ => RiskState.unavailable,
};

double? asDouble(Object? value) => value is num ? value.toDouble() : null;

class ExpertScore {
  const ExpertScore({
    required this.id,
    required this.label,
    required this.probability,
    required this.riskState,
    this.rawLogit,
    this.modelVersion,
  });

  final String id;
  final String label;
  final double? probability;
  final RiskState riskState;
  final double? rawLogit;
  final String? modelVersion;
}

class WindowScore {
  const WindowScore({
    required this.index,
    required this.timeSeconds,
    required this.probability,
    required this.riskState,
    required this.expertProbabilities,
  });

  final int index;
  final double timeSeconds;
  final double? probability;
  final RiskState riskState;
  final Map<String, double> expertProbabilities;
}

class AnalysisReport {
  const AnalysisReport({
    required this.sourceLabel,
    required this.riskState,
    required this.probability,
    required this.experts,
    required this.windows,
    required this.decisionExpert,
    required this.agreement,
    required this.confidence,
    required this.suspiciousWindows,
    required this.totalWindows,
    required this.peakTimeSeconds,
    this.recommendedAction,
    this.isDemo = false,
  });

  final String sourceLabel;
  final RiskState riskState;
  final double? probability;
  final List<ExpertScore> experts;
  final List<WindowScore> windows;
  final String decisionExpert;
  final String agreement;
  final String confidence;
  final int suspiciousWindows;
  final int totalWindows;
  final double? peakTimeSeconds;
  final String? recommendedAction;
  final bool isDemo;

  String get action => recommendedAction ?? riskState.action;

  factory AnalysisReport.demo({double phase = 0}) {
    final values = List<double>.generate(18, (index) {
      final wave = math.sin((index + phase) * .56) * .12;
      return (0.48 + wave + (index > 10 ? .19 : 0))
          .clamp(0.08, 0.94)
          .toDouble();
    });
    final windows = List<WindowScore>.generate(values.length, (index) {
      final overall = values[index];
      return WindowScore(
        index: index + 1,
        timeSeconds: index * .5,
        probability: overall,
        riskState: overall >= .65 ? RiskState.high : RiskState.uncertain,
        expertProbabilities: {
          'wavlm': (overall - .12 + math.sin(index) * .04)
              .clamp(0.02, .98)
              .toDouble(),
          'hybrid': (overall + .05).clamp(0.02, .98).toDouble(),
          'ssl': (overall + .11 + math.cos(index) * .03)
              .clamp(0.02, .98)
              .toDouble(),
        },
      );
    });
    final probability = values.last;
    return AnalysisReport(
      sourceLabel: 'Judge walkthrough · simulated voice call',
      riskState: probability >= .65 ? RiskState.high : RiskState.uncertain,
      probability: probability,
      experts: [
        ExpertScore(
          id: 'wavlm',
          label: 'Expert 1 · WavLM Base+',
          probability: .54,
          riskState: RiskState.uncertain,
          rawLogit: .72,
          modelVersion: 'wavlm-base-plus-ep6',
        ),
        ExpertScore(
          id: 'hybrid',
          label: 'Expert 2 · LFCC-LCNN Hybrid',
          probability: .76,
          riskState: RiskState.high,
          rawLogit: 3.08,
          modelVersion: 'hybrid_clean_plus_newclips_final',
        ),
        ExpertScore(
          id: 'ssl',
          label: 'Expert 3 · TakHemlata SSL',
          probability: .81,
          riskState: RiskState.high,
          rawLogit: 2.61,
          modelVersion: 'best_SSL_model_LA',
        ),
      ],
      windows: windows,
      decisionExpert: 'Weighted calibrated ensemble',
      agreement: 'Two of three experts indicate elevated risk',
      confidence: 'Medium',
      suspiciousWindows: windows
          .where((window) => (window.probability ?? 0) >= .65)
          .length,
      totalWindows: windows.length,
      peakTimeSeconds: 8.5,
      recommendedAction:
          'Pause the high-value request and use a known callback or MFA.',
      isDemo: true,
    );
  }
}

class BackendHealth {
  const BackendHealth({
    required this.experts,
    required this.decisionExpert,
    required this.decisionLabel,
    required this.fusionMode,
    required this.windowSeconds,
    required this.hopSeconds,
    required this.sampleRate,
  });

  final List<ExpertScore> experts;
  final String decisionExpert;
  final String decisionLabel;
  final String fusionMode;
  final double windowSeconds;
  final double hopSeconds;
  final int sampleRate;

  factory BackendHealth.fromJson(Map<String, dynamic> json) {
    final rawExperts = (json['expert_details'] as List? ?? const []);
    return BackendHealth(
      experts: rawExperts.whereType<Map>().map((raw) {
        final entry = Map<String, dynamic>.from(raw);
        return ExpertScore(
          id: '${entry['name'] ?? ''}',
          label: '${entry['label'] ?? entry['name'] ?? 'Unknown model'}',
          probability: null,
          riskState: RiskState.collecting,
          modelVersion: entry['calibrator_version']?.toString(),
        );
      }).toList(),
      decisionExpert: '${json['decision_expert'] ?? 'not reported'}',
      decisionLabel:
          '${json['decision_label'] ?? json['fusion_mode'] ?? 'not reported'}',
      fusionMode: '${json['fusion_mode'] ?? 'not reported'}',
      windowSeconds: asDouble(json['window_sec']) ?? 4,
      hopSeconds: asDouble(json['hop_sec']) ?? .5,
      sampleRate: (json['target_sample_rate'] as num?)?.toInt() ?? 16000,
    );
  }
}
