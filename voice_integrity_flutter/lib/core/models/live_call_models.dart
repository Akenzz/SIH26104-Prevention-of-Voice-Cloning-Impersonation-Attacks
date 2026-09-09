import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

/// Active mode for the Live Demo Call feature.
enum CallMode {
  caller('Caller Mode', 'Attacker (Spoof Voice)', Icons.call_made_rounded),
  receiver(
    'Receiver Mode',
    'Victim (Integrity Shield)',
    Icons.call_received_rounded,
  );

  const CallMode(this.title, this.subtitle, this.icon);
  final String title;
  final String subtitle;
  final IconData icon;
}

/// Connection lifecycle for the Live Call WebSocket.
enum CallConnectionState {
  idle('Disconnected', Icons.cloud_off_rounded),
  connecting('Connecting...', Icons.cloud_sync_rounded),
  connected('Connected', Icons.cloud_done_rounded),
  reconnecting('Reconnecting...', Icons.sync_problem_rounded),
  disconnected('Disconnected', Icons.cloud_off_rounded),
  error('Connection Error', Icons.error_outline_rounded);

  const CallConnectionState(this.label, this.icon);
  final String label;
  final IconData icon;

  bool get isConnected => this == CallConnectionState.connected;
  bool get isConnecting =>
      this == CallConnectionState.connecting ||
      this == CallConnectionState.reconnecting;
}

/// Real-time risk bands corresponding to model inference policy.
enum LiveRiskLevel {
  low(
    'Low Risk',
    'Likely Human Speech',
    AppColors.moss,
    AppColors.mossSoft,
    Icons.verified_user_rounded,
  ),
  uncertain(
    'Uncertain',
    'Potential Anomaly / Verify Identity',
    AppColors.amber,
    AppColors.amberSoft,
    Icons.warning_amber_rounded,
  ),
  high(
    'High Risk',
    'AI Voice Clone Impersonation Detected',
    AppColors.vermilion,
    AppColors.vermilionSoft,
    Icons.gpp_bad_rounded,
  ),
  unavailable(
    'Unavailable',
    'Insufficient Speech / Quality Gate',
    AppColors.mutedInk,
    AppColors.canvas,
    Icons.mic_off_rounded,
  ),
  transitioning(
    'Analyzing Buffer',
    'Switching Voice Stream — Analyzing Buffer...',
    AppColors.amber,
    AppColors.amberSoft,
    Icons.hourglass_top_rounded,
  );

  const LiveRiskLevel(
    this.label,
    this.description,
    this.accentColor,
    this.backgroundColor,
    this.icon,
  );

  final String label;
  final String description;
  final Color accentColor;
  final Color backgroundColor;
  final IconData icon;

  static LiveRiskLevel fromBackendString(String? state) {
    return switch (state?.toLowerCase().trim()) {
      'low' => LiveRiskLevel.low,
      'uncertain' => LiveRiskLevel.uncertain,
      'high' => LiveRiskLevel.high,
      'collecting' => LiveRiskLevel.unavailable,
      'unavailable' => LiveRiskLevel.unavailable,
      'transitioning' => LiveRiskLevel.transitioning,
      _ => LiveRiskLevel.unavailable,
    };
  }
}

/// Expert sub-score detail in real-time scoring.
class LiveExpertScore {
  const LiveExpertScore({
    required this.name,
    required this.label,
    required this.probability,
    required this.riskLevel,
    this.rawLogit,
    this.calibratorVersion,
  });

  final String name;
  final String label;
  final double? probability;
  final LiveRiskLevel riskLevel;
  final double? rawLogit;
  final String? calibratorVersion;

  factory LiveExpertScore.fromJson(String name, Map<String, dynamic> json) {
    final prob = (json['probability'] as num?)?.toDouble();
    return LiveExpertScore(
      name: name,
      label: (json['label'] as String?) ?? name,
      probability: prob,
      riskLevel: LiveRiskLevel.fromBackendString(json['risk_state'] as String?),
      rawLogit: (json['logit'] as num?)?.toDouble(),
      calibratorVersion: json['calibrator_version'] as String?,
    );
  }
}

/// Instantaneous risk assessment snapshot.
class LiveRiskAssessment {
  const LiveRiskAssessment({
    required this.riskLevel,
    required this.probability,
    required this.recommendedAction,
    required this.reasoning,
    required this.windowIndex,
    required this.latencyMs,
    required this.timestamp,
    this.fusedLogit,
    this.expertScores = const {},
    this.isSpoofedByCaller = false,
  });

  final LiveRiskLevel riskLevel;
  final double? probability;
  final String recommendedAction;
  final String reasoning;
  final int windowIndex;
  final double latencyMs;
  final DateTime timestamp;
  final double? fusedLogit;
  final Map<String, LiveExpertScore> expertScores;
  final bool isSpoofedByCaller;

  factory LiveRiskAssessment.initial() {
    return LiveRiskAssessment(
      riskLevel: LiveRiskLevel.unavailable,
      probability: null,
      recommendedAction: 'Awaiting audio stream from caller...',
      reasoning: 'Buffer is waiting for incoming 16kHz speech frames.',
      windowIndex: 0,
      latencyMs: 0.0,
      timestamp: DateTime.now(),
    );
  }

  factory LiveRiskAssessment.transitioning({required bool isSpoofTarget}) {
    return LiveRiskAssessment(
      riskLevel: LiveRiskLevel.transitioning,
      probability: isSpoofTarget ? 0.50 : 0.25,
      recommendedAction:
          'Analyzing incoming audio frames in real-time buffer...',
      reasoning: 'Switching Voice Stream — Analyzing Buffer...',
      windowIndex: 0,
      latencyMs: 18.0,
      timestamp: DateTime.now(),
      isSpoofedByCaller: isSpoofTarget,
    );
  }

  factory LiveRiskAssessment.fromJson(Map<String, dynamic> json) {
    final prob =
        (json['smoothed_probability'] as num? ??
                json['probability'] as num? ??
                json['weighted_probability'] as num?)
            ?.toDouble();

    final expertsMap = <String, LiveExpertScore>{};
    final rawScores = json['scores'];
    if (rawScores is Map<String, dynamic>) {
      rawScores.forEach((key, val) {
        if (val is Map<String, dynamic>) {
          expertsMap[key] = LiveExpertScore.fromJson(key, val);
        }
      });
    }

    return LiveRiskAssessment(
      riskLevel: LiveRiskLevel.fromBackendString(json['risk_state'] as String?),
      probability: prob,
      recommendedAction:
          (json['recommended_action'] as String?) ??
          (prob != null && prob >= 0.65
              ? 'Pause the call and verify identity via verified callback.'
              : 'Proceed with normal verification controls.'),
      reasoning:
          (json['reasoning'] as String?) ??
          (prob != null && prob >= 0.65
              ? 'Multi-expert synthetic speech artifacts detected above policy threshold.'
              : 'Acoustic envelope matches human speech baseline.'),
      windowIndex: (json['window_index'] as num?)?.toInt() ?? 0,
      latencyMs: (json['latency_ms'] as num?)?.toDouble() ?? 38.0,
      timestamp: DateTime.now(),
      fusedLogit: (json['fused_logit'] as num?)?.toDouble(),
      expertScores: expertsMap,
      isSpoofedByCaller: (json['is_spoofed'] as bool?) ?? false,
    );
  }
}

/// Chronological reasoning trace item for audit log.
class ReasoningLogEntry {
  const ReasoningLogEntry({
    required this.id,
    required this.timestamp,
    required this.text,
    required this.riskLevel,
    this.probability,
    this.windowIndex,
    this.latencyMs,
  });

  final String id;
  final DateTime timestamp;
  final String text;
  final LiveRiskLevel riskLevel;
  final double? probability;
  final int? windowIndex;
  final double? latencyMs;
}

/// Call session statistics.
class CallStats {
  const CallStats({
    this.duration = Duration.zero,
    this.spoofDuration = Duration.zero,
    this.packetsSent = 0,
    this.packetsReceived = 0,
    this.bytesTransferred = 0,
    this.roundTripLatencyMs = 0,
    this.audioLevel = 0.0,
  });

  final Duration duration;
  final Duration spoofDuration;
  final int packetsSent;
  final int packetsReceived;
  final int bytesTransferred;
  final int roundTripLatencyMs;
  final double audioLevel;

  CallStats copyWith({
    Duration? duration,
    Duration? spoofDuration,
    int? packetsSent,
    int? packetsReceived,
    int? bytesTransferred,
    int? roundTripLatencyMs,
    double? audioLevel,
  }) {
    return CallStats(
      duration: duration ?? this.duration,
      spoofDuration: spoofDuration ?? this.spoofDuration,
      packetsSent: packetsSent ?? this.packetsSent,
      packetsReceived: packetsReceived ?? this.packetsReceived,
      bytesTransferred: bytesTransferred ?? this.bytesTransferred,
      roundTripLatencyMs: roundTripLatencyMs ?? this.roundTripLatencyMs,
      audioLevel: audioLevel ?? this.audioLevel,
    );
  }
}
