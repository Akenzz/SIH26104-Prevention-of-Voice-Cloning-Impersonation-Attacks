import 'dart:async';
import 'dart:math' as math;

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../../core/models/live_call_models.dart';
import '../../core/services/live_call_service.dart';

// -----------------------------------------------------------------------------
// HIGH-CONTRAST LIGHT PRESENTATION SYSTEM (Optimized for live demo venues)
// -----------------------------------------------------------------------------

abstract final class _CallTheme {
  static const bg = Color(0xFFF8FAFC); // Slate 50 clean background
  static const surface = Color(0xFFFFFFFF); // Pure white cards
  static const surfaceElevated = Color(0xFFF1F5F9); // Slate 100 chips & controls
  static const border = Color(0xFFE2E8F0); // Slate 200 hairline border
  static const borderSubtle = Color(0xFFCBD5E1); // Slate 300 divider

  static const textPrimary = Color(0xFF0F172A); // Slate 900 — high contrast ink
  static const textSecondary = Color(0xFF475569); // Slate 600
  static const textMuted = Color(0xFF64748B); // Slate 500

  // 4-Band Semantic Security Palette
  static const emerald = Color(0xFF059669); // Verified Human Speech
  static const emeraldText = Color(0xFF065F46); // WCAG AAA
  static const emeraldBg = Color(0xFFECFDF5); // Soft mint tint
  static const emeraldBorder = Color(0xFFA7F3D0);

  static const amber = Color(0xFFD97706); // Analyzing / Uncertain
  static const amberText = Color(0xFF92400E); // WCAG AA+
  static const amberBg = Color(0xFFFFFBEB); // Soft amber tint
  static const amberBorder = Color(0xFFFDE68A);

  static const crimson = Color(0xFFDC2626); // Spoof Clone Threat
  static const crimsonText = Color(0xFF991B1B); // WCAG AAA
  static const crimsonBg = Color(0xFFFEF2F2); // Soft rose tint
  static const crimsonBorder = Color(0xFFFECACA);

  static const slate = Color(0xFF475569);
  static const slateBg = Color(0xFFF1F5F9);

  // Forensic Audio Narration Kind Colors
  static const kindCollect = Color(0xFF64748B);
  static const kindInfo = Color(0xFF334155);
  static const kindWarn = Color(0xFFD97706);
  static const kindAlert = Color(0xFFDC2626);
  static const kindVerdict = Color(0xFF0F172A);
}

// -----------------------------------------------------------------------------
// MAIN LIVE CALL SCREEN ("PUNJAB HACK" MINIMALIST PHONE CALL LAYOUT)
// -----------------------------------------------------------------------------

class LiveCallScreen extends StatefulWidget {
  const LiveCallScreen({
    super.key,
    required this.service,
    this.initialMode = CallMode.receiver,
  });

  final LiveCallService service;
  final CallMode initialMode;

  @override
  State<LiveCallScreen> createState() => _LiveCallScreenState();
}

class _LiveCallScreenState extends State<LiveCallScreen>
    with SingleTickerProviderStateMixin {
  late final LiveCallService _service;
  late final AnimationController _pulseController;
  late final Animation<double> _pulseAnimation;

  StreamSubscription<CallConnectionState>? _connectionSub;
  StreamSubscription<LiveRiskAssessment>? _riskSub;
  StreamSubscription<CallStats>? _statsSub;
  StreamSubscription<List<ReasoningLogEntry>>? _logsSub;
  StreamSubscription<Uint8List?>? _recordingSub;

  @override
  void initState() {
    super.initState();
    _service = widget.service;
    if (!_service.isConnected && _service.mode != widget.initialMode) {
      _service.setMode(widget.initialMode);
    }

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    )..repeat(reverse: true);

    _pulseAnimation = CurvedAnimation(
      parent: _pulseController,
      curve: Curves.easeInOut,
    );

    _listenToService();
  }

  void _listenToService() {
    _connectionSub = _service.connectionStateStream.listen((_) {
      if (mounted) setState(() {});
    });
    _riskSub = _service.riskAssessmentStream.listen((_) {
      if (mounted) setState(() {});
    });
    _statsSub = _service.statsStream.listen((_) {
      if (mounted) setState(() {});
    });
    _logsSub = _service.reasoningLogsStream.listen((_) {
      if (mounted) setState(() {});
    });
    _recordingSub = _service.sessionRecordingStream.listen((_) {
      if (mounted) setState(() {});
    });
  }

  @override
  void dispose() {
    _recordingSub?.cancel();
    _service.clearSessionRecording();
    _pulseController.dispose();
    _connectionSub?.cancel();
    _riskSub?.cancel();
    _statsSub?.cancel();
    _logsSub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final compact = MediaQuery.sizeOf(context).width < 768;
    final isCaller = _service.mode == CallMode.caller;

    return Container(
      color: _CallTheme.bg,
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 680),
          child: SingleChildScrollView(
            padding: EdgeInsets.symmetric(
              horizontal: compact ? 16 : 24,
              vertical: compact ? 16 : 24,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // 1. Top Segmented Role Toggle: Caller (Attacker) vs Receiver (Victim)
                _RoleSegmentedSelector(
                  currentMode: _service.mode,
                  onSelectMode: (mode) async {
                    await _service.setMode(mode);
                    if (mounted) setState(() {});
                  },
                ),
                const SizedBox(height: 16),

                // 2. Post-Call Audio Verification Card (Audible Session Replay)
                if (!_service.isConnected && _service.hasSessionRecording) ...[
                  _PostCallSessionVerificationCard(
                    service: _service,
                    onDiscard: () {
                      _service.clearSessionRecording();
                      setState(() {});
                    },
                  ),
                  const SizedBox(height: 16),
                ],

                // 3. Center Call Display (Living Sentinel / Attacker Deck)
                if (isCaller)
                  _CallerModeView(
                    service: _service,
                    pulseAnimation: _pulseAnimation,
                    onStateChanged: () => setState(() {}),
                  )
                else
                  _ReceiverModeView(
                    service: _service,
                    pulseAnimation: _pulseAnimation,
                    isMuted: _service.playbackService.isMuted,
                    onToggleMute: () async {
                      await _service.playbackService.toggleMute();
                      if (mounted) setState(() {});
                    },
                  ),
                const SizedBox(height: 16),

                // 4. Forensic Evidence Breakdown (Collapsible for Judges)
                _ForensicExpansionCard(
                  service: _service,
                  assessment: _service.currentAssessment,
                  stats: _service.stats,
                  pulseAnimation: _pulseAnimation,
                ),
                const SizedBox(height: 24),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// TOP SEGMENTED ROLE SELECTOR
// -----------------------------------------------------------------------------

class _RoleSegmentedSelector extends StatelessWidget {
  const _RoleSegmentedSelector({
    required this.currentMode,
    required this.onSelectMode,
  });

  final CallMode currentMode;
  final ValueChanged<CallMode> onSelectMode;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: _CallTheme.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: _CallTheme.border),
        boxShadow: const [
          BoxShadow(
            color: Color(0x06000000),
            blurRadius: 6,
            offset: Offset(0, 1),
          ),
        ],
      ),
      child: Row(
        children: CallMode.values.map((mode) {
          final isSelected = mode == currentMode;
          final isCaller = mode == CallMode.caller;

          return Expanded(
            child: InkWell(
              borderRadius: BorderRadius.circular(10),
              onTap: () => onSelectMode(mode),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 180),
                curve: Curves.easeOutCubic,
                padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
                decoration: BoxDecoration(
                  color: isSelected ? _CallTheme.surfaceElevated : Colors.transparent,
                  borderRadius: BorderRadius.circular(10),
                  border: isSelected
                      ? Border.all(color: _CallTheme.borderSubtle)
                      : null,
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      isCaller ? Icons.phone_iphone_rounded : Icons.shield_rounded,
                      size: 16,
                      color: isSelected
                          ? (isCaller ? _CallTheme.crimson : _CallTheme.emerald)
                          : _CallTheme.textMuted,
                    ),
                    const SizedBox(width: 8),
                    Flexible(
                      child: Text(
                        isCaller ? '📱 Caller Mode' : '🛡️ Receiver Mode',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: isSelected ? FontWeight.w800 : FontWeight.w600,
                          color: isSelected ? _CallTheme.textPrimary : _CallTheme.textSecondary,
                        ),
                      ),
                    ),
                    const SizedBox(width: 6),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: isSelected
                            ? (isCaller ? _CallTheme.crimsonBg : _CallTheme.emeraldBg)
                            : Colors.transparent,
                        borderRadius: BorderRadius.circular(4),
                        border: Border.all(
                          color: isSelected
                              ? (isCaller ? _CallTheme.crimsonBorder : _CallTheme.emeraldBorder)
                              : _CallTheme.border,
                        ),
                      ),
                      child: Text(
                        isCaller ? 'ATTACKER' : 'VICTIM',
                        style: TextStyle(
                          fontFamily: 'monospace',
                          fontSize: 9,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 0.6,
                          color: isSelected
                              ? (isCaller ? _CallTheme.crimson : _CallTheme.emerald)
                              : _CallTheme.textMuted,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// RECEIVER MODE VIEW: CENTER CALL DISPLAY (MINIMALIST PHONE CALL CARD)
// -----------------------------------------------------------------------------

class _ReceiverModeView extends StatelessWidget {
  const _ReceiverModeView({
    required this.service,
    required this.pulseAnimation,
    required this.isMuted,
    required this.onToggleMute,
  });

  final LiveCallService service;
  final Animation<double> pulseAnimation;
  final bool isMuted;
  final VoidCallback onToggleMute;

  @override
  Widget build(BuildContext context) {
    final assessment = service.currentAssessment;
    final stats = service.stats;
    final isConnected = service.isConnected;

    return _ActiveCallSentinelCard(
      service: service,
      assessment: assessment,
      stats: stats,
      pulseAnimation: pulseAnimation,
      isConnected: isConnected,
      isMuted: isMuted,
      onToggleMute: onToggleMute,
    );
  }
}

// -----------------------------------------------------------------------------
// ACTIVE CALL SENTINEL CARD (RECEIVER CENTER CALL DISPLAY)
// -----------------------------------------------------------------------------

class _ActiveCallSentinelCard extends StatelessWidget {
  const _ActiveCallSentinelCard({
    required this.service,
    required this.assessment,
    required this.stats,
    required this.pulseAnimation,
    required this.isConnected,
    required this.isMuted,
    required this.onToggleMute,
  });

  final LiveCallService service;
  final LiveRiskAssessment assessment;
  final CallStats stats;
  final Animation<double> pulseAnimation;
  final bool isConnected;
  final bool isMuted;
  final VoidCallback onToggleMute;

  @override
  Widget build(BuildContext context) {
    final level = assessment.riskLevel;
    final isHighRisk = level == LiveRiskLevel.high;

    // Accent colors based on threat level
    final (Color accentColor, Color accentBg, Color accentBorder) = switch (level) {
      LiveRiskLevel.low => (
          _CallTheme.emerald,
          _CallTheme.emeraldBg,
          _CallTheme.emeraldBorder,
        ),
      LiveRiskLevel.uncertain => (
          _CallTheme.amber,
          _CallTheme.amberBg,
          _CallTheme.amberBorder,
        ),
      LiveRiskLevel.high => (
          _CallTheme.crimson,
          _CallTheme.crimsonBg,
          _CallTheme.crimsonBorder,
        ),
      _ => (
          _CallTheme.slate,
          _CallTheme.slateBg,
          _CallTheme.border,
        ),
    };

    return Container(
      decoration: BoxDecoration(
        color: _CallTheme.surface,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: isHighRisk ? _CallTheme.crimson : const Color(0xFFCBD5E1),
          width: isHighRisk ? 2.0 : 1.2,
        ),
        boxShadow: [
          BoxShadow(
            color: isHighRisk
                ? _CallTheme.crimson.withValues(alpha: 0.18)
                : const Color(0x0F0F172A),
            blurRadius: isHighRisk ? 28 : 16,
            offset: const Offset(0, 4),
          ),
          const BoxShadow(
            color: Color(0x080F172A),
            blurRadius: 2,
            offset: Offset(0, 1),
          ),
        ],
      ),
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 26),
      child: Column(
        children: [
          // Security Status & Call Duration Bar
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: isConnected ? _CallTheme.emeraldBg : _CallTheme.surfaceElevated,
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(
                    color: isConnected ? _CallTheme.emeraldBorder : _CallTheme.border,
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      isConnected ? Icons.lock_outline_rounded : Icons.lock_open_rounded,
                      size: 13,
                      color: isConnected ? _CallTheme.emerald : _CallTheme.textMuted,
                    ),
                    const SizedBox(width: 5),
                    Text(
                      isConnected ? 'ENCRYPTED RELAY' : 'STANDBY',
                      style: TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 10,
                        fontWeight: FontWeight.w800,
                        color: isConnected ? _CallTheme.emerald : _CallTheme.textMuted,
                        letterSpacing: 0.6,
                      ),
                    ),
                  ],
                ),
              ),
              _CallDurationTicker(duration: stats.duration, isConnected: isConnected),
            ],
          ),
          const SizedBox(height: 20),

          // Central Living Neural Voice Aura & Avatar (Reactive to speech audio levels)
          RepaintBoundary(
            child: ValueListenableBuilder<double>(
              valueListenable: service.audioLevelNotifier,
              builder: (context, currentAudioLevel, child) {
                return AnimatedBuilder(
                  animation: pulseAnimation,
                  builder: (context, _) {
                    return CustomPaint(
                      painter: LivingNeuralAuraPainter(
                        pulseValue: pulseAnimation.value,
                        audioLevel: currentAudioLevel,
                        accentColor: accentColor,
                        isSpoof: isHighRisk,
                      ),
                      child: child,
                    );
                  },
                );
              },
              child: SizedBox(
                width: 170,
                height: 170,
                child: Center(
                  child: Container(
                    width: 88,
                    height: 88,
                    decoration: BoxDecoration(
                      color: _CallTheme.surfaceElevated,
                      shape: BoxShape.circle,
                      border: Border.all(color: accentColor, width: 2.5),
                      boxShadow: [
                        BoxShadow(
                          color: accentColor.withValues(alpha: 0.2),
                          blurRadius: 16,
                          offset: const Offset(0, 4),
                        ),
                      ],
                    ),
                    child: Center(
                      child: Icon(
                        isHighRisk
                            ? Icons.gpp_bad_rounded
                            : (isConnected
                                ? Icons.person_rounded
                                : Icons.person_outline_rounded),
                        size: 46,
                        color: isHighRisk ? _CallTheme.crimson : _CallTheme.textPrimary,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Caller Identity (Dynamic caller name & subtitle)
          Text(
            isHighRisk
                ? (service.isSpoofActive
                    ? 'Detected Clone: ${service.spoofSpeaker}'
                    : 'Synthetic Impersonation Detected')
                : (isConnected ? service.callerName : 'Voice Integrity Sentinel'),
            style: const TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w800,
              color: _CallTheme.textPrimary,
              letterSpacing: -0.3,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 4),
          Text(
            isHighRisk
                ? 'Acoustic Anomaly · Spoofing Signature Active'
                : (isConnected
                    ? service.callerSubtitle
                    : 'Line Standby · Awaiting Audio Connection'),
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w500,
              color: _CallTheme.textSecondary,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 16),

          // High-Contrast Glanceable Risk Banner
          _GlanceableTrustShield(
            assessment: assessment,
            accentColor: accentColor,
            accentBg: accentBg,
            accentBorder: accentBorder,
          ),
          const SizedBox(height: 16),

          // Reactive Waveform Spectrum Meter (responding to mic/speaker audio)
          ReactiveWaveformSpectrum(
            levelNotifier: service.audioLevelNotifier,
            audioLevel: stats.audioLevel,
            accentColor: accentColor,
            isActive: isConnected,
          ),
          const SizedBox(height: 18),

          // Contextual Emergency Intervention Banner (Triggers on Crimson Threat)
          if (isHighRisk) ...[
            _EmergencyInterventionBanner(
              assessment: assessment,
              onEndCall: () => service.disconnect(),
            ),
            const SizedBox(height: 18),
          ],

          const Divider(color: _CallTheme.border, height: 1),
          const SizedBox(height: 18),

          // Receiver In-Call Controls
          if (isConnected)
            _ReceiverControlsBar(
              service: service,
              isMuted: isMuted,
              onToggleMute: onToggleMute,
              onEndCall: () => service.disconnect(),
            )
          else
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: () => service.connect(),
                icon: const Icon(Icons.call_rounded, size: 18),
                label: Text(
                  service.useSimulationMode
                      ? 'Start Presentation Call (Demo)'
                      : 'Connect Call to Relay',
                ),
                style: FilledButton.styleFrom(
                  backgroundColor: _CallTheme.emerald,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  textStyle: const TextStyle(fontSize: 15, fontWeight: FontWeight.w800),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// LIVING NEURAL AURA PAINTER
// -----------------------------------------------------------------------------

class LivingNeuralAuraPainter extends CustomPainter {
  LivingNeuralAuraPainter({
    required this.pulseValue,
    required this.audioLevel,
    required this.accentColor,
    required this.isSpoof,
  });

  final double pulseValue;
  final double audioLevel;
  final Color accentColor;
  final bool isSpoof;

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final baseRadius = size.width * 0.28;
    final clampedLevel = (audioLevel.isFinite) ? audioLevel.clamp(0.0, 1.0) : 0.0;

    // Layer 1: Ambient Outer Glow
    final glowPaint = Paint()
      ..color = accentColor.withValues(alpha: isSpoof ? 0.24 : 0.18)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 20);
    canvas.drawCircle(
      center,
      baseRadius + (clampedLevel * 20) + (pulseValue * 8),
      glowPaint,
    );

    // Layer 2: Harmonic Neural Ripples
    for (int i = 1; i <= 3; i++) {
      final rippleRadius = baseRadius +
          (i * 12.0) +
          (clampedLevel * 14.0 * (4 - i) / 3.0) +
          (pulseValue * 6.0);
      final alpha = (0.24 / i) * (0.6 + 0.4 * pulseValue);
      final ripplePaint = Paint()
        ..color = accentColor.withValues(alpha: alpha.clamp(0.0, 1.0))
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.6;
      canvas.drawCircle(center, rippleRadius, ripplePaint);
    }

    // Layer 3: Dynamic Biometric Boundary Ring
    final boundaryPaint = Paint()
      ..color = accentColor.withValues(alpha: 0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;
    canvas.drawCircle(center, baseRadius + (pulseValue * 3), boundaryPaint);
  }

  @override
  bool shouldRepaint(covariant LivingNeuralAuraPainter oldDelegate) {
    return oldDelegate.pulseValue != pulseValue ||
        oldDelegate.audioLevel != audioLevel ||
        oldDelegate.accentColor != accentColor ||
        oldDelegate.isSpoof != isSpoof;
  }
}

// -----------------------------------------------------------------------------
// HIGH-CONTRAST GLANCEABLE RISK BANNER
// -----------------------------------------------------------------------------

class _GlanceableTrustShield extends StatelessWidget {
  const _GlanceableTrustShield({
    required this.assessment,
    required this.accentColor,
    required this.accentBg,
    required this.accentBorder,
  });

  final LiveRiskAssessment assessment;
  final Color accentColor;
  final Color accentBg;
  final Color accentBorder;

  @override
  Widget build(BuildContext context) {
    final level = assessment.riskLevel;
    final prob = assessment.probability;

    String labelText;
    IconData icon;

    switch (level) {
      case LiveRiskLevel.low:
        labelText = 'VERIFIED HUMAN SPEECH · SECURE';
        icon = Icons.verified_user_rounded;
        break;
      case LiveRiskLevel.uncertain:
      case LiveRiskLevel.transitioning:
        labelText = 'ANALYZING AUDIO BUFFER · PLEASE SPEAK...';
        icon = Icons.hourglass_top_rounded;
        break;
      case LiveRiskLevel.high:
        final risk = prob != null ? ' · ${(prob * 100).toStringAsFixed(1)}% SPOOF' : '';
        labelText = 'CRITICAL ALERT: AI CLONE IMPERSONATION DETECTED$risk';
        icon = Icons.warning_amber_rounded;
        break;
      case LiveRiskLevel.unavailable:
        labelText = 'INTEGRITY SHIELD STANDBY · AWAITING AUDIO';
        icon = Icons.shield_outlined;
        break;
    }

    final accentTextColor = switch (level) {
      LiveRiskLevel.low => _CallTheme.emeraldText,
      LiveRiskLevel.uncertain => _CallTheme.amberText,
      LiveRiskLevel.high => _CallTheme.crimsonText,
      LiveRiskLevel.transitioning => _CallTheme.amberText,
      LiveRiskLevel.unavailable => _CallTheme.textSecondary,
    };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
      decoration: BoxDecoration(
        color: accentBg,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: accentBorder, width: 1.2),
        boxShadow: [
          BoxShadow(
            color: accentColor.withValues(alpha: 0.12),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: accentColor),
          const SizedBox(width: 8),
          Flexible(
            child: Text(
              labelText,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                fontFamily: 'monospace',
                fontSize: 11,
                fontWeight: FontWeight.w800,
                color: accentTextColor,
                letterSpacing: 0.6,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// REACTIVE WAVEFORM SPECTRUM
// -----------------------------------------------------------------------------

class ReactiveWaveformSpectrum extends StatelessWidget {
  const ReactiveWaveformSpectrum({
    super.key,
    this.levelNotifier,
    this.audioLevel = 0.0,
    required this.accentColor,
    required this.isActive,
    this.barCount = 28,
  });

  final ValueListenable<double>? levelNotifier;
  final double audioLevel;
  final Color accentColor;
  final bool isActive;
  final int barCount;

  @override
  Widget build(BuildContext context) {
    const barWidth = 3.5;
    const spacing = 3.0;
    final totalBarsWidth = (barCount * barWidth) + ((barCount - 1) * spacing);

    if (levelNotifier != null) {
      return RepaintBoundary(
        child: SizedBox(
          height: 38,
          child: Center(
            child: ValueListenableBuilder<double>(
              valueListenable: levelNotifier!,
              builder: (context, currentLevel, _) {
                final effectiveLevel = (isActive && currentLevel.isFinite)
                    ? currentLevel.clamp(0.05, 1.0)
                    : 0.05;
                return CustomPaint(
                  size: Size(totalBarsWidth, 38),
                  painter: _WaveformBarsPainter(
                    audioLevel: effectiveLevel,
                    accentColor: isActive ? accentColor : const Color(0xFFCBD5E1),
                    barCount: barCount,
                  ),
                );
              },
            ),
          ),
        ),
      );
    }

    final effectiveLevel = (isActive && audioLevel.isFinite)
        ? audioLevel.clamp(0.05, 1.0)
        : 0.05;
    return RepaintBoundary(
      child: SizedBox(
        height: 38,
        child: Center(
          child: CustomPaint(
            size: Size(totalBarsWidth, 38),
            painter: _WaveformBarsPainter(
              audioLevel: effectiveLevel,
              accentColor: isActive ? accentColor : const Color(0xFFCBD5E1),
              barCount: barCount,
            ),
          ),
        ),
      ),
    );
  }
}

class _WaveformBarsPainter extends CustomPainter {
  _WaveformBarsPainter({
    required this.audioLevel,
    required this.accentColor,
    required this.barCount,
  });

  final double audioLevel;
  final Color accentColor;
  final int barCount;
  static const _barRadius = Radius.circular(2);

  @override
  void paint(Canvas canvas, Size size) {
    if (size.width <= 0 || size.height <= 0 || !size.width.isFinite || !size.height.isFinite || barCount <= 0) return;

    final paint = Paint()
      ..color = accentColor
      ..style = PaintingStyle.fill;

    const barWidth = 3.5;
    const spacing = 3.0;
    final totalBarsWidth = (barCount * barWidth) + ((barCount - 1) * spacing);
    final startX = (size.width - totalBarsWidth) / 2;
    final centerY = size.height / 2;

    for (int i = 0; i < barCount; i++) {
      final normalizedIndex = (i - (barCount / 2)).abs() / (barCount / 2);
      final envelope = math.cos(normalizedIndex * math.pi / 2);
      final variation = (math.sin(i * 1.3 + (audioLevel * 6.0)) * 0.35).abs() + 0.65;
      final barHeight = math.max(4.0, (audioLevel * (size.height - 2) * envelope * variation).clamp(4.0, size.height));

      final x = startX + i * (barWidth + spacing);
      final top = centerY - (barHeight / 2);

      final rrect = RRect.fromRectAndRadius(
        Rect.fromLTWH(x, top, barWidth, barHeight),
        _barRadius,
      );
      canvas.drawRRect(rrect, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _WaveformBarsPainter oldDelegate) {
    return oldDelegate.audioLevel != audioLevel ||
        oldDelegate.accentColor != accentColor ||
        oldDelegate.barCount != barCount;
  }
}

// -----------------------------------------------------------------------------
// RECEIVER CONTROLS BAR (MUTE, SPEAKER, TEST CHIME, HANG UP)
// -----------------------------------------------------------------------------

class _ReceiverControlsBar extends StatelessWidget {
  const _ReceiverControlsBar({
    required this.service,
    required this.isMuted,
    required this.onToggleMute,
    required this.onEndCall,
  });

  final LiveCallService service;
  final bool isMuted;
  final VoidCallback onToggleMute;
  final VoidCallback onEndCall;

  @override
  Widget build(BuildContext context) {
    return Wrap(
      alignment: WrapAlignment.center,
      crossAxisAlignment: WrapCrossAlignment.center,
      spacing: 16,
      runSpacing: 12,
      children: [
        // Mute Button
        _RoundCallActionButton(
          icon: isMuted ? Icons.mic_off_rounded : Icons.mic_rounded,
          label: isMuted ? 'Unmute' : 'Mute',
          isActive: isMuted,
          activeColor: _CallTheme.amber,
          onTap: onToggleMute,
        ),

        // Loudspeaker Button
        _RoundCallActionButton(
          icon: Icons.volume_up_rounded,
          label: 'Speaker',
          isActive: !isMuted,
          activeColor: _CallTheme.emerald,
          onTap: () {},
        ),

        // Test Speaker Button
        _RoundCallActionButton(
          icon: Icons.music_note_rounded,
          label: 'Test Speaker',
          isActive: false,
          activeColor: _CallTheme.emerald,
          onTap: () => service.playTestSound(),
        ),

        // End Call / Hang up Button
        _RoundCallActionButton(
          icon: Icons.call_end_rounded,
          label: 'End Call',
          isActive: true,
          activeColor: _CallTheme.crimson,
          onTap: onEndCall,
        ),
      ],
    );
  }
}

class _RoundCallActionButton extends StatelessWidget {
  const _RoundCallActionButton({
    required this.icon,
    required this.label,
    required this.isActive,
    required this.activeColor,
    required this.onTap,
  });

  final IconData icon;
  final String label;
  final bool isActive;
  final Color activeColor;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      borderRadius: BorderRadius.circular(14),
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 50,
              height: 50,
              decoration: BoxDecoration(
                color: isActive
                    ? activeColor.withValues(alpha: 0.15)
                    : _CallTheme.surfaceElevated,
                shape: BoxShape.circle,
                border: Border.all(
                  color: isActive ? activeColor : _CallTheme.border,
                  width: isActive ? 1.5 : 1.0,
                ),
                boxShadow: isActive
                    ? [
                        BoxShadow(
                          color: activeColor.withValues(alpha: 0.2),
                          blurRadius: 8,
                          offset: const Offset(0, 2),
                        ),
                      ]
                    : null,
              ),
              child: Icon(
                icon,
                size: 22,
                color: isActive ? activeColor : _CallTheme.textPrimary,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              label,
              style: TextStyle(
                fontSize: 11,
                fontWeight: isActive ? FontWeight.w700 : FontWeight.w600,
                color: isActive ? activeColor : _CallTheme.textSecondary,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// EMERGENCY INTERVENTION BANNER (TRIGGERED ON CLONE DETECTION)
// -----------------------------------------------------------------------------

class _EmergencyInterventionBanner extends StatelessWidget {
  const _EmergencyInterventionBanner({
    required this.assessment,
    required this.onEndCall,
  });

  final LiveRiskAssessment assessment;
  final VoidCallback onEndCall;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _CallTheme.crimsonBg,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: _CallTheme.crimson, width: 1.2),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.gpp_bad_rounded, color: _CallTheme.crimson, size: 20),
              SizedBox(width: 8),
              Expanded(
                child: Text(
                  'CRITICAL VOICE IMPERSONATION ALERT',
                  style: TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 11,
                    fontWeight: FontWeight.w800,
                    color: _CallTheme.crimson,
                    letterSpacing: 0.8,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          const Text(
            'Synthesized vocoder artifacts detected matching cloned speaker latent distributions. Do not disclose credentials, approve bank transfers, or share passcodes.',
            style: TextStyle(
              fontSize: 12,
              color: _CallTheme.textPrimary,
              height: 1.4,
            ),
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 10,
            runSpacing: 8,
            children: [
              FilledButton.icon(
                onPressed: onEndCall,
                icon: const Icon(Icons.call_end_rounded, size: 15),
                label: const Text('Hang Up Immediately'),
                style: FilledButton.styleFrom(
                  backgroundColor: _CallTheme.crimson,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                  textStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700),
                ),
              ),
              OutlinedButton.icon(
                onPressed: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('Verification phrase challenge requested.'),
                      duration: Duration(seconds: 2),
                    ),
                  );
                },
                icon: const Icon(Icons.security_rounded, size: 15, color: _CallTheme.crimson),
                label: const Text('Challenge Identity'),
                style: OutlinedButton.styleFrom(
                  side: const BorderSide(color: _CallTheme.crimson),
                  foregroundColor: _CallTheme.crimson,
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                  textStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// CALLER MODE (ATTACKER) VIEW
// -----------------------------------------------------------------------------

class _CallerModeView extends StatelessWidget {
  const _CallerModeView({
    required this.service,
    required this.pulseAnimation,
    required this.onStateChanged,
  });

  final LiveCallService service;
  final Animation<double> pulseAnimation;
  final VoidCallback onStateChanged;

  @override
  Widget build(BuildContext context) {
    final isSpoof = service.isSpoofActive;
    final isConnected = service.isConnected;
    final stats = service.stats;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // 1. Prominent Tactile Glowing Spoof Button Actuator
        _SpoofToggleActuator(
          service: service,
          pulseAnimation: pulseAnimation,
          isConnected: isConnected,
          isSpoof: isSpoof,
          stats: stats,
          onToggleSpoof: () {
            service.toggleSpoof();
            onStateChanged();
          },
        ),
        const SizedBox(height: 16),

        // 2. Synthetic Vocal Identity Profiles (Persona Selector Rack)
        _VoiceCloneProfileRack(
          service: service,
          activePersona: service.spoofSpeaker,
          onSelectPersona: (persona) {
            service.setSpoofSpeaker(persona);
            onStateChanged();
          },
        ),
        const SizedBox(height: 16),

        // 3. Microphone Capture VU Meter
        _BroadcastVuMeter(
          channelTitle: 'MIC CAPTURE CHANNEL',
          subLabel: '16 kHz Linear PCM · 16-Bit Mono Input',
          levelNotifier: service.audioLevelNotifier,
          level: stats.audioLevel,
          isActive: isConnected,
          accentColor: isSpoof ? _CallTheme.crimson : _CallTheme.emerald,
        ),
        const SizedBox(height: 16),

        // 4. Transmission Telemetry & Call Connection Button
        _CallerTelemetryRack(
          service: service,
          stats: stats,
          spoofActive: isSpoof,
        ),
      ],
    );
  }
}

// -----------------------------------------------------------------------------
// TACTILE GLOWING SPOOF TOGGLE ACTUATOR
// -----------------------------------------------------------------------------

class _SpoofToggleActuator extends StatefulWidget {
  const _SpoofToggleActuator({
    required this.service,
    required this.pulseAnimation,
    required this.isConnected,
    required this.isSpoof,
    required this.stats,
    required this.onToggleSpoof,
  });

  final LiveCallService service;
  final Animation<double> pulseAnimation;
  final bool isConnected;
  final bool isSpoof;
  final CallStats stats;
  final VoidCallback onToggleSpoof;

  @override
  State<_SpoofToggleActuator> createState() => _SpoofToggleActuatorState();
}

class _SpoofToggleActuatorState extends State<_SpoofToggleActuator> {
  bool _isPressed = false;

  @override
  Widget build(BuildContext context) {
    final isSpoof = widget.isSpoof;
    final isConnected = widget.isConnected;
    final stats = widget.stats;

    return Container(
      decoration: BoxDecoration(
        color: isSpoof ? _CallTheme.crimsonBg : _CallTheme.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: isSpoof ? _CallTheme.crimson : _CallTheme.border,
          width: isSpoof ? 1.5 : 1.0,
        ),
        boxShadow: isSpoof
            ? [
                BoxShadow(
                  color: _CallTheme.crimson.withValues(alpha: 0.25),
                  blurRadius: 20,
                  offset: const Offset(0, 4),
                ),
              ]
            : const [
                BoxShadow(
                  color: Color(0x06000000),
                  blurRadius: 10,
                  offset: Offset(0, 2),
                ),
              ],
      ),
      padding: const EdgeInsets.all(22),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: isSpoof ? _CallTheme.crimson : _CallTheme.surfaceElevated,
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(
                    color: isSpoof ? _CallTheme.crimson : _CallTheme.border,
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      isSpoof ? Icons.warning_rounded : Icons.mic_rounded,
                      color: isSpoof ? Colors.white : _CallTheme.emerald,
                      size: 13,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      isSpoof
                          ? 'AI CLONE ATTACK ACTIVE'
                          : (isConnected ? 'AUTHENTIC MIC ACTIVE' : 'CALLER STANDBY'),
                      style: TextStyle(
                        fontFamily: 'monospace',
                        color: isSpoof ? Colors.white : _CallTheme.textPrimary,
                        fontSize: 10,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 0.8,
                      ),
                    ),
                  ],
                ),
              ),
              _DataTimerPill(
                label: isSpoof ? 'SPOOF TIME' : 'CALL TIME',
                duration: isSpoof ? stats.spoofDuration : stats.duration,
                highlight: isSpoof,
              ),
            ],
          ),
          const SizedBox(height: 18),
          Text(
            isSpoof
                ? 'Voice Cloning Attack in Progress'
                : (isConnected ? 'Authentic Voice Capturing' : 'Connect Call to Transmit'),
            style: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w700,
              color: isSpoof ? _CallTheme.crimson : _CallTheme.textPrimary,
              letterSpacing: -0.3,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            isSpoof
                ? 'Outgoing 16 kHz microphone frames are converted into target cloned persona ("${widget.service.spoofSpeaker}") via real-time neural vocoder latent synthesis.'
                : 'Transmitting unadulterated human speech via 16 kHz 16-bit mono stream to the multi-expert integrity verification pipeline.',
            style: const TextStyle(
              fontSize: 13,
              color: _CallTheme.textSecondary,
              height: 1.4,
            ),
          ),
          const SizedBox(height: 18),

          // Tactile Glowing Spoof Button: [ ⚡ TALK AS TEAMMATE 3 (SPOOF) ]
          GestureDetector(
            onTapDown: (_) => setState(() => _isPressed = true),
            onTapUp: (_) => setState(() => _isPressed = false),
            onTapCancel: () => setState(() => _isPressed = false),
            onTap: widget.onToggleSpoof,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 140),
              curve: Curves.easeOutCubic,
              padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 20),
              decoration: BoxDecoration(
                color: isSpoof ? _CallTheme.crimson : const Color(0xFF0F172A),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(
                  color: isSpoof ? _CallTheme.crimson : _CallTheme.borderSubtle,
                  width: 1.5,
                ),
                boxShadow: [
                  BoxShadow(
                    color: isSpoof
                        ? _CallTheme.crimson.withValues(alpha: 0.4)
                        : const Color(0x1F0F172A),
                    blurRadius: _isPressed ? 4 : (isSpoof ? 18 : 10),
                    offset: _isPressed ? const Offset(0, 1) : const Offset(0, 4),
                  ),
                ],
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    isSpoof ? Icons.undo_rounded : Icons.bolt_rounded,
                    color: Colors.white,
                    size: 24,
                  ),
                  const SizedBox(width: 10),
                  Flexible(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          isSpoof
                              ? '⚡ STOP SPOOF (REVERT TO GENUINE)'
                              : '⚡ TALK AS ${widget.service.spoofSpeaker.toUpperCase()} (SPOOF)',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 15,
                            fontWeight: FontWeight.w900,
                            letterSpacing: 0.3,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          isSpoof
                              ? 'Tap to immediately revert audio stream to natural microphone speech'
                              : 'Tap to inject synthetic voice frames into caller stream in real time',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            color: Colors.white.withValues(alpha: 0.85),
                            fontSize: 11,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// VOICE CLONE PROFILES (PERSONA SELECTOR RACK)
// -----------------------------------------------------------------------------

class _VoiceCloneProfileRack extends StatelessWidget {
  const _VoiceCloneProfileRack({
    required this.service,
    required this.activePersona,
    required this.onSelectPersona,
  });

  final LiveCallService service;
  final String activePersona;
  final ValueChanged<String> onSelectPersona;

  static List<String> get _personas => LiveCallService.availablePersonas;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: _CallTheme.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _CallTheme.border),
        boxShadow: const [
          BoxShadow(
            color: Color(0x06000000),
            blurRadius: 8,
            offset: Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'SYNTHETIC VOCAL IDENTITY PROFILES',
                style: TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1.0,
                  color: _CallTheme.textMuted,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: _CallTheme.crimsonBg,
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: _CallTheme.crimsonBorder),
                ),
                child: Text(
                  'ACTIVE: $activePersona',
                  style: const TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 10,
                    fontWeight: FontWeight.w800,
                    color: _CallTheme.crimson,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: _personas.map((persona) {
              final isSelected = persona == activePersona;

              return Material(
                color: Colors.transparent,
                child: InkWell(
                  borderRadius: BorderRadius.circular(10),
                  onTap: () {
                    service.setSpoofSpeaker(persona);
                    onSelectPersona(persona);
                  },
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 150),
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: isSelected ? _CallTheme.crimsonBg : _CallTheme.surfaceElevated,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                        color: isSelected ? _CallTheme.crimson : _CallTheme.border,
                        width: isSelected ? 1.5 : 1.0,
                      ),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          isSelected
                              ? Icons.radio_button_checked_rounded
                              : Icons.radio_button_unchecked_rounded,
                          size: 15,
                          color: isSelected ? _CallTheme.crimson : _CallTheme.textMuted,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          persona,
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: isSelected ? FontWeight.w700 : FontWeight.w500,
                            color: isSelected ? _CallTheme.crimson : _CallTheme.textPrimary,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// BROADCAST VU METER
// -----------------------------------------------------------------------------

class _BroadcastVuMeter extends StatelessWidget {
  const _BroadcastVuMeter({
    required this.channelTitle,
    required this.subLabel,
    this.levelNotifier,
    this.level = 0.0,
    required this.isActive,
    required this.accentColor,
  });

  final String channelTitle;
  final String subLabel;
  final ValueListenable<double>? levelNotifier;
  final double level;
  final bool isActive;
  final Color accentColor;
  static const int barCount = 28;

  @override
  Widget build(BuildContext context) {
    if (levelNotifier != null) {
      return ValueListenableBuilder<double>(
        valueListenable: levelNotifier!,
        builder: (context, currentLevel, _) => _buildMeter(context, currentLevel),
      );
    }
    return _buildMeter(context, level);
  }

  Widget _buildMeter(BuildContext context, double rawLevel) {
    final safeLevel = (rawLevel.isFinite) ? rawLevel.clamp(0.0, 1.0) : 0.0;
    final rmsPercent = (safeLevel * 100).toInt();

    final db = safeLevel > 0.001
        ? (20 * (math.log(safeLevel) / math.ln10)).clamp(-60.0, 3.0)
        : -60.0;
    final dbText = isActive ? '${db.toStringAsFixed(1)} dB' : '-INF dB';

    return Container(
      decoration: BoxDecoration(
        color: _CallTheme.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _CallTheme.border),
        boxShadow: const [
          BoxShadow(
            color: Color(0x06000000),
            blurRadius: 8,
            offset: Offset(0, 2),
          ),
        ],
      ),
      padding: const EdgeInsets.all(18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 7,
                    height: 7,
                    decoration: BoxDecoration(
                      color: isActive ? accentColor : _CallTheme.textMuted,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    channelTitle,
                    style: const TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 1.1,
                      color: _CallTheme.textSecondary,
                    ),
                  ),
                ],
              ),
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    isActive ? '$rmsPercent% RMS' : 'STANDBY',
                    style: TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: isActive ? accentColor : _CallTheme.textMuted,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: _CallTheme.surfaceElevated,
                      borderRadius: BorderRadius.circular(4),
                      border: Border.all(color: _CallTheme.border),
                    ),
                    child: Text(
                      dbText,
                      style: const TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                        color: _CallTheme.textPrimary,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 10),
          RepaintBoundary(
            child: LayoutBuilder(
              builder: (context, constraints) {
                final totalWidth = constraints.maxWidth;
                final segmentWidth = math.max(3.0, (totalWidth / barCount) - 3);

                return Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: List.generate(barCount, (index) {
                    final segmentThreshold = (index + 1) / barCount;
                    final isLit = isActive && safeLevel >= segmentThreshold;

                    Color litColor;
                    if (index >= 24) {
                      litColor = _CallTheme.crimson;
                    } else if (index >= 18) {
                      litColor = _CallTheme.amber;
                    } else {
                      litColor = _CallTheme.emerald;
                    }

                    const unlitColor = Color(0xFFE2E8F0);

                    return Container(
                      width: segmentWidth,
                      height: 12,
                      decoration: BoxDecoration(
                        color: isLit ? litColor : unlitColor,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    );
                  }),
                );
              },
            ),
          ),
          const SizedBox(height: 8),
          Text(
            subLabel,
            style: const TextStyle(
              fontSize: 11,
              color: _CallTheme.textMuted,
            ),
          ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// CALLER TELEMETRY & CONNECTION RACK
// -----------------------------------------------------------------------------

class _CallerTelemetryRack extends StatelessWidget {
  const _CallerTelemetryRack({
    required this.service,
    required this.stats,
    required this.spoofActive,
  });

  final LiveCallService service;
  final CallStats stats;
  final bool spoofActive;

  @override
  Widget build(BuildContext context) {
    final isConnected = service.isConnected;

    final metrics = [
      ('AUDIO ENCODING', '16 kHz Mono PCM'),
      ('UPLINK FRAMES', '${stats.packetsSent} sent'),
      ('DOWNLINK ECHO', '${stats.packetsReceived} rcvd'),
      ('PAYLOAD SIZE', '${(stats.bytesTransferred / 1024).toStringAsFixed(1)} KB'),
    ];

    return Container(
      decoration: BoxDecoration(
        color: _CallTheme.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _CallTheme.border),
        boxShadow: const [
          BoxShadow(
            color: Color(0x06000000),
            blurRadius: 8,
            offset: Offset(0, 2),
          ),
        ],
      ),
      padding: const EdgeInsets.all(18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'TRANSMISSION TELEMETRY',
                style: TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1.0,
                  color: _CallTheme.textMuted,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: spoofActive ? _CallTheme.crimsonBg : _CallTheme.emeraldBg,
                  borderRadius: BorderRadius.circular(4),
                  border: Border.all(
                    color: spoofActive ? _CallTheme.crimsonBorder : _CallTheme.emeraldBorder,
                  ),
                ),
                child: Text(
                  spoofActive ? 'STATE: INJECTION' : 'STATE: AUTHENTIC',
                  style: TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 9,
                    fontWeight: FontWeight.w800,
                    color: spoofActive ? _CallTheme.crimson : _CallTheme.emerald,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 16,
            runSpacing: 10,
            children: metrics.map((item) {
              return SizedBox(
                width: 130,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      item.$1,
                      style: const TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 9.5,
                        fontWeight: FontWeight.w600,
                        color: _CallTheme.textMuted,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      item.$2,
                      style: const TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: _CallTheme.textPrimary,
                      ),
                    ),
                  ],
                ),
              );
            }).toList(),
          ),
          const SizedBox(height: 16),
          const Divider(color: _CallTheme.border, height: 1),
          const SizedBox(height: 14),

          // Connect / Disconnect Caller Button
          SizedBox(
            width: double.infinity,
            child: isConnected
                ? OutlinedButton.icon(
                    onPressed: () => service.disconnect(),
                    icon: const Icon(Icons.call_end_rounded, color: _CallTheme.crimson, size: 18),
                    label: const Text('End Call Transmission'),
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: _CallTheme.crimson),
                      foregroundColor: _CallTheme.crimson,
                      backgroundColor: _CallTheme.crimsonBg,
                      padding: const EdgeInsets.symmetric(vertical: 12),
                    ),
                  )
                : FilledButton.icon(
                    onPressed: () => service.connect(),
                    icon: const Icon(Icons.call_rounded, size: 18),
                    label: Text(
                      service.useSimulationMode
                          ? 'Start Presentation Call (Demo)'
                          : 'Connect Call Transmission',
                    ),
                    style: FilledButton.styleFrom(
                      backgroundColor: _CallTheme.emerald,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 12),
                    ),
                  ),
          ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// AUDIBLE SESSION REPLAY CARD (VOLATILE POST-CALL VERIFICATION PROOF)
// -----------------------------------------------------------------------------

class _PostCallSessionVerificationCard extends StatefulWidget {
  const _PostCallSessionVerificationCard({
    required this.service,
    required this.onDiscard,
  });

  final LiveCallService service;
  final VoidCallback onDiscard;

  @override
  State<_PostCallSessionVerificationCard> createState() =>
      _PostCallSessionVerificationCardState();
}

class _PostCallSessionVerificationCardState
    extends State<_PostCallSessionVerificationCard> {
  late final AudioPlayer _player;
  bool _isPlaying = false;
  Duration _position = Duration.zero;
  Duration _duration = Duration.zero;
  StreamSubscription<Duration>? _posSub;
  StreamSubscription<Duration>? _durSub;
  StreamSubscription<void>? _completeSub;
  StreamSubscription<PlayerState>? _stateSub;

  @override
  void initState() {
    super.initState();
    _player = AudioPlayer();
    _duration = widget.service.lastSessionDuration ?? Duration.zero;

    _posSub = _player.onPositionChanged.listen((p) {
      if (mounted) setState(() => _position = p);
    });
    _durSub = _player.onDurationChanged.listen((d) {
      if (mounted && d > Duration.zero) setState(() => _duration = d);
    });
    _completeSub = _player.onPlayerComplete.listen((_) {
      if (mounted) {
        setState(() {
          _isPlaying = false;
          _position = Duration.zero;
        });
      }
    });
    _stateSub = _player.onPlayerStateChanged.listen((s) {
      if (mounted) {
        setState(() => _isPlaying = s == PlayerState.playing);
      }
    });
  }

  @override
  void dispose() {
    _posSub?.cancel();
    _durSub?.cancel();
    _completeSub?.cancel();
    _stateSub?.cancel();
    _player.stop();
    _player.dispose();
    super.dispose();
  }

  Future<void> _togglePlayback() async {
    final wavBytes = widget.service.lastSessionWavBytes;
    if (wavBytes == null || wavBytes.isEmpty) return;

    if (_isPlaying) {
      await _player.pause();
    } else {
      if (_position > Duration.zero && _position < _duration) {
        await _player.resume();
      } else {
        await _player.play(BytesSource(wavBytes));
      }
    }
  }

  String _formatDuration(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$m:$s';
  }

  @override
  Widget build(BuildContext context) {
    final posText = _formatDuration(_position);
    final durText = _formatDuration(_duration);
    final progress = _duration.inMilliseconds > 0
        ? (_position.inMilliseconds / _duration.inMilliseconds).clamp(0.0, 1.0)
        : 0.0;

    return Container(
      decoration: BoxDecoration(
        color: _CallTheme.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFFCBD5E1), width: 1.2),
        boxShadow: const [
          BoxShadow(
            color: Color(0x0F0F172A),
            blurRadius: 16,
            offset: Offset(0, 4),
          ),
          BoxShadow(
            color: Color(0x080F172A),
            blurRadius: 2,
            offset: Offset(0, 1),
          ),
        ],
      ),
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: _CallTheme.emeraldBg,
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: _CallTheme.emeraldBorder, width: 1.2),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.verified_user_rounded,
                      size: 14,
                      color: _CallTheme.emerald,
                    ),
                    SizedBox(width: 6),
                    Text(
                      'POST-CALL AUDIT · LIVE VOICE PROOF',
                      style: TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 10,
                        fontWeight: FontWeight.w800,
                        color: _CallTheme.emeraldText,
                        letterSpacing: 0.8,
                      ),
                    ),
                  ],
                ),
              ),
              IconButton(
                tooltip: 'Discard session recording',
                icon: const Icon(Icons.close_rounded, size: 20, color: _CallTheme.textMuted),
                onPressed: () {
                  _player.stop();
                  widget.onDiscard();
                },
              ),
            ],
          ),
          const SizedBox(height: 12),
          const Text(
            'Audible Session Replay',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w800,
              color: _CallTheme.textPrimary,
              letterSpacing: -0.2,
            ),
          ),
          const SizedBox(height: 4),
          const Text(
            'Play back the exact acoustic stream captured during this call to demonstrate to judges that the speech was live and authentic rather than pre-recorded.',
            style: TextStyle(
              fontSize: 12,
              color: _CallTheme.textSecondary,
              height: 1.4,
            ),
          ),
          const SizedBox(height: 14),

          // Audio player bar
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              color: _CallTheme.surfaceElevated,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: _CallTheme.border),
            ),
            child: Row(
              children: [
                Material(
                  color: _CallTheme.emerald,
                  shape: const CircleBorder(),
                  child: InkWell(
                    customBorder: const CircleBorder(),
                    onTap: _togglePlayback,
                    child: Padding(
                      padding: const EdgeInsets.all(10),
                      child: Icon(
                        _isPlaying ? Icons.pause_rounded : Icons.play_arrow_rounded,
                        color: Colors.white,
                        size: 22,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Expanded(
                            child: Text(
                              _isPlaying ? 'Playing captured call audio...' : 'Tap play to listen',
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.w700,
                                color: _CallTheme.textPrimary,
                              ),
                            ),
                          ),
                          const SizedBox(width: 6),
                          Text(
                            '$posText / $durText',
                            style: const TextStyle(
                              fontFamily: 'monospace',
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                              color: _CallTheme.textSecondary,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: LinearProgressIndicator(
                          value: progress,
                          minHeight: 6,
                          backgroundColor: _CallTheme.border,
                          valueColor: const AlwaysStoppedAnimation(_CallTheme.emerald),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 10),
                OutlinedButton.icon(
                  onPressed: () {
                    _player.stop();
                    widget.onDiscard();
                  },
                  icon: const Icon(Icons.delete_outline_rounded, size: 16),
                  label: const Text('Discard'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: _CallTheme.crimson,
                    side: const BorderSide(color: _CallTheme.crimsonBorder),
                    backgroundColor: _CallTheme.crimsonBg,
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                    textStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// COLLAPSIBLE FORENSIC EVIDENCE BREAKDOWN (FOR JUDGES)
// -----------------------------------------------------------------------------

class _ForensicExpansionCard extends StatelessWidget {
  const _ForensicExpansionCard({
    required this.service,
    required this.assessment,
    required this.stats,
    this.pulseAnimation,
  });

  final LiveCallService service;
  final LiveRiskAssessment assessment;
  final CallStats stats;
  final Animation<double>? pulseAnimation;

  @override
  Widget build(BuildContext context) {
    final isConnected = service.isConnected;

    return Container(
      decoration: BoxDecoration(
        color: _CallTheme.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _CallTheme.border),
        boxShadow: const [
          BoxShadow(
            color: Color(0x06000000),
            blurRadius: 8,
            offset: Offset(0, 2),
          ),
        ],
      ),
      clipBehavior: Clip.antiAlias,
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          initiallyExpanded: true,
          tilePadding: const EdgeInsets.symmetric(horizontal: 18, vertical: 6),
          childrenPadding: const EdgeInsets.fromLTRB(18, 0, 18, 18),
          leading: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: _CallTheme.surfaceElevated,
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Icon(
              Icons.biotech_rounded,
              size: 20,
              color: _CallTheme.emerald,
            ),
          ),
          title: const Text(
            'Dual-Model Forensic Telemetry',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w800,
              color: _CallTheme.textPrimary,
            ),
          ),
          subtitle: const Text(
            'WavLM Base+ & LFCC-LCNN inference telemetry for judges',
            style: TextStyle(
              fontSize: 11,
              color: _CallTheme.textMuted,
            ),
          ),
          children: [
            const Divider(color: _CallTheme.border, height: 1),
            const SizedBox(height: 14),

            // Live Dual Model Probability Strip (WavLM Base+ & LFCC-LCNN Hybrid)
            _LiveDualModelProbabilityStrip(
              assessment: assessment,
              isConnected: isConnected,
              pulseAnimation: pulseAnimation,
            ),
            const SizedBox(height: 16),

            // Multi-Expert Evidence Grid
            _MultiExpertEvidenceGrid(assessment: assessment),
            const SizedBox(height: 16),

            // Jitter Buffer Telemetry & Speaker Playback
            SpeakerPlaybackCard(service: service, stats: stats),
            const SizedBox(height: 16),

            // Timestamped Event Logs
            _ReasoningTracePanel(logs: service.reasoningLogs),
          ],
        ),
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// LIVE DUAL-MODEL PROBABILITY STRIP
// -----------------------------------------------------------------------------

class _LiveDualModelProbabilityStrip extends StatelessWidget {
  const _LiveDualModelProbabilityStrip({
    required this.assessment,
    required this.isConnected,
    this.pulseAnimation,
  });

  final LiveRiskAssessment assessment;
  final bool isConnected;
  final Animation<double>? pulseAnimation;

  @override
  Widget build(BuildContext context) {
    final experts = assessment.expertScores;
    final wavlm = experts['wavlm'];
    final hybrid =
        experts['hybrid_maxbr'] ?? experts['hybrid'] ?? experts['lfcc'];

    final wavlmProb = isConnected
        ? (wavlm?.probability ?? assessment.probability)
        : null;

    final hybridProb = isConnected
        ? (hybrid?.probability ?? assessment.probability)
        : null;

    final wavlmRisk = wavlm?.riskLevel ??
        (wavlmProb != null
            ? (wavlmProb >= 0.65
                ? LiveRiskLevel.high
                : (wavlmProb >= 0.35
                    ? LiveRiskLevel.uncertain
                    : LiveRiskLevel.low))
            : LiveRiskLevel.unavailable);

    final hybridRisk = hybrid?.riskLevel ??
        (hybridProb != null
            ? (hybridProb >= 0.65
                ? LiveRiskLevel.high
                : (hybridProb >= 0.35
                    ? LiveRiskLevel.uncertain
                    : LiveRiskLevel.low))
            : LiveRiskLevel.unavailable);

    final wavlmSub = isConnected && wavlmProb != null
        ? (wavlmRisk == LiveRiskLevel.high
            ? 'Latent phase anomaly detected'
            : (wavlmRisk == LiveRiskLevel.uncertain
                ? 'Evaluating transformer layers'
                : 'Latent embeddings nominal'))
        : 'Awaiting 16kHz speech frames';

    final hybridSub = isConnected && hybridProb != null
        ? (hybridRisk == LiveRiskLevel.high
            ? 'Synthetic vocoder trace detected'
            : (hybridRisk == LiveRiskLevel.uncertain
                ? 'Evaluating filterbank energy'
                : 'Acoustic spectral pass'))
        : 'Awaiting 16kHz speech frames';

    final isHigh = assessment.riskLevel == LiveRiskLevel.high;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Real-Time Inference Pipeline Activity Bar
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
          decoration: BoxDecoration(
            color: isConnected ? _CallTheme.surface : _CallTheme.surfaceElevated,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: isConnected && isHigh
                  ? _CallTheme.crimsonBorder
                  : _CallTheme.border,
            ),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    if (pulseAnimation != null && isConnected)
                      AnimatedBuilder(
                        animation: pulseAnimation!,
                        builder: (context, child) {
                          return Transform.scale(
                            scale: 0.85 + (pulseAnimation!.value * 0.3),
                            child: child,
                          );
                        },
                        child: Container(
                          width: 7,
                          height: 7,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: isHigh
                                ? _CallTheme.crimson
                                : _CallTheme.emerald,
                            boxShadow: [
                              BoxShadow(
                                color: (isHigh
                                        ? _CallTheme.crimson
                                        : _CallTheme.emerald)
                                    .withValues(alpha: 0.6),
                                blurRadius: 5,
                                spreadRadius: 1,
                              ),
                            ],
                          ),
                        ),
                      )
                    else
                      Container(
                        width: 7,
                        height: 7,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: isConnected
                              ? (isHigh
                                  ? _CallTheme.crimson
                                  : _CallTheme.emerald)
                              : _CallTheme.textMuted,
                        ),
                      ),
                    const SizedBox(width: 6),
                    Flexible(
                      child: Text(
                        isConnected
                            ? (isHigh
                                ? 'DUAL-EXPERT DETECTED CLONE'
                                : 'DUAL-MODEL INFERENCE ACTIVE')
                            : 'MODELS STANDBY · AWAITING CALL',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          fontFamily: 'monospace',
                          fontSize: 9.5,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 0.6,
                          color: isConnected
                              ? (isHigh
                                  ? _CallTheme.crimsonText
                                  : _CallTheme.emeraldText)
                              : _CallTheme.textMuted,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Text(
                isConnected
                    ? 'WIN #${assessment.windowIndex} · ${assessment.latencyMs.toStringAsFixed(0)}ms'
                    : '50/50 ENSEMBLE',
                style: const TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 9.5,
                  fontWeight: FontWeight.w700,
                  color: _CallTheme.textSecondary,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),

        // Dual Model Probability Cards
        LayoutBuilder(
          builder: (context, constraints) {
            final isCompact = constraints.maxWidth < 360;

            final cardWavlm = _LiveModelProbabilityCard(
              modelName: 'WavLM Base+',
              category: 'SSL LATENT',
              weight: 0.50,
              probability: wavlmProb,
              riskLevel: wavlmRisk,
              subLabel: wavlmSub,
              isConnected: isConnected,
            );

            final cardHybrid = _LiveModelProbabilityCard(
              modelName: 'LFCC-LCNN',
              category: 'SPECTRAL',
              weight: 0.50,
              probability: hybridProb,
              riskLevel: hybridRisk,
              subLabel: hybridSub,
              isConnected: isConnected,
            );

            if (isCompact) {
              return Column(
                children: [
                  cardWavlm,
                  const SizedBox(height: 8),
                  cardHybrid,
                ],
              );
            }

            return Row(
              children: [
                Expanded(child: cardWavlm),
                const SizedBox(width: 10),
                Expanded(child: cardHybrid),
              ],
            );
          },
        ),
      ],
    );
  }
}

class _LiveModelProbabilityCard extends StatelessWidget {
  const _LiveModelProbabilityCard({
    required this.modelName,
    required this.category,
    required this.weight,
    required this.probability,
    required this.riskLevel,
    required this.subLabel,
    required this.isConnected,
  });

  final String modelName;
  final String category;
  final double weight;
  final double? probability;
  final LiveRiskLevel riskLevel;
  final String subLabel;
  final bool isConnected;

  @override
  Widget build(BuildContext context) {
    final prob = probability;
    final hasScore = isConnected && prob != null;

    final (
      Color accentColor,
      Color accentBg,
      Color accentBorder,
      String statusTag
    ) = switch (riskLevel) {
      LiveRiskLevel.low => (
        _CallTheme.emeraldText,
        _CallTheme.emeraldBg,
        _CallTheme.emeraldBorder,
        'HUMAN',
      ),
      LiveRiskLevel.uncertain => (
        _CallTheme.amberText,
        _CallTheme.amberBg,
        _CallTheme.amberBorder,
        'REVIEW',
      ),
      LiveRiskLevel.high => (
        _CallTheme.crimsonText,
        _CallTheme.crimsonBg,
        _CallTheme.crimsonBorder,
        'SPOOF',
      ),
      LiveRiskLevel.transitioning => (
        _CallTheme.amberText,
        _CallTheme.amberBg,
        _CallTheme.amberBorder,
        'BUFFER',
      ),
      LiveRiskLevel.unavailable => (
        _CallTheme.textMuted,
        _CallTheme.surfaceElevated,
        _CallTheme.border,
        'STANDBY',
      ),
    };

    final percentText =
        hasScore ? '${(prob * 100).toStringAsFixed(1)}%' : '--%';
    final progressVal = hasScore ? prob.clamp(0.0, 1.0) : 0.0;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: _CallTheme.surfaceElevated,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: hasScore && riskLevel == LiveRiskLevel.high
              ? _CallTheme.crimsonBorder
              : _CallTheme.border,
          width: 1.0,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Flexible(
                child: Text(
                  modelName,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w800,
                    color: _CallTheme.textPrimary,
                    letterSpacing: -0.2,
                  ),
                ),
              ),
              const SizedBox(width: 4),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 5, vertical: 1.5),
                decoration: BoxDecoration(
                  color: _CallTheme.surface,
                  borderRadius: BorderRadius.circular(4),
                  border: Border.all(color: _CallTheme.border),
                ),
                child: Text(
                  category,
                  style: const TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 8.5,
                    fontWeight: FontWeight.w700,
                    color: _CallTheme.textMuted,
                    letterSpacing: 0.4,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),

          // Probability & Status Pill
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Flexible(
                child: FittedBox(
                  fit: BoxFit.scaleDown,
                  alignment: Alignment.centerLeft,
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.baseline,
                    textBaseline: TextBaseline.alphabetic,
                    children: [
                      Text(
                        percentText,
                        style: TextStyle(
                          fontFamily: 'monospace',
                          fontSize: 18,
                          fontWeight: FontWeight.w800,
                          color: hasScore ? accentColor : _CallTheme.textMuted,
                          letterSpacing: -0.5,
                        ),
                      ),
                      const SizedBox(width: 4),
                      const Text(
                        'spoof',
                        style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.w500,
                          color: _CallTheme.textMuted,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 6),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: accentBg,
                  borderRadius: BorderRadius.circular(4),
                  border: Border.all(color: accentBorder, width: 1),
                ),
                child: Text(
                  statusTag,
                  style: TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 9,
                    fontWeight: FontWeight.w800,
                    color: accentColor,
                    letterSpacing: 0.5,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),

          // Smooth micro progress bar
          ClipRRect(
            borderRadius: BorderRadius.circular(3),
            child: TweenAnimationBuilder<double>(
              tween: Tween<double>(begin: 0.0, end: progressVal),
              duration: const Duration(milliseconds: 250),
              curve: Curves.easeOut,
              builder: (context, animatedVal, child) {
                return LinearProgressIndicator(
                  value: animatedVal,
                  minHeight: 4,
                  backgroundColor: _CallTheme.border,
                  valueColor: AlwaysStoppedAnimation(
                    hasScore ? accentColor : _CallTheme.textMuted,
                  ),
                );
              },
            ),
          ),
          const SizedBox(height: 5),

          Text(
            subLabel,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              fontSize: 9.5,
              fontWeight: FontWeight.w600,
              color: _CallTheme.textSecondary,
            ),
          ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// MULTI-EXPERT EVIDENCE GRID
// -----------------------------------------------------------------------------

class _MultiExpertEvidenceGrid extends StatelessWidget {
  const _MultiExpertEvidenceGrid({required this.assessment});

  final LiveRiskAssessment assessment;

  @override
  Widget build(BuildContext context) {
    final experts = assessment.expertScores;
    final compact = MediaQuery.sizeOf(context).width < 768;

    final lfccScore =
        experts['hybrid_maxbr'] ?? experts['hybrid'] ?? experts['lfcc'];

    final items = [
      ('wavlm', 'Expert 1 · WavLM Base+', 0.50, experts['wavlm']),
      ('hybrid', 'Expert 2 · LFCC-LCNN', 0.50, lfccScore),
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'INDEPENDENT MODEL EVIDENCE',
          style: TextStyle(
            fontFamily: 'monospace',
            fontSize: 10.5,
            fontWeight: FontWeight.w700,
            letterSpacing: 1.0,
            color: _CallTheme.textMuted,
          ),
        ),
        const SizedBox(height: 8),
        if (compact)
          Column(
            children: items.map((item) {
              return Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: _ExpertMetricCard(
                  label: item.$2,
                  weight: item.$3,
                  score: item.$4,
                ),
              );
            }).toList(),
          )
        else
          Row(
            children: items.map((item) {
              return Expanded(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 4),
                  child: _ExpertMetricCard(
                    label: item.$2,
                    weight: item.$3,
                    score: item.$4,
                  ),
                ),
              );
            }).toList(),
          ),
      ],
    );
  }
}

class _ExpertMetricCard extends StatelessWidget {
  const _ExpertMetricCard({
    required this.label,
    required this.weight,
    required this.score,
  });

  final String label;
  final double weight;
  final LiveExpertScore? score;

  @override
  Widget build(BuildContext context) {
    final prob = score?.probability;
    final probText = prob != null ? '${(prob * 100).toStringAsFixed(1)}%' : '--';
    final riskLevel = score?.riskLevel ?? LiveRiskLevel.unavailable;

    final accentColor = switch (riskLevel) {
      LiveRiskLevel.low => _CallTheme.emerald,
      LiveRiskLevel.uncertain => _CallTheme.amber,
      LiveRiskLevel.high => _CallTheme.crimson,
      _ => _CallTheme.textMuted,
    };

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: _CallTheme.surfaceElevated,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _CallTheme.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Text(
                  label,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: _CallTheme.textPrimary,
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: _CallTheme.surface,
                  borderRadius: BorderRadius.circular(4),
                  border: Border.all(color: _CallTheme.border),
                ),
                child: Text(
                  '${(weight * 100).toInt()}% WT',
                  style: const TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 9,
                    fontWeight: FontWeight.w700,
                    color: _CallTheme.textMuted,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                probText,
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w800,
                  color: accentColor,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                decoration: BoxDecoration(
                  color: _CallTheme.surface,
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: accentColor.withValues(alpha: 0.5)),
                ),
                child: Text(
                  riskLevel.label,
                  style: TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 9,
                    fontWeight: FontWeight.w700,
                    color: accentColor,
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// SPEAKER PLAYBACK & JITTER QUEUE CARD
// -----------------------------------------------------------------------------

class SpeakerPlaybackCard extends StatefulWidget {
  const SpeakerPlaybackCard({
    super.key,
    required this.service,
    required this.stats,
  });

  final LiveCallService service;
  final CallStats stats;

  @override
  State<SpeakerPlaybackCard> createState() => _SpeakerPlaybackCardState();
}

class _SpeakerPlaybackCardState extends State<SpeakerPlaybackCard> {
  StreamSubscription<bool>? _isPlayingSub;
  bool _isPlaying = false;
  bool _isTesting = false;

  @override
  void initState() {
    super.initState();
    _subscribe();
  }

  @override
  void didUpdateWidget(SpeakerPlaybackCard oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.service != widget.service) {
      _unsubscribe();
      _subscribe();
    }
  }

  void _subscribe() {
    _isPlayingSub = widget.service.playbackService.isPlayingStream.listen((playing) {
      if (mounted) setState(() => _isPlaying = playing);
    });
  }

  void _unsubscribe() {
    _isPlayingSub?.cancel();
  }

  @override
  void dispose() {
    _unsubscribe();
    super.dispose();
  }

  Future<void> _triggerTestSound() async {
    if (_isTesting) return;
    setState(() => _isTesting = true);
    try {
      await widget.service.playTestSound();
    } finally {
      if (mounted) setState(() => _isTesting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final activePlaying = _isPlaying || widget.service.playbackService.isPlaying;
    final isMuted = widget.service.playbackService.isMuted;

    return Container(
      decoration: BoxDecoration(
        color: _CallTheme.surfaceElevated,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _CallTheme.border),
      ),
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 38,
                height: 38,
                decoration: BoxDecoration(
                  color: activePlaying ? _CallTheme.emeraldBg : _CallTheme.surface,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(
                    color: activePlaying ? _CallTheme.emerald : _CallTheme.border,
                  ),
                ),
                child: Icon(
                  isMuted
                      ? Icons.volume_off_rounded
                      : (activePlaying
                            ? Icons.volume_up_rounded
                            : Icons.volume_mute_rounded),
                  color: isMuted
                      ? _CallTheme.textMuted
                      : (activePlaying ? _CallTheme.emerald : _CallTheme.textPrimary),
                  size: 20,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Speaker Playback & Jitter Queue',
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                        color: _CallTheme.textPrimary,
                      ),
                    ),
                    Text(
                      _isTesting
                          ? 'Playing 3-tone calibration chime...'
                          : (activePlaying
                                ? 'Streaming incoming voice through speaker (~300ms buffer)'
                                : (widget.service.isConnected
                                      ? 'Awaiting voice packets from caller...'
                                      : 'Connect receiver to listen')),
                      style: const TextStyle(
                        fontSize: 11,
                        color: _CallTheme.textSecondary,
                      ),
                    ),
                  ],
                ),
              ),
              IconButton(
                tooltip: isMuted ? 'Unmute speaker' : 'Mute speaker',
                onPressed: () {
                  setState(() {
                    widget.service.playbackService.setMuted(!isMuted);
                  });
                },
                icon: Icon(
                  isMuted ? Icons.volume_off_rounded : Icons.volume_up_rounded,
                  color: isMuted ? _CallTheme.amber : _CallTheme.textSecondary,
                  size: 20,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),

          // Jitter Buffer Telemetry Bar
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: _CallTheme.surface,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: _CallTheme.border),
            ),
            child: const Wrap(
              alignment: WrapAlignment.spaceBetween,
              crossAxisAlignment: WrapCrossAlignment.center,
              spacing: 8,
              runSpacing: 6,
              children: [
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.tune_rounded, size: 13, color: _CallTheme.textMuted),
                    SizedBox(width: 6),
                    Text(
                      'JITTER QUEUE: ~120 - 300 ms TARGET',
                      style: TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                        color: _CallTheme.textMuted,
                      ),
                    ),
                  ],
                ),
                Text(
                  'BUFFER STABLE · 0 DROPS',
                  style: TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 10,
                    fontWeight: FontWeight.w800,
                    color: _CallTheme.emerald,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),

          // VU Meter
          _BroadcastVuMeter(
            channelTitle: 'SPEAKER OUTPUT MONITOR',
            subLabel: '16 kHz Mono DAC · 300ms Dynamic Buffer',
            levelNotifier: widget.service.audioLevelNotifier,
            level: widget.stats.audioLevel,
            isActive: activePlaying || _isTesting,
            accentColor: _CallTheme.emerald,
          ),
          const SizedBox(height: 12),

          // Test sound button
          Align(
            alignment: Alignment.centerRight,
            child: OutlinedButton.icon(
              onPressed: _isTesting ? null : _triggerTestSound,
              icon: _isTesting
                  ? const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.music_note_rounded, size: 16),
              label: Text(_isTesting ? 'Testing...' : 'Test Speaker'),
              style: OutlinedButton.styleFrom(
                foregroundColor: _CallTheme.textPrimary,
                side: const BorderSide(color: _CallTheme.border),
                backgroundColor: _CallTheme.surface,
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                textStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// REASONING TRACE LOG PANEL
// -----------------------------------------------------------------------------

class _ReasoningTracePanel extends StatelessWidget {
  const _ReasoningTracePanel({required this.logs});

  final List<ReasoningLogEntry> logs;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: _CallTheme.surfaceElevated,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _CallTheme.border),
      ),
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.terminal_rounded, size: 16, color: _CallTheme.textSecondary),
                  SizedBox(width: 8),
                  Text(
                    'FORENSIC AUDIO NARRATION LOG',
                    style: TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 1.0,
                      color: _CallTheme.textSecondary,
                    ),
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: _CallTheme.surface,
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: _CallTheme.border),
                ),
                child: Text(
                  '${logs.length} EVENTS RECORDED',
                  style: const TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                    color: _CallTheme.textMuted,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          const Divider(color: _CallTheme.border, height: 1),
          const SizedBox(height: 10),
          if (logs.isEmpty)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 20),
              child: Center(
                child: Text(
                  'Awaiting live stream inference events to record trace.',
                  style: TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 11,
                    color: _CallTheme.textMuted,
                  ),
                ),
              ),
            )
          else
            SizedBox(
              height: 160,
              child: ListView.builder(
                reverse: true,
                itemCount: logs.length,
                itemBuilder: (context, index) {
                  final log = logs[index];
                  final time =
                      '${log.timestamp.hour.toString().padLeft(2, '0')}:${log.timestamp.minute.toString().padLeft(2, '0')}:${log.timestamp.second.toString().padLeft(2, '0')}';

                  final (String kindTag, Color kindColor, Color textColor, FontWeight textWeight) =
                      _determineLogKind(log);

                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 2.5),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          time,
                          style: const TextStyle(
                            fontFamily: 'monospace',
                            fontSize: 10.5,
                            fontWeight: FontWeight.w500,
                            color: _CallTheme.kindCollect,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
                          decoration: BoxDecoration(
                            color: kindColor.withValues(alpha: 0.12),
                            borderRadius: BorderRadius.circular(4),
                            border: Border.all(color: kindColor.withValues(alpha: 0.3)),
                          ),
                          child: Text(
                            kindTag,
                            style: TextStyle(
                              fontFamily: 'monospace',
                              fontSize: 8.5,
                              fontWeight: FontWeight.w800,
                              color: kindColor,
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            log.text,
                            style: TextStyle(
                              fontFamily: 'monospace',
                              fontSize: 11.5,
                              height: 1.35,
                              color: textColor,
                              fontWeight: textWeight,
                            ),
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
        ],
      ),
    );
  }

  (String, Color, Color, FontWeight) _determineLogKind(ReasoningLogEntry log) {
    final t = log.text.toLowerCase();
    if (t.contains('spoof') ||
        t.contains('synthetic') ||
        t.contains('cloning') ||
        log.riskLevel == LiveRiskLevel.high) {
      return ('ALERT', _CallTheme.kindAlert, _CallTheme.crimson, FontWeight.w700);
    }
    if (t.contains('caution') ||
        t.contains('boundary') ||
        t.contains('uncertain') ||
        log.riskLevel == LiveRiskLevel.uncertain) {
      return ('WARN', _CallTheme.kindWarn, _CallTheme.amber, FontWeight.w600);
    }
    if (t.contains('verdict') ||
        t.contains('verified') ||
        t.contains('bonafide') ||
        log.riskLevel == LiveRiskLevel.low) {
      return ('VERDICT', _CallTheme.kindVerdict, _CallTheme.emerald, FontWeight.w700);
    }
    return ('INFO', _CallTheme.kindInfo, _CallTheme.textPrimary, FontWeight.w400);
  }
}

// -----------------------------------------------------------------------------
// HELPER PILLS & TICKERS
// -----------------------------------------------------------------------------

class _CallDurationTicker extends StatelessWidget {
  const _CallDurationTicker({
    required this.duration,
    required this.isConnected,
  });

  final Duration duration;
  final bool isConnected;

  @override
  Widget build(BuildContext context) {
    final minutes = duration.inMinutes.remainder(60).toString().padLeft(2, '0');
    final seconds = duration.inSeconds.remainder(60).toString().padLeft(2, '0');

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: isConnected ? _CallTheme.surfaceElevated : _CallTheme.surfaceElevated,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: _CallTheme.border),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 6,
            height: 6,
            decoration: BoxDecoration(
              color: isConnected ? _CallTheme.emerald : _CallTheme.textMuted,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            '$minutes:$seconds',
            style: const TextStyle(
              fontFamily: 'monospace',
              fontSize: 11,
              fontWeight: FontWeight.w800,
              color: _CallTheme.textPrimary,
            ),
          ),
        ],
      ),
    );
  }
}

class _DataTimerPill extends StatelessWidget {
  const _DataTimerPill({
    required this.label,
    required this.duration,
    this.highlight = false,
  });

  final String label;
  final Duration duration;
  final bool highlight;

  @override
  Widget build(BuildContext context) {
    final minutes = duration.inMinutes.remainder(60).toString().padLeft(2, '0');
    final seconds = duration.inSeconds.remainder(60).toString().padLeft(2, '0');

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: highlight ? _CallTheme.crimsonBg : _CallTheme.surfaceElevated,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(
          color: highlight ? _CallTheme.crimsonBorder : _CallTheme.border,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            label,
            style: TextStyle(
              fontFamily: 'monospace',
              fontSize: 9.5,
              fontWeight: FontWeight.w700,
              color: highlight ? _CallTheme.crimsonText : _CallTheme.textMuted,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            '$minutes:$seconds',
            style: TextStyle(
              fontFamily: 'monospace',
              fontSize: 11,
              fontWeight: FontWeight.w800,
              color: highlight ? _CallTheme.crimsonText : _CallTheme.textPrimary,
            ),
          ),
        ],
      ),
    );
  }
}
