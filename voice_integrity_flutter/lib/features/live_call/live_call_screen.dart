import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../../core/models/live_call_models.dart';
import '../../core/services/live_call_service.dart';
import '../../core/theme/app_theme.dart';

/// The interactive Live Fraud Call Demo screen featuring:
/// 1. Caller Mode (Attacker): 16kHz mic capture, unmistakable "Talk as Teammate 3"
///    spoof toggle, glowing red visual feedback, timers, and VU meter.
/// 2. Receiver Mode (Victim): Speaker playback with ~120ms jitter queue,
///    real-time 4-band risk assessment (Green, Amber, Red, Gray) + 1.5s Amber transition state,
///    live probability meter, multi-expert breakdown, and reasoning trace.
class LiveCallScreen extends StatefulWidget {
  const LiveCallScreen({
    super.key,
    required this.service,
    this.initialMode = CallMode.caller,
  });

  final LiveCallService service;
  final CallMode initialMode;

  @override
  State<LiveCallScreen> createState() => _LiveCallScreenState();
}

class _LiveCallScreenState extends State<LiveCallScreen>
    with SingleTickerProviderStateMixin {
  late final LiveCallService _service;
  late final TextEditingController _hostController;
  late final TextEditingController _portController;
  late final AnimationController _pulseController;
  late final Animation<double> _pulseAnimation;

  StreamSubscription<CallConnectionState>? _connectionSub;
  StreamSubscription<LiveRiskAssessment>? _riskSub;
  StreamSubscription<CallStats>? _statsSub;
  StreamSubscription<List<ReasoningLogEntry>>? _logsSub;

  bool _showSettings = false;

  @override
  void initState() {
    super.initState();
    _service = widget.service;
    _service.setMode(widget.initialMode);

    _hostController = TextEditingController(text: _service.serverHost);
    _portController = TextEditingController(text: '${_service.serverPort}');

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
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
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _connectionSub?.cancel();
    _riskSub?.cancel();
    _statsSub?.cancel();
    _logsSub?.cancel();
    _hostController.dispose();
    _portController.dispose();
    super.dispose();
  }

  void _applySettings() {
    final host = _hostController.text.trim();
    final port = int.tryParse(_portController.text.trim()) ?? 8001;
    _service.setServerEndpoint(host, port);
    setState(() => _showSettings = false);
  }

  @override
  Widget build(BuildContext context) {
    final compact = MediaQuery.sizeOf(context).width < 768;
    final isCaller = _service.mode == CallMode.caller;

    return SingleChildScrollView(
      padding: EdgeInsets.symmetric(
        horizontal: compact ? 16 : 32,
        vertical: compact ? 16 : 28,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header eyebrow and title
          const _Eyebrow('LIVE FRAUD CALL DEFENSE CONSOLE'),
          const SizedBox(height: 10),
          _PageTitle(
            desktop: 'Two-party live call.\nReal-time clone defense.',
            mobile: 'Live call.\nClone defense.',
          ),
          const SizedBox(height: 12),
          Text(
            compact
                ? 'Test voice impersonation attacks live between Caller and Receiver.'
                : 'Deploy one client as Caller (Attacker) with 16kHz PCM streaming and spoof toggle, and another as Receiver (Victim) with speaker audio playback and real-time multi-expert integrity scoring.',
            style: const TextStyle(
              fontSize: 14,
              color: AppColors.mutedInk,
              height: 1.4,
            ),
          ),
          const SizedBox(height: 20),

          // Mode Selector
          _ModeSelectorPill(
            currentMode: _service.mode,
            onSelectMode: (mode) => setState(() => _service.setMode(mode)),
          ),
          const SizedBox(height: 16),

          // Connection and Server Configuration Bar
          _ConnectionControlCard(
            service: _service,
            showSettings: _showSettings,
            hostController: _hostController,
            portController: _portController,
            onToggleSettings: () =>
                setState(() => _showSettings = !_showSettings),
            onApplySettings: _applySettings,
            onConnect: () => _service.connect(),
            onDisconnect: () => _service.disconnect(),
            onToggleSimulation: (val) =>
                setState(() => _service.setSimulationMode(val)),
          ),
          const SizedBox(height: 20),

          // Main View depending on Caller vs Receiver Mode
          if (isCaller)
            _CallerModeView(service: _service, pulseAnimation: _pulseAnimation)
          else
            _ReceiverModeView(
              service: _service,
              pulseAnimation: _pulseAnimation,
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
  const _CallerModeView({required this.service, required this.pulseAnimation});

  final LiveCallService service;
  final Animation<double> pulseAnimation;

  @override
  Widget build(BuildContext context) {
    final isSpoof = service.isSpoofActive;
    final isConnected = service.isConnected;
    final stats = service.stats;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Spoof Hero Card with Glowing Red Animation when active
        AnimatedBuilder(
          animation: pulseAnimation,
          builder: (context, child) {
            final glowColor = isSpoof
                ? AppColors.vermilion.withValues(
                    alpha: 0.35 + (pulseAnimation.value * 0.4),
                  )
                : Colors.transparent;
            final borderColor = isSpoof
                ? AppColors.vermilion
                : (isConnected ? AppColors.moss : AppColors.line);

            return Container(
              decoration: BoxDecoration(
                color: isSpoof ? AppColors.vermilionSoft : AppColors.paper,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: borderColor,
                  width: isSpoof ? 2.5 : 1.5,
                ),
                boxShadow: isSpoof
                    ? [
                        BoxShadow(
                          color: glowColor,
                          blurRadius: 18 + (pulseAnimation.value * 12),
                          spreadRadius: 2 + (pulseAnimation.value * 3),
                        ),
                      ]
                    : [
                        const BoxShadow(
                          color: Color(0x0A000000),
                          blurRadius: 10,
                          offset: Offset(0, 3),
                        ),
                      ],
              ),
              padding: const EdgeInsets.all(22),
              child: child,
            );
          },
          child: Column(
            children: [
              // Status Badge & Timer
              Wrap(
                alignment: WrapAlignment.spaceBetween,
                crossAxisAlignment: WrapCrossAlignment.center,
                spacing: 8,
                runSpacing: 8,
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 5,
                    ),
                    decoration: BoxDecoration(
                      color: isSpoof
                          ? AppColors.vermilion
                          : (isConnected ? AppColors.moss : AppColors.mutedInk),
                      borderRadius: BorderRadius.circular(100),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          isSpoof
                              ? Icons.warning_rounded
                              : (isConnected
                                    ? Icons.mic_rounded
                                    : Icons.mic_off_rounded),
                          color: Colors.white,
                          size: 14,
                        ),
                        const SizedBox(width: 5),
                        Text(
                          isSpoof
                              ? 'AI SPOOF: ${service.spoofSpeaker.toUpperCase()}'
                              : (isConnected
                                    ? 'GENUINE STREAM'
                                    : 'CALLER STANDBY'),
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 11,
                            fontWeight: FontWeight.w800,
                            letterSpacing: 0.5,
                          ),
                        ),
                      ],
                    ),
                  ),
                  _LiveTimerPill(
                    label: isSpoof ? 'SPOOF TIME' : 'CALL TIME',
                    duration: isSpoof ? stats.spoofDuration : stats.duration,
                    highlight: isSpoof,
                  ),
                ],
              ),
              const SizedBox(height: 24),

              // Title and Explanatory Copy
              Text(
                isSpoof
                    ? 'Voice Cloning Attack Active'
                    : (isConnected
                          ? 'Authentic Voice Capturing'
                          : 'Connect Call to Transmit'),
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontFamily: 'Georgia',
                  fontSize: 22,
                  fontWeight: FontWeight.w700,
                  color: isSpoof ? AppColors.vermilion : AppColors.ink,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                isSpoof
                    ? 'Microphone stream is transformed into cloned vocal identity ("${service.spoofSpeaker}") using target acoustic profiles.'
                    : 'Transmitting natural human speech via 16kHz PCM 16-bit mono stream to the fraud evaluation pipeline.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 13,
                  color: isSpoof
                      ? AppColors.vermilion.withValues(alpha: 0.9)
                      : AppColors.mutedInk,
                  height: 1.35,
                ),
              ),
              const SizedBox(height: 26),

              // THE UNMISTAKABLE SPOOF TOGGLE BUTTON
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 14,
                    ),
                    backgroundColor: isSpoof
                        ? AppColors.ink
                        : AppColors.vermilion,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                    elevation: isSpoof ? 0 : 3,
                  ),
                  onPressed: isConnected
                      ? () => service.toggleSpoof()
                      : () {
                          service.connect().then((_) {
                            service.toggleSpoof();
                          });
                        },
                  icon: Icon(
                    isSpoof
                        ? Icons.undo_rounded
                        : Icons.record_voice_over_rounded,
                    size: 22,
                  ),
                  label: FittedBox(
                    fit: BoxFit.scaleDown,
                    child: Text(
                      isSpoof
                          ? 'Stop Spoof (Revert to Genuine)'
                          : 'Talk as Teammate 3 (Spoof)',
                      textAlign: TextAlign.center,
                      maxLines: 1,
                      style: const TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 0.3,
                      ),
                    ),
                  ),
                ),
              ),
              if (!isConnected) ...[
                const SizedBox(height: 8),
                const Text(
                  'Pressing toggle will automatically initialize connection.',
                  style: TextStyle(fontSize: 11, color: AppColors.mutedInk),
                ),
              ],
            ],
          ),
        ),
        const SizedBox(height: 18),

        // Microphone Audio VU Level Meter
        _AudioLevelMeterCard(
          title: 'Microphone Audio Stream Level',
          subtitle: '16kHz PCM 16-bit Mono Capture',
          level: stats.audioLevel,
          isActive: isConnected,
          accentColor: isSpoof ? AppColors.vermilion : AppColors.moss,
        ),
        const SizedBox(height: 18),

        // Caller Stream Telemetry
        _TelemetryCard(stats: stats, isCaller: true, spoofActive: isSpoof),
      ],
    );
  }
}

// -----------------------------------------------------------------------------
// RECEIVER MODE (VICTIM) VIEW
// -----------------------------------------------------------------------------

class _ReceiverModeView extends StatelessWidget {
  const _ReceiverModeView({
    required this.service,
    required this.pulseAnimation,
  });

  final LiveCallService service;
  final Animation<double> pulseAnimation;

  @override
  Widget build(BuildContext context) {
    final assessment = service.currentAssessment;
    final level = assessment.riskLevel;
    final isTransition = level == LiveRiskLevel.transitioning;
    final stats = service.stats;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Real-Time Risk Assessment Banner Card
        AnimatedContainer(
          duration: const Duration(milliseconds: 300),
          padding: const EdgeInsets.all(22),
          decoration: BoxDecoration(
            color: level.backgroundColor,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: level.accentColor,
              width: isTransition ? 2.5 : 1.5,
            ),
            boxShadow: isTransition
                ? [
                    BoxShadow(
                      color: AppColors.amber.withValues(alpha: 0.3),
                      blurRadius: 16,
                      spreadRadius: 2,
                    ),
                  ]
                : const [
                    BoxShadow(
                      color: Color(0x08000000),
                      blurRadius: 8,
                      offset: Offset(0, 2),
                    ),
                  ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Status Icon or Pulsing Spinner for Transition
                  if (isTransition)
                    AnimatedBuilder(
                      animation: pulseAnimation,
                      builder: (context, _) {
                        return Container(
                          width: 48,
                          height: 48,
                          decoration: BoxDecoration(
                            color: AppColors.amber,
                            borderRadius: BorderRadius.circular(14),
                          ),
                          child: const Icon(
                            Icons.hourglass_top_rounded,
                            color: Colors.white,
                            size: 26,
                          ),
                        );
                      },
                    )
                  else
                    Container(
                      width: 48,
                      height: 48,
                      decoration: BoxDecoration(
                        color: level.accentColor,
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: Icon(level.icon, color: Colors.white, size: 26),
                    ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          level.label.toUpperCase(),
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w800,
                            letterSpacing: 1.1,
                            color: level.accentColor,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          level.description,
                          style: const TextStyle(
                            fontFamily: 'Georgia',
                            fontSize: 18,
                            fontWeight: FontWeight.w700,
                            color: AppColors.ink,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              const Divider(color: AppColors.line, height: 1),
              const SizedBox(height: 14),

              // Recommended Action
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(
                    Icons.directions_run_rounded,
                    size: 18,
                    color: level.accentColor,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Action: ${assessment.recommendedAction}',
                      style: const TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: AppColors.ink,
                        height: 1.35,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
        const SizedBox(height: 18),

        // Live Calibrated Probability Meter (0.00 - 1.00)
        _LiveProbabilityMeterCard(assessment: assessment),
        const SizedBox(height: 18),

        // Speaker Playback Controls Card
        _SpeakerPlaybackCard(service: service, stats: stats),
        const SizedBox(height: 18),

        // Multi-Expert Evidence Breakdown
        _ExpertEvidenceRow(assessment: assessment),
        const SizedBox(height: 18),

        // Live Reasoning Trace Logs
        _ReasoningTraceCard(logs: service.reasoningLogs),
      ],
    );
  }
}

// -----------------------------------------------------------------------------
// LIVE PROBABILITY METER WIDGET
// -----------------------------------------------------------------------------

class _LiveProbabilityMeterCard extends StatelessWidget {
  const _LiveProbabilityMeterCard({required this.assessment});

  final LiveRiskAssessment assessment;

  @override
  Widget build(BuildContext context) {
    final prob = assessment.probability;
    final percent = prob != null ? (prob * 100).toStringAsFixed(1) : '--';

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'LIVE PROBABILITY GAUGE',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1.0,
                        color: AppColors.mutedInk,
                      ),
                    ),
                    SizedBox(height: 2),
                    Text(
                      'Calibrated Synthetic Score',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                        color: AppColors.ink,
                      ),
                    ),
                  ],
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: assessment.riskLevel.backgroundColor,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: assessment.riskLevel.accentColor),
                  ),
                  child: Text(
                    '$percent%',
                    style: TextStyle(
                      fontFamily: 'Georgia',
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                      color: assessment.riskLevel.accentColor,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 18),

            // Continuous Visual Color Bar with Indicator
            LayoutBuilder(
              builder: (context, constraints) {
                final barWidth = constraints.maxWidth;
                final pointerX = prob != null
                    ? (prob.clamp(0.0, 1.0) * barWidth)
                    : (0.0);

                return Column(
                  children: [
                    // The multi-color spectrum bar
                    Container(
                      height: 16,
                      width: barWidth,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(8),
                        gradient: const LinearGradient(
                          colors: [
                            AppColors.moss,
                            AppColors.amber,
                            AppColors.vermilion,
                          ],
                          stops: [0.35, 0.65, 1.0],
                        ),
                      ),
                    ),
                    const SizedBox(height: 6),

                    // Pointer arrow and threshold guides
                    Stack(
                      children: [
                        SizedBox(height: 24, width: barWidth),
                        // Indicator marker
                        if (prob != null)
                          Positioned(
                            left: (pointerX - 10).clamp(0.0, barWidth - 20),
                            child: const Column(
                              children: [
                                Icon(
                                  Icons.arrow_drop_up_rounded,
                                  size: 20,
                                  color: AppColors.ink,
                                ),
                              ],
                            ),
                          ),
                      ],
                    ),
                  ],
                );
              },
            ),

            // Threshold Legend Labels
            const Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Text(
                    '0.00 · Low Risk\n(Likely Human)',
                    style: TextStyle(
                      fontSize: 11,
                      color: AppColors.moss,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
                Expanded(
                  child: Text(
                    '0.35 - 0.65 · Uncertain\n(Secondary Verify)',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 11,
                      color: AppColors.amber,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
                Expanded(
                  child: Text(
                    '≥ 0.65 · High Risk\n(Synthetic Clone)',
                    textAlign: TextAlign.right,
                    style: TextStyle(
                      fontSize: 11,
                      color: AppColors.vermilion,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// SPEAKER PLAYBACK CONTROLS CARD
// -----------------------------------------------------------------------------

class _SpeakerPlaybackCard extends StatefulWidget {
  const _SpeakerPlaybackCard({required this.service, required this.stats});

  final LiveCallService service;
  final CallStats stats;

  @override
  State<_SpeakerPlaybackCard> createState() => _SpeakerPlaybackCardState();
}

class _SpeakerPlaybackCardState extends State<_SpeakerPlaybackCard> {
  @override
  Widget build(BuildContext context) {
    final playback = widget.service.playbackService;
    final isMuted = playback.isMuted;
    final isPlaying = playback.isPlaying;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 38,
                  height: 38,
                  decoration: BoxDecoration(
                    color: isPlaying ? AppColors.mossSoft : AppColors.paper,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                      color: isPlaying ? AppColors.moss : AppColors.line,
                    ),
                  ),
                  child: Icon(
                    isMuted
                        ? Icons.volume_off_rounded
                        : (isPlaying
                              ? Icons.volume_up_rounded
                              : Icons.volume_mute_rounded),
                    color: isMuted
                        ? AppColors.mutedInk
                        : (isPlaying ? AppColors.moss : AppColors.ink),
                    size: 20,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Device Speaker Playback',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          color: AppColors.ink,
                        ),
                      ),
                      Text(
                        isPlaying
                            ? 'Streaming incoming voice through speaker (~120ms jitter queue)'
                            : (widget.service.isConnected
                                  ? 'Waiting for voice audio packets...'
                                  : 'Connect receiver to listen'),
                        style: const TextStyle(
                          fontSize: 12,
                          color: AppColors.mutedInk,
                        ),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  tooltip: isMuted ? 'Unmute speaker' : 'Mute speaker',
                  onPressed: () {
                    playback.toggleMute();
                    setState(() {});
                  },
                  icon: Icon(
                    isMuted
                        ? Icons.volume_off_rounded
                        : Icons.volume_up_rounded,
                    color: isMuted ? AppColors.vermilion : AppColors.moss,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 14),

            // Speaker VU Level Meter
            _AudioLevelMeterCard(
              title: 'Speaker Acoustic Output',
              subtitle: '16kHz Audio Stream Playback',
              level: widget.stats.audioLevel,
              isActive: isPlaying,
              accentColor: AppColors.moss,
            ),
          ],
        ),
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// MULTI-EXPERT EVIDENCE BREAKDOWN
// -----------------------------------------------------------------------------

class _ExpertEvidenceRow extends StatelessWidget {
  const _ExpertEvidenceRow({required this.assessment});

  final LiveRiskAssessment assessment;

  @override
  Widget build(BuildContext context) {
    final experts = assessment.expertScores;
    final compact = MediaQuery.sizeOf(context).width < 768;

    // Standard list of the 3 runtime experts
    final items = [
      ('wavlm', 'Expert 1 · WavLM Base+', 0.20, experts['wavlm']),
      ('hybrid', 'Expert 2 · LFCC-LCNN', 0.20, experts['hybrid']),
      ('ssl', 'Expert 3 · TakHemlata SSL', 0.60, experts['ssl']),
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'INDEPENDENT MODEL EVIDENCE',
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w800,
            letterSpacing: 1.0,
            color: AppColors.mutedInk,
          ),
        ),
        const SizedBox(height: 8),
        if (compact)
          Column(
            children: items.map((item) {
              return Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: _ExpertMiniCard(
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
                  child: _ExpertMiniCard(
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

class _ExpertMiniCard extends StatelessWidget {
  const _ExpertMiniCard({
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
    final probText = prob != null
        ? '${(prob * 100).toStringAsFixed(1)}%'
        : '--';
    final riskLevel = score?.riskLevel ?? LiveRiskLevel.unavailable;

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.paper,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.line),
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
                    color: AppColors.ink,
                  ),
                ),
              ),
              Text(
                '${(weight * 100).toInt()}% wt',
                style: const TextStyle(fontSize: 10, color: AppColors.mutedInk),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                probText,
                style: TextStyle(
                  fontFamily: 'Georgia',
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  color: riskLevel.accentColor,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: riskLevel.backgroundColor,
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  riskLevel.label,
                  style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                    color: riskLevel.accentColor,
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
// LIVE REASONING TRACE CARD
// -----------------------------------------------------------------------------

class _ReasoningTraceCard extends StatelessWidget {
  const _ReasoningTraceCard({required this.logs});

  final List<ReasoningLogEntry> logs;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Row(
                  children: [
                    Icon(
                      Icons.history_edu_rounded,
                      size: 18,
                      color: AppColors.slate,
                    ),
                    SizedBox(width: 8),
                    Text(
                      'REAL-TIME REASONING TRACE',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1.0,
                        color: AppColors.mutedInk,
                      ),
                    ),
                  ],
                ),
                Text(
                  '${logs.length} events logged',
                  style: const TextStyle(
                    fontSize: 11,
                    color: AppColors.mutedInk,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            const Divider(color: AppColors.line, height: 1),
            const SizedBox(height: 10),

            if (logs.isEmpty)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 24),
                child: Center(
                  child: Text(
                    'No reasoning events received yet. Connect to stream inference.',
                    style: TextStyle(fontSize: 12, color: AppColors.mutedInk),
                  ),
                ),
              )
            else
              ConstrainedBox(
                constraints: const BoxConstraints(maxHeight: 220),
                child: ListView.separated(
                  shrinkWrap: true,
                  itemCount: logs.length,
                  separatorBuilder: (context, index) =>
                      const Divider(color: AppColors.line, height: 12),
                  itemBuilder: (context, index) {
                    final log = logs[index];
                    final time =
                        '${log.timestamp.hour.toString().padLeft(2, '0')}:${log.timestamp.minute.toString().padLeft(2, '0')}:${log.timestamp.second.toString().padLeft(2, '0')}';

                    return Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          time,
                          style: const TextStyle(
                            fontFamily: 'Courier',
                            fontSize: 11,
                            color: AppColors.mutedInk,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Container(
                          width: 8,
                          height: 8,
                          margin: const EdgeInsets.only(top: 4),
                          decoration: BoxDecoration(
                            color: log.riskLevel.accentColor,
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            log.text,
                            style: const TextStyle(
                              fontSize: 12,
                              height: 1.3,
                              color: AppColors.ink,
                            ),
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// AUDIO LEVEL VU METER CARD
// -----------------------------------------------------------------------------

class _AudioLevelMeterCard extends StatelessWidget {
  const _AudioLevelMeterCard({
    required this.title,
    required this.subtitle,
    required this.level,
    required this.isActive,
    required this.accentColor,
  });

  final String title;
  final String subtitle;
  final double level;
  final bool isActive;
  final Color accentColor;

  @override
  Widget build(BuildContext context) {
    const barCount = 18;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.paper,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.line),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: AppColors.ink,
                ),
              ),
              Text(
                isActive ? '${(level * 100).toInt()}% dB RMS' : 'Standby',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: isActive ? accentColor : AppColors.mutedInk,
                ),
              ),
            ],
          ),
          const SizedBox(height: 2),
          Text(
            subtitle,
            style: const TextStyle(fontSize: 11, color: AppColors.mutedInk),
          ),
          const SizedBox(height: 12),

          // Equalizer VU Bars
          LayoutBuilder(
            builder: (context, constraints) {
              return Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: List.generate(barCount, (index) {
                  final threshold = (index + 1) / barCount;
                  final isLit = isActive && level >= threshold;

                  // Color gradient from green to amber to red as level climbs
                  final barColor = index > 14
                      ? AppColors.vermilion
                      : (index > 9 ? AppColors.amber : AppColors.moss);

                  final height = 6.0 + (index * 1.1);

                  return AnimatedContainer(
                    duration: const Duration(milliseconds: 60),
                    width: math.max(4.0, (constraints.maxWidth / barCount) - 4),
                    height: height,
                    decoration: BoxDecoration(
                      color: isLit
                          ? barColor
                          : AppColors.line.withValues(alpha: 0.5),
                      borderRadius: BorderRadius.circular(3),
                    ),
                  );
                }),
              );
            },
          ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// TELEMETRY CARD
// -----------------------------------------------------------------------------

class _TelemetryCard extends StatelessWidget {
  const _TelemetryCard({
    required this.stats,
    required this.isCaller,
    required this.spoofActive,
  });

  final CallStats stats;
  final bool isCaller;
  final bool spoofActive;

  @override
  Widget build(BuildContext context) {
    final compact = MediaQuery.sizeOf(context).width < 768;

    final items = [
      ('Audio Stream', '16 kHz Mono PCM'),
      ('Packets Sent', '${stats.packetsSent} frames'),
      ('Packets Rcvd', '${stats.packetsReceived} frames'),
      (
        'Transferred',
        '${(stats.bytesTransferred / 1024).toStringAsFixed(1)} KB',
      ),
    ];

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'TRANSMISSION TELEMETRY',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w800,
                letterSpacing: 1.0,
                color: AppColors.mutedInk,
              ),
            ),
            const SizedBox(height: 14),
            Wrap(
              spacing: compact ? 12 : 24,
              runSpacing: 12,
              children: items.map((item) {
                return SizedBox(
                  width: compact ? 130 : 150,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        item.$1,
                        style: const TextStyle(
                          fontSize: 11,
                          color: AppColors.mutedInk,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        item.$2,
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          color: AppColors.ink,
                        ),
                      ),
                    ],
                  ),
                );
              }).toList(),
            ),
          ],
        ),
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// CONNECTION CONTROL & SETTINGS CARD
// -----------------------------------------------------------------------------

class _ConnectionControlCard extends StatelessWidget {
  const _ConnectionControlCard({
    required this.service,
    required this.showSettings,
    required this.hostController,
    required this.portController,
    required this.onToggleSettings,
    required this.onApplySettings,
    required this.onConnect,
    required this.onDisconnect,
    required this.onToggleSimulation,
  });

  final LiveCallService service;
  final bool showSettings;
  final TextEditingController hostController;
  final TextEditingController portController;
  final VoidCallback onToggleSettings;
  final VoidCallback onApplySettings;
  final VoidCallback onConnect;
  final VoidCallback onDisconnect;
  final ValueChanged<bool> onToggleSimulation;

  @override
  Widget build(BuildContext context) {
    final state = service.connectionState;
    final isConnected = service.isConnected;
    final isSimulation = service.useSimulationMode;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          children: [
            Row(
              children: [
                Container(
                  width: 12,
                  height: 12,
                  decoration: BoxDecoration(
                    color: isConnected
                        ? AppColors.moss
                        : (state == CallConnectionState.error
                              ? AppColors.vermilion
                              : AppColors.mutedInk),
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        isSimulation
                            ? 'Simulation Demo Mode Active'
                            : (isConnected
                                  ? 'Connected to ws://${service.serverHost}:${service.serverPort}'
                                  : state.label),
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          color: AppColors.ink,
                        ),
                      ),
                      Text(
                        isSimulation
                            ? 'Running self-contained realistic demo for offline judging'
                            : 'Endpoint: /ws/${service.mode == CallMode.caller ? 'caller' : 'receiver'}',
                        style: const TextStyle(
                          fontSize: 11,
                          color: AppColors.mutedInk,
                        ),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  tooltip: 'Server configuration',
                  onPressed: onToggleSettings,
                  icon: Icon(
                    showSettings ? Icons.close_rounded : Icons.settings_rounded,
                    size: 20,
                  ),
                ),
                const SizedBox(width: 4),
                if (isConnected)
                  OutlinedButton.icon(
                    onPressed: onDisconnect,
                    icon: const Icon(
                      Icons.call_end_rounded,
                      color: AppColors.vermilion,
                      size: 18,
                    ),
                    label: const Text(
                      'End Call',
                      style: TextStyle(color: AppColors.vermilion),
                    ),
                  )
                else
                  FilledButton.icon(
                    onPressed: onConnect,
                    icon: const Icon(Icons.call_rounded, size: 18),
                    label: Text(isSimulation ? 'Start Demo' : 'Connect Call'),
                  ),
              ],
            ),

            // Expandable settings view
            if (showSettings) ...[
              const SizedBox(height: 16),
              const Divider(color: AppColors.line, height: 1),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    flex: 3,
                    child: TextField(
                      controller: hostController,
                      decoration: const InputDecoration(
                        labelText: 'WebSocket Host',
                        hintText: '127.0.0.1 or LAN IP',
                        border: OutlineInputBorder(),
                        isDense: true,
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    flex: 2,
                    child: TextField(
                      controller: portController,
                      keyboardType: TextInputType.number,
                      decoration: const InputDecoration(
                        labelText: 'Port',
                        hintText: '8001',
                        border: OutlineInputBorder(),
                        isDense: true,
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  FilledButton(
                    onPressed: onApplySettings,
                    child: const Text('Save'),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text(
                  'Offline Presentation Simulation',
                  style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700),
                ),
                subtitle: const Text(
                  'Run presentation simulation without requiring a live python server on port 8001.',
                  style: TextStyle(fontSize: 11, color: AppColors.mutedInk),
                ),
                value: service.useSimulationMode,
                onChanged: onToggleSimulation,
              ),
            ],
          ],
        ),
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// MODE SELECTOR PILL
// -----------------------------------------------------------------------------

class _ModeSelectorPill extends StatelessWidget {
  const _ModeSelectorPill({
    required this.currentMode,
    required this.onSelectMode,
  });

  final CallMode currentMode;
  final ValueChanged<CallMode> onSelectMode;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(5),
      decoration: BoxDecoration(
        color: AppColors.paper,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.line),
      ),
      child: Row(
        children: CallMode.values.map((mode) {
          final isSelected = mode == currentMode;
          return Expanded(
            child: GestureDetector(
              onTap: () => onSelectMode(mode),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                padding: const EdgeInsets.symmetric(vertical: 12),
                decoration: BoxDecoration(
                  color: isSelected ? AppColors.ink : Colors.transparent,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      mode.icon,
                      size: 18,
                      color: isSelected ? Colors.white : AppColors.mutedInk,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      mode.title,
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w800,
                        color: isSelected ? Colors.white : AppColors.ink,
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
// TIMER PILL
// -----------------------------------------------------------------------------

class _LiveTimerPill extends StatelessWidget {
  const _LiveTimerPill({
    required this.label,
    required this.duration,
    required this.highlight,
  });

  final String label;
  final Duration duration;
  final bool highlight;

  @override
  Widget build(BuildContext context) {
    final minutes = duration.inMinutes.toString().padLeft(2, '0');
    final seconds = (duration.inSeconds % 60).toString().padLeft(2, '0');

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: highlight
            ? AppColors.vermilion.withValues(alpha: 0.15)
            : AppColors.canvas,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: highlight ? AppColors.vermilion : AppColors.line,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.timer_outlined,
            size: 14,
            color: highlight ? AppColors.vermilion : AppColors.mutedInk,
          ),
          const SizedBox(width: 5),
          Text(
            '$label $minutes:$seconds',
            style: TextStyle(
              fontFamily: 'Courier',
              fontSize: 11,
              fontWeight: FontWeight.w800,
              color: highlight ? AppColors.vermilion : AppColors.ink,
            ),
          ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// TYPOGRAPHY HELPERS (CONSISTENT WITH DASHBOARD SHELL)
// -----------------------------------------------------------------------------

class _Eyebrow extends StatelessWidget {
  const _Eyebrow(this.text);
  final String text;

  @override
  Widget build(BuildContext context) => Text(
    text,
    style: const TextStyle(
      fontSize: 11,
      fontWeight: FontWeight.w800,
      letterSpacing: 1.6,
      color: AppColors.mutedInk,
    ),
  );
}

class _PageTitle extends StatelessWidget {
  const _PageTitle({required this.desktop, required this.mobile});
  final String desktop;
  final String mobile;

  @override
  Widget build(BuildContext context) {
    final phone = MediaQuery.sizeOf(context).width < 600;
    return Text(
      phone ? mobile : desktop,
      style: TextStyle(
        fontFamily: 'Georgia',
        fontSize: phone ? 28 : 40,
        fontWeight: FontWeight.w700,
        height: 1.05,
        color: AppColors.ink,
        letterSpacing: -0.5,
      ),
    );
  }
}
