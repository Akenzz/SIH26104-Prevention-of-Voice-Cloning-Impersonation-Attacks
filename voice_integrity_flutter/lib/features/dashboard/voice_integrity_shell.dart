import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../../core/models/voice_models.dart';
import '../../core/services/live_call_service.dart';
import '../../core/services/voice_integrity_api.dart';
import '../../core/theme/app_theme.dart';
import '../live_call/live_call_screen.dart';

enum _Destination { overview, analysis, liveCall, live, models }

class VoiceIntegrityShell extends StatefulWidget {
  const VoiceIntegrityShell({super.key});

  @override
  State<VoiceIntegrityShell> createState() => _VoiceIntegrityShellState();
}

class _VoiceIntegrityShellState extends State<VoiceIntegrityShell> {
  final _endpointController = TextEditingController(
    text: VoiceIntegrityApi.defaultEndpoint,
  );
  final _liveCallService = LiveCallService();
  _Destination _destination = _Destination.overview;
  AnalysisReport _report = AnalysisReport.demo();
  BackendHealth? _health;
  bool _checkingHealth = false;
  String? _healthError;

  VoiceIntegrityApi get _api => VoiceIntegrityApi(_endpointController.text);

  @override
  void initState() {
    super.initState();
    _checkHealth();
  }

  @override
  void dispose() {
    _liveCallService.dispose();
    _endpointController.dispose();
    super.dispose();
  }

  Future<void> _checkHealth() async {
    setState(() {
      _checkingHealth = true;
      _healthError = null;
    });
    try {
      final health = await _api.fetchHealth();
      if (!mounted) return;
      setState(() => _health = health);
    } catch (error) {
      if (!mounted) return;
      setState(() => _healthError = '$error');
    } finally {
      if (mounted) setState(() => _checkingHealth = false);
    }
  }

  void _setReport(AnalysisReport report) => setState(() => _report = report);

  @override
  Widget build(BuildContext context) {
    final isWide = MediaQuery.sizeOf(context).width >= 1024;
    final page = switch (_destination) {
      _Destination.overview => _OverviewPage(
        report: _report,
        health: _health,
        checkingHealth: _checkingHealth,
        healthError: _healthError,
        onCheckHealth: _checkHealth,
        onNavigate: (destination) => setState(() => _destination = destination),
      ),
      _Destination.analysis => _FileAnalysisPage(
        api: _api,
        report: _report,
        onReport: _setReport,
      ),
      _Destination.liveCall => LiveCallScreen(service: _liveCallService),
      _Destination.live => _LiveMonitorPage(
        report: _report,
        onReport: _setReport,
      ),
      _Destination.models => _ModelReviewPage(
        endpointController: _endpointController,
        health: _health,
        checkingHealth: _checkingHealth,
        healthError: _healthError,
        onCheckHealth: _checkHealth,
      ),
    };

    final navigation = _Navigation(
      selected: _destination,
      expanded: isWide,
      onSelect: (destination) => setState(() => _destination = destination),
    );

    return Scaffold(
      body: SafeArea(
        child: Row(
          children: [
            if (isWide) navigation,
            Expanded(
              child: Column(
                children: [
                  _AppHeader(
                    report: _report,
                    isBackendConnected: _health != null,
                    checkingHealth: _checkingHealth,
                    onOpenModels: () =>
                        setState(() => _destination = _Destination.models),
                  ),
                  Expanded(child: page),
                ],
              ),
            ),
          ],
        ),
      ),
      bottomNavigationBar: isWide ? null : navigation,
    );
  }
}

class _AppHeader extends StatelessWidget {
  const _AppHeader({
    required this.report,
    required this.isBackendConnected,
    required this.checkingHealth,
    required this.onOpenModels,
  });

  final AnalysisReport report;
  final bool isBackendConnected;
  final bool checkingHealth;
  final VoidCallback onOpenModels;

  @override
  Widget build(BuildContext context) {
    final phone = MediaQuery.sizeOf(context).width < 600;
    return Container(
      height: phone ? 64 : 74,
      padding: EdgeInsets.symmetric(horizontal: phone ? 16 : 20),
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: AppColors.line)),
      ),
      child: Row(
        children: [
          const _ShieldMark(compact: true),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              phone ? 'Voice Integrity' : 'VOICE INTEGRITY',
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                fontWeight: FontWeight.w800,
                letterSpacing: phone ? .15 : 1.25,
              ),
            ),
          ),
          _ConnectionPill(
            connected: isBackendConnected,
            loading: checkingHealth,
            compact: phone,
          ),
          const SizedBox(width: 8),
          IconButton(
            tooltip: 'Model and system review',
            onPressed: onOpenModels,
            icon: const Icon(Icons.tune_rounded),
          ),
        ],
      ),
    );
  }
}

class _Navigation extends StatelessWidget {
  const _Navigation({
    required this.selected,
    required this.expanded,
    required this.onSelect,
  });

  final _Destination selected;
  final bool expanded;
  final ValueChanged<_Destination> onSelect;

  static const _items = [
    (_Destination.overview, 'Overview', Icons.grid_view_rounded),
    (_Destination.analysis, 'Review audio', Icons.audio_file_rounded),
    (_Destination.liveCall, 'Live call', Icons.phone_in_talk_rounded),
    (_Destination.live, 'Guided demo', Icons.graphic_eq_rounded),
    (_Destination.models, 'Models', Icons.account_tree_outlined),
  ];

  @override
  Widget build(BuildContext context) {
    if (!expanded) {
      final compact = MediaQuery.sizeOf(context).width < 375;
      return NavigationBar(
        height: 72,
        labelBehavior: compact
            ? NavigationDestinationLabelBehavior.onlyShowSelected
            : NavigationDestinationLabelBehavior.alwaysShow,
        selectedIndex: _items.indexWhere((item) => item.$1 == selected),
        onDestinationSelected: (index) => onSelect(_items[index].$1),
        destinations: _items
            .map(
              (item) =>
                  NavigationDestination(icon: Icon(item.$3), label: item.$2),
            )
            .toList(),
      );
    }
    return Container(
      width: 236,
      padding: const EdgeInsets.fromLTRB(16, 20, 16, 16),
      decoration: const BoxDecoration(
        color: AppColors.paper,
        border: Border(right: BorderSide(color: AppColors.line)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.fromLTRB(8, 8, 8, 30),
            child: Row(
              children: [
                _ShieldMark(),
                SizedBox(width: 12),
                Expanded(
                  child: Text(
                    'Voice\nIntegrity',
                    style: TextStyle(
                      fontFamily: 'Georgia',
                      fontWeight: FontWeight.w700,
                      fontSize: 19,
                      height: .95,
                    ),
                  ),
                ),
              ],
            ),
          ),
          ..._items.map(
            (item) => Padding(
              padding: const EdgeInsets.only(bottom: 5),
              child: _SideNavigationItem(
                selected: item.$1 == selected,
                label: item.$2,
                icon: item.$3,
                onPressed: () => onSelect(item.$1),
              ),
            ),
          ),
          const Spacer(),
          const _SideNote(),
        ],
      ),
    );
  }
}

class _SideNavigationItem extends StatelessWidget {
  const _SideNavigationItem({
    required this.selected,
    required this.label,
    required this.icon,
    required this.onPressed,
  });

  final bool selected;
  final String label;
  final IconData icon;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: selected ? AppColors.mossSoft : Colors.transparent,
      borderRadius: BorderRadius.circular(13),
      child: InkWell(
        onTap: onPressed,
        borderRadius: BorderRadius.circular(13),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 12),
          child: Row(
            children: [
              Icon(
                icon,
                size: 20,
                color: selected ? AppColors.moss : AppColors.mutedInk,
              ),
              const SizedBox(width: 12),
              Text(
                label,
                style: TextStyle(
                  fontWeight: FontWeight.w700,
                  color: selected ? AppColors.moss : AppColors.ink,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _SideNote extends StatelessWidget {
  const _SideNote();

  @override
  Widget build(BuildContext context) {
    return const Padding(
      padding: EdgeInsets.all(8),
      child: Text(
        'Decision support\n—not identity proof.',
        style: TextStyle(color: AppColors.mutedInk, height: 1.4, fontSize: 12),
      ),
    );
  }
}

class _OverviewPage extends StatelessWidget {
  const _OverviewPage({
    required this.report,
    required this.health,
    required this.checkingHealth,
    required this.healthError,
    required this.onCheckHealth,
    required this.onNavigate,
  });

  final AnalysisReport report;
  final BackendHealth? health;
  final bool checkingHealth;
  final String? healthError;
  final VoidCallback onCheckHealth;
  final ValueChanged<_Destination> onNavigate;

  @override
  Widget build(BuildContext context) {
    final compact = MediaQuery.sizeOf(context).width < 768;
    final system = _SystemStatusCard(
      health: health,
      checking: checkingHealth,
      healthError: healthError,
      onCheck: onCheckHealth,
    );
    return _PageScroll(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const _Eyebrow('VOICE RISK REVIEW CONSOLE'),
          const SizedBox(height: 11),
          const _PageTitle(
            desktop: 'Know when to\nverify the caller.',
            mobile: 'Verify the\nvoice first.',
          ),
          const SizedBox(height: 12),
          if (!compact)
            ConstrainedBox(
              constraints: BoxConstraints(maxWidth: 650),
              child: Text(
                'A live decision aid for spotting synthetic or cloned speech. It tells you what the audio suggests—and what to do next.',
              ),
            ),
          SizedBox(height: compact ? 20 : 24),
          if (compact) ...[
            _VerdictCard(report: report),
            const SizedBox(height: 16),
            _MobileActionStack(onNavigate: onNavigate),
            const SizedBox(height: 16),
            system,
            const SizedBox(height: 16),
            _MobileEvidenceDisclosure(report: report),
          ] else ...[
            Wrap(
              spacing: 12,
              runSpacing: 12,
              children: [
                FilledButton.icon(
                  onPressed: () => onNavigate(_Destination.analysis),
                  icon: const Icon(Icons.upload_file_outlined),
                  label: const Text('Review an audio file'),
                ),
                FilledButton.tonalIcon(
                  onPressed: () => onNavigate(_Destination.liveCall),
                  icon: const Icon(Icons.phone_in_talk_rounded),
                  label: const Text('Live fraud call demo'),
                ),
                OutlinedButton.icon(
                  onPressed: () => onNavigate(_Destination.live),
                  icon: const Icon(Icons.play_circle_outline_rounded),
                  label: const Text('Run judge walkthrough'),
                ),
              ],
            ),
            const SizedBox(height: 28),
            LayoutBuilder(
              builder: (context, constraints) {
                final wide = constraints.maxWidth >= 920;
                final verdict = _VerdictCard(report: report);
                return wide
                    ? Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Expanded(flex: 13, child: verdict),
                          const SizedBox(width: 18),
                          Expanded(flex: 9, child: system),
                        ],
                      )
                    : Column(
                        children: [verdict, const SizedBox(height: 18), system],
                      );
              },
            ),
            const SizedBox(height: 18),
            _EvidenceStrip(report: report),
            const SizedBox(height: 18),
            _TimelineCard(report: report),
          ],
          const SizedBox(height: 18),
          const _HonestyNote(),
        ],
      ),
    );
  }
}

class _PageScroll extends StatelessWidget {
  const _PageScroll({required this.child});
  final Widget child;

  @override
  Widget build(BuildContext context) {
    final compact = MediaQuery.sizeOf(context).width < 768;
    return Scrollbar(
      child: SingleChildScrollView(
        padding: EdgeInsets.fromLTRB(
          compact ? 16 : 20,
          compact ? 20 : 28,
          compact ? 16 : 20,
          42,
        ),
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 1260),
            child: child,
          ),
        ),
      ),
    );
  }
}

class _PageTitle extends StatelessWidget {
  const _PageTitle({required this.desktop, required this.mobile});
  final String desktop;
  final String mobile;

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.sizeOf(context).width;
    final compact = width < 768;
    return Text(
      compact ? mobile : desktop,
      style: Theme.of(context).textTheme.displaySmall?.copyWith(
        fontSize: compact ? (width < 390 ? 32 : 35) : null,
      ),
    );
  }
}

class _MobileActionStack extends StatelessWidget {
  const _MobileActionStack({required this.onNavigate});
  final ValueChanged<_Destination> onNavigate;

  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      FilledButton.icon(
        onPressed: () => onNavigate(_Destination.analysis),
        icon: const Icon(Icons.upload_file_outlined),
        label: const Text('Review an audio file'),
      ),
      const SizedBox(height: 10),
      FilledButton.tonalIcon(
        onPressed: () => onNavigate(_Destination.liveCall),
        icon: const Icon(Icons.phone_in_talk_rounded),
        label: const Text('Live fraud call demo'),
      ),
      const SizedBox(height: 10),
      OutlinedButton.icon(
        onPressed: () => onNavigate(_Destination.live),
        icon: const Icon(Icons.play_circle_outline_rounded),
        label: const Text('Run judge walkthrough'),
      ),
    ],
  );
}

class _MobileEvidenceDisclosure extends StatelessWidget {
  const _MobileEvidenceDisclosure({required this.report});
  final AnalysisReport report;

  @override
  Widget build(BuildContext context) => Card(
    child: ExpansionTile(
      tilePadding: const EdgeInsets.symmetric(horizontal: 18, vertical: 5),
      childrenPadding: const EdgeInsets.fromLTRB(18, 0, 18, 18),
      title: const Text(
        'Why this verdict?',
        style: TextStyle(fontWeight: FontWeight.w800),
      ),
      subtitle: Text(
        report.agreement,
        style: const TextStyle(fontSize: 13, color: AppColors.mutedInk),
      ),
      children: [
        const Align(
          alignment: Alignment.centerLeft,
          child: Text(
            'INDEPENDENT SIGNALS',
            style: TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w800,
              color: AppColors.mutedInk,
              letterSpacing: .8,
            ),
          ),
        ),
        const SizedBox(height: 10),
        ...report.experts.map((expert) => _MobileEvidenceLine(expert: expert)),
        const Divider(height: 28, color: AppColors.line),
        const Align(
          alignment: Alignment.centerLeft,
          child: Text(
            'RISK OVER TIME',
            style: TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w800,
              color: AppColors.mutedInk,
              letterSpacing: .8,
            ),
          ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          height: 142,
          width: double.infinity,
          child: CustomPaint(painter: _TimelinePainter(report.windows)),
        ),
      ],
    ),
  );
}

class _MobileEvidenceLine extends StatelessWidget {
  const _MobileEvidenceLine({required this.expert});
  final ExpertScore expert;

  @override
  Widget build(BuildContext context) {
    final probability = expert.probability;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          Expanded(
            child: Text(
              expert.label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w700),
            ),
          ),
          const SizedBox(width: 10),
          Text(
            probability == null ? '—' : '${(probability * 100).round()}%',
            style: TextStyle(
              fontWeight: FontWeight.w800,
              color: _riskColor(expert.riskState),
            ),
          ),
          const SizedBox(width: 8),
          _StatusTag(risk: expert.riskState, compact: true),
        ],
      ),
    );
  }
}

class _Eyebrow extends StatelessWidget {
  const _Eyebrow(this.text);
  final String text;

  @override
  Widget build(BuildContext context) => Text(
    text,
    style: Theme.of(
      context,
    ).textTheme.labelSmall?.copyWith(color: AppColors.moss),
  );
}

class _VerdictCard extends StatelessWidget {
  const _VerdictCard({required this.report});
  final AnalysisReport report;

  @override
  Widget build(BuildContext context) {
    final color = _riskColor(report.riskState);
    return Card(
      child: Padding(
        padding: EdgeInsets.all(
          MediaQuery.sizeOf(context).width < 768 ? 20 : 24,
        ),
        child: LayoutBuilder(
          builder: (context, constraints) {
            final compact = constraints.maxWidth < 520;
            final vertical = compact;
            final dial = _RiskDial(
              probability: report.probability,
              risk: report.riskState,
              compact: compact,
            );
            final words = Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    _StatusTag(risk: report.riskState),
                    if (report.isDemo) ...[
                      const SizedBox(width: 8),
                      const _TinyTag(text: 'SHOWCASE DATA'),
                    ],
                  ],
                ),
                const SizedBox(height: 17),
                Text(
                  report.riskState.label,
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                const SizedBox(height: 8),
                Text(
                  report.action,
                  style: Theme.of(context).textTheme.bodyLarge,
                ),
                if (!compact) ...[
                  const SizedBox(height: 19),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: _riskSoftColor(report.riskState),
                      borderRadius: BorderRadius.circular(13),
                    ),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(
                          Icons.assistant_navigation,
                          color: color,
                          size: 19,
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            'Next step: ${report.action}',
                            style: TextStyle(
                              color: color,
                              height: 1.35,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ],
            );
            return vertical
                ? Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [dial, const SizedBox(height: 22), words],
                  )
                : Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      dial,
                      const SizedBox(width: 26),
                      Expanded(child: words),
                    ],
                  );
          },
        ),
      ),
    );
  }
}

class _RiskDial extends StatelessWidget {
  const _RiskDial({
    required this.probability,
    required this.risk,
    this.compact = false,
  });
  final double? probability;
  final RiskState risk;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final pct = probability == null ? '—' : '${(probability! * 100).round()}%';
    final dimension = compact ? 136.0 : 164.0;
    return SizedBox(
      width: dimension,
      height: dimension,
      child: Stack(
        alignment: Alignment.center,
        children: [
          CustomPaint(
            size: Size.square(dimension),
            painter: _RiskDialPainter(probability ?? 0, _riskColor(risk)),
          ),
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                pct,
                style: TextStyle(
                  fontFamily: 'Georgia',
                  fontSize: compact ? 32 : 37,
                  fontWeight: FontWeight.w700,
                  letterSpacing: -1.5,
                ),
              ),
              const SizedBox(height: 2),
              const Text(
                'SYNTHETIC RISK',
                style: TextStyle(
                  fontSize: 9,
                  fontWeight: FontWeight.w800,
                  color: AppColors.mutedInk,
                  letterSpacing: .8,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _RiskDialPainter extends CustomPainter {
  _RiskDialPainter(this.value, this.color);
  final double value;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final rect = Rect.fromCircle(center: center, radius: size.width * .405);
    const start = math.pi * .78;
    const span = math.pi * 1.44;
    final background = Paint()
      ..color = AppColors.line
      ..style = PaintingStyle.stroke
      ..strokeWidth = 11
      ..strokeCap = StrokeCap.round;
    final foreground = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 11
      ..strokeCap = StrokeCap.round;
    canvas.drawArc(rect, start, span, false, background);
    canvas.drawArc(
      rect,
      start,
      span * value.clamp(0, 1).toDouble(),
      false,
      foreground,
    );
  }

  @override
  bool shouldRepaint(covariant _RiskDialPainter oldDelegate) =>
      oldDelegate.value != value || oldDelegate.color != color;
}

class _SystemStatusCard extends StatelessWidget {
  const _SystemStatusCard({
    required this.health,
    required this.checking,
    required this.healthError,
    required this.onCheck,
  });
  final BackendHealth? health;
  final bool checking;
  final String? healthError;
  final VoidCallback onCheck;

  @override
  Widget build(BuildContext context) {
    final online = health != null;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(22),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const _Eyebrow('SYSTEM READINESS'),
            const SizedBox(height: 12),
            Row(
              children: [
                _ConnectionPill(connected: online, loading: checking),
                const Spacer(),
                TextButton(
                  onPressed: checking ? null : onCheck,
                  child: const Text('Check backend'),
                ),
              ],
            ),
            const SizedBox(height: 18),
            _Metric(
              label: 'Decision path',
              value: health?.decisionLabel ?? 'Showcase simulator',
            ),
            const SizedBox(height: 13),
            _Metric(
              label: 'Signal cadence',
              value: health == null
                  ? '4.0 s windows · 0.5 s hop'
                  : '${health!.windowSeconds.toStringAsFixed(1)} s windows · ${health!.hopSeconds.toStringAsFixed(1)} s hop',
            ),
            const SizedBox(height: 13),
            _Metric(
              label: 'Input format',
              value: health == null
                  ? '16 kHz mono audio'
                  : '${health!.sampleRate ~/ 1000} kHz mono audio',
            ),
            if (healthError != null) ...[
              const SizedBox(height: 14),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: AppColors.vermilion.withValues(alpha: 0.08),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(
                    color: AppColors.vermilion.withValues(alpha: 0.25),
                  ),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(
                      Icons.error_outline_rounded,
                      color: AppColors.vermilion,
                      size: 16,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _friendlyError(healthError!),
                        style: const TextStyle(
                          color: AppColors.vermilion,
                          fontSize: 12,
                          fontWeight: FontWeight.w500,
                          height: 1.35,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _Metric extends StatelessWidget {
  const _Metric({required this.label, required this.value});
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Text(
        label.toUpperCase(),
        style: Theme.of(context).textTheme.labelSmall?.copyWith(
          color: AppColors.mutedInk,
          fontSize: 9,
        ),
      ),
      const SizedBox(height: 4),
      Text(
        value,
        style: const TextStyle(fontWeight: FontWeight.w700, height: 1.2),
      ),
    ],
  );
}

class _EvidenceStrip extends StatelessWidget {
  const _EvidenceStrip({required this.report});
  final AnalysisReport report;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final columns = constraints.maxWidth > 850 ? 3 : 1;
        return GridView.count(
          crossAxisCount: columns,
          crossAxisSpacing: 18,
          mainAxisSpacing: 14,
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          childAspectRatio: columns == 3 ? 2.65 : 2.1,
          children: [
            _EvidenceFact(
              icon: Icons.hub_outlined,
              label: 'DECISION MODEL',
              value: report.decisionExpert,
              helper: 'The route that produces the displayed risk.',
            ),
            _EvidenceFact(
              icon: Icons.groups_2_outlined,
              label: 'EXPERT AGREEMENT',
              value: report.agreement,
              helper:
                  'Independent scores are evidence, not a vote of identity.',
            ),
            _EvidenceFact(
              icon: Icons.timer_outlined,
              label: 'PEAK SIGNAL',
              value: report.peakTimeSeconds == null
                  ? 'Not reported'
                  : '${report.peakTimeSeconds!.toStringAsFixed(1)} seconds',
              helper:
                  '${report.suspiciousWindows} of ${report.totalWindows} windows crossed high risk.',
            ),
          ],
        );
      },
    );
  }
}

class _EvidenceFact extends StatelessWidget {
  const _EvidenceFact({
    required this.icon,
    required this.label,
    required this.value,
    required this.helper,
  });
  final IconData icon;
  final String label;
  final String value;
  final String helper;

  @override
  Widget build(BuildContext context) => Card(
    child: Padding(
      padding: const EdgeInsets.all(18),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: AppColors.moss, size: 22),
          const SizedBox(width: 13),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: Theme.of(context).textTheme.labelSmall?.copyWith(
                    fontSize: 9,
                    color: AppColors.mutedInk,
                  ),
                ),
                const SizedBox(height: 5),
                Text(
                  value,
                  style: const TextStyle(
                    fontWeight: FontWeight.w800,
                    height: 1.22,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  helper,
                  style: const TextStyle(
                    fontSize: 12,
                    color: AppColors.mutedInk,
                    height: 1.28,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    ),
  );
}

class _TimelineCard extends StatelessWidget {
  const _TimelineCard({required this.report});
  final AnalysisReport report;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(22),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const _Eyebrow('RISK OVER TIME'),
            const SizedBox(height: 6),
            Text(
              report.sourceLabel,
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 3),
            const Text(
              'The primary line is the calibrated decision probability for each audio window.',
            ),
            const SizedBox(height: 18),
            SizedBox(
              height: 190,
              width: double.infinity,
              child: CustomPaint(painter: _TimelinePainter(report.windows)),
            ),
            const SizedBox(height: 6),
            const Wrap(
              spacing: 16,
              runSpacing: 5,
              children: [
                _LegendDot(color: AppColors.moss, label: 'Low risk < 35%'),
                _LegendDot(color: AppColors.amber, label: 'Review 35–64%'),
                _LegendDot(
                  color: AppColors.vermilion,
                  label: 'High risk ≥ 65%',
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _TimelinePainter extends CustomPainter {
  _TimelinePainter(this.windows);
  final List<WindowScore> windows;

  @override
  void paint(Canvas canvas, Size size) {
    const left = 6.0;
    const right = 6.0;
    const top = 10.0;
    const bottom = 20.0;
    final graph = Rect.fromLTWH(
      left,
      top,
      size.width - left - right,
      size.height - top - bottom,
    );
    final linePaint = Paint()..strokeWidth = 1;
    for (final threshold in [.35, .65]) {
      final y = graph.bottom - graph.height * threshold;
      linePaint.color = threshold == .35
          ? AppColors.amber.withValues(alpha: .45)
          : AppColors.vermilion.withValues(alpha: .45);
      canvas.drawLine(Offset(graph.left, y), Offset(graph.right, y), linePaint);
    }
    if (windows.isEmpty) {
      final text = TextPainter(
        text: const TextSpan(
          text: 'No scored windows yet',
          style: TextStyle(color: AppColors.mutedInk),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      text.paint(
        canvas,
        Offset((size.width - text.width) / 2, (size.height - text.height) / 2),
      );
      return;
    }
    final points = <Offset>[];
    for (var index = 0; index < windows.length; index++) {
      final p = windows[index].probability ?? 0;
      final x =
          graph.left +
          graph.width *
              (windows.length == 1 ? .5 : index / (windows.length - 1));
      final y = graph.bottom - graph.height * p.clamp(0, 1);
      points.add(Offset(x, y));
    }
    final fill = Path()..moveTo(points.first.dx, graph.bottom);
    for (final point in points) {
      fill.lineTo(point.dx, point.dy);
    }
    fill.lineTo(points.last.dx, graph.bottom);
    fill.close();
    canvas.drawPath(
      fill,
      Paint()..color = AppColors.moss.withValues(alpha: .10),
    );
    final line = Path()..moveTo(points.first.dx, points.first.dy);
    for (final point in points.skip(1)) {
      line.lineTo(point.dx, point.dy);
    }
    canvas.drawPath(
      line,
      Paint()
        ..color = AppColors.moss
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round,
    );
    for (final point in points.where(
      (point) => point.dy < graph.bottom - graph.height * .65,
    )) {
      canvas.drawCircle(point, 4.5, Paint()..color = AppColors.vermilion);
      canvas.drawCircle(point, 2.2, Paint()..color = AppColors.paper);
    }
  }

  @override
  bool shouldRepaint(covariant _TimelinePainter oldDelegate) =>
      oldDelegate.windows != windows;
}

class _LegendDot extends StatelessWidget {
  const _LegendDot({required this.color, required this.label});
  final Color color;
  final String label;

  @override
  Widget build(BuildContext context) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Container(
        width: 8,
        height: 8,
        decoration: BoxDecoration(color: color, shape: BoxShape.circle),
      ),
      const SizedBox(width: 5),
      Text(
        label,
        style: const TextStyle(fontSize: 11, color: AppColors.mutedInk),
      ),
    ],
  );
}

class _HonestyNote extends StatelessWidget {
  const _HonestyNote();

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(18),
    decoration: BoxDecoration(
      color: AppColors.ink,
      borderRadius: BorderRadius.circular(18),
    ),
    child: const Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(Icons.info_outline_rounded, color: AppColors.paper, size: 20),
        SizedBox(width: 12),
        Expanded(
          child: Text(
            'Responsible use: this detector estimates whether audio resembles machine-generated speech. It does not identify a person, prove fraud, or replace a verification process.',
            style: TextStyle(color: AppColors.paper, height: 1.4),
          ),
        ),
      ],
    ),
  );
}

class _FileAnalysisPage extends StatefulWidget {
  const _FileAnalysisPage({
    required this.api,
    required this.report,
    required this.onReport,
  });
  final VoiceIntegrityApi api;
  final AnalysisReport report;
  final ValueChanged<AnalysisReport> onReport;

  @override
  State<_FileAnalysisPage> createState() => _FileAnalysisPageState();
}

class _FileAnalysisPageState extends State<_FileAnalysisPage> {
  bool _analysing = false;
  String? _error;
  String? _filename;

  Future<void> _chooseAudio() async {
    setState(() => _error = null);
    final picked = await FilePicker.platform.pickFiles(
      type: FileType.audio,
      withData: true,
    );
    final file = picked?.files.singleOrNull;
    if (file == null) return;
    final bytes = file.bytes;
    if (bytes == null) {
      setState(
        () => _error =
            'This platform did not return the selected file bytes. Try a smaller WAV, MP3, or FLAC file.',
      );
      return;
    }
    await _analyse(bytes, file.name);
  }

  Future<void> _analyse(Uint8List bytes, String filename) async {
    setState(() {
      _analysing = true;
      _error = null;
      _filename = filename;
    });
    try {
      final report = await widget.api.analyzeAudio(
        bytes: bytes,
        filename: filename,
      );
      if (!mounted) return;
      widget.onReport(report);
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = _friendlyError('$error'));
    } finally {
      if (mounted) setState(() => _analysing = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final report = widget.report;
    return _PageScroll(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const _Eyebrow('FORENSIC FILE REVIEW'),
          const SizedBox(height: 10),
          const _PageTitle(
            desktop: 'Review the call,\nnot just the verdict.',
            mobile: 'Review a\nrecording.',
          ),
          const SizedBox(height: 11),
          Text(
            MediaQuery.sizeOf(context).width < 768
                ? 'Upload a recording to see the decision, action, and model evidence.'
                : 'Upload WAV, MP3, or FLAC audio. The backend evaluates every window, then returns an explainable timeline and per-model evidence.',
          ),
          const SizedBox(height: 25),
          _UploadPanel(
            filename: _filename,
            loading: _analysing,
            onChoose: _analysing ? null : _chooseAudio,
            onDemo: _analysing
                ? null
                : () => widget.onReport(AnalysisReport.demo()),
          ),
          if (_error != null) ...[
            const SizedBox(height: 14),
            _InlineAlert(message: _error!),
          ],
          const SizedBox(height: 20),
          _VerdictCard(report: report),
          const SizedBox(height: 18),
          _ExpertReviewCard(report: report),
          const SizedBox(height: 18),
          _TimelineCard(report: report),
          const SizedBox(height: 18),
          const _HonestyNote(),
        ],
      ),
    );
  }
}

class _UploadPanel extends StatelessWidget {
  const _UploadPanel({
    required this.filename,
    required this.loading,
    required this.onChoose,
    required this.onDemo,
  });
  final String? filename;
  final bool loading;
  final VoidCallback? onChoose;
  final VoidCallback? onDemo;

  @override
  Widget build(BuildContext context) => Container(
    width: double.infinity,
    padding: const EdgeInsets.all(24),
    decoration: BoxDecoration(
      color: AppColors.paper,
      border: Border.all(color: AppColors.line, width: 1.25),
      borderRadius: BorderRadius.circular(20),
    ),
    child: LayoutBuilder(
      builder: (context, constraints) {
        final compact = constraints.maxWidth < 620;
        final content = [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: AppColors.mossSoft,
              borderRadius: BorderRadius.circular(14),
            ),
            child: const Icon(Icons.graphic_eq_rounded, color: AppColors.moss),
          ),
          const SizedBox(width: 15),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  filename ?? 'Choose an audio recording',
                  style: const TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  loading
                      ? 'Processing every clean audio window…'
                      : 'WAV, MP3, and FLAC. Files are sent only to the configured local backend.',
                  style: const TextStyle(
                    color: AppColors.mutedInk,
                    fontSize: 13,
                    height: 1.3,
                  ),
                ),
              ],
            ),
          ),
        ];
        final controls = Wrap(
          spacing: 10,
          runSpacing: 10,
          children: [
            FilledButton.icon(
              onPressed: onChoose,
              icon: loading
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        color: AppColors.paper,
                      ),
                    )
                  : const Icon(Icons.upload_file_outlined),
              label: Text(loading ? 'Analysing' : 'Choose audio'),
            ),
            OutlinedButton(
              onPressed: onDemo,
              child: const Text('Load judge demo'),
            ),
          ],
        );
        return compact
            ? Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(children: content),
                  const SizedBox(height: 18),
                  controls,
                ],
              )
            : Row(children: [...content, controls]);
      },
    ),
  );
}

class _InlineAlert extends StatelessWidget {
  const _InlineAlert({required this.message});
  final String message;

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: AppColors.vermilionSoft,
      borderRadius: BorderRadius.circular(14),
    ),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Icon(Icons.error_outline_rounded, color: AppColors.vermilion),
        const SizedBox(width: 10),
        Expanded(
          child: Text(
            message,
            style: const TextStyle(color: AppColors.vermilion, height: 1.35),
          ),
        ),
      ],
    ),
  );
}

class _ExpertReviewCard extends StatelessWidget {
  const _ExpertReviewCard({required this.report});
  final AnalysisReport report;

  @override
  Widget build(BuildContext context) => Card(
    child: Padding(
      padding: const EdgeInsets.all(22),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const _Eyebrow('INDEPENDENT MODEL EVIDENCE'),
          const SizedBox(height: 7),
          const Text(
            'Each model has its own probability calibration. A higher raw logit and probability mean the audio is more synthetic-like.',
          ),
          const SizedBox(height: 18),
          ...report.experts.map(
            (expert) => Padding(
              padding: const EdgeInsets.only(bottom: 13),
              child: _ExpertRow(expert: expert),
            ),
          ),
        ],
      ),
    ),
  );
}

class _ExpertRow extends StatelessWidget {
  const _ExpertRow({required this.expert});
  final ExpertScore expert;

  @override
  Widget build(BuildContext context) {
    final probability = expert.probability;
    final color = _riskColor(expert.riskState);
    return Container(
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        border: Border.all(color: AppColors.line),
        borderRadius: BorderRadius.circular(15),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  expert.label,
                  style: const TextStyle(fontWeight: FontWeight.w800),
                ),
              ),
              _StatusTag(risk: expert.riskState, compact: true),
            ],
          ),
          const SizedBox(height: 13),
          Row(
            children: [
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: LinearProgressIndicator(
                    value: probability,
                    minHeight: 8,
                    backgroundColor: AppColors.line,
                    valueColor: AlwaysStoppedAnimation(color),
                  ),
                ),
              ),
              const SizedBox(width: 13),
              Text(
                probability == null
                    ? '—'
                    : '${(probability * 100).toStringAsFixed(1)}%',
                style: TextStyle(fontWeight: FontWeight.w800, color: color),
              ),
            ],
          ),
          if (expert.rawLogit != null || expert.modelVersion != null) ...[
            const SizedBox(height: 11),
            Wrap(
              spacing: 12,
              runSpacing: 4,
              children: [
                if (expert.rawLogit != null)
                  _MiniMetric(
                    label: 'RAW LOGIT',
                    value: expert.rawLogit!.toStringAsFixed(2),
                  ),
                if (expert.modelVersion != null)
                  _MiniMetric(label: 'VERSION', value: expert.modelVersion!),
              ],
            ),
          ],
        ],
      ),
    );
  }
}

class _MiniMetric extends StatelessWidget {
  const _MiniMetric({required this.label, required this.value});
  final String label;
  final String value;

  @override
  Widget build(BuildContext context) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Text(
        '$label  ',
        style: const TextStyle(
          fontSize: 9,
          color: AppColors.mutedInk,
          fontWeight: FontWeight.w800,
          letterSpacing: .6,
        ),
      ),
      Text(
        value,
        style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w700),
      ),
    ],
  );
}

class _LiveMonitorPage extends StatefulWidget {
  const _LiveMonitorPage({required this.report, required this.onReport});
  final AnalysisReport report;
  final ValueChanged<AnalysisReport> onReport;

  @override
  State<_LiveMonitorPage> createState() => _LiveMonitorPageState();
}

class _LiveMonitorPageState extends State<_LiveMonitorPage> {
  Timer? _timer;
  var _tick = 0;
  bool get _running => _timer != null;

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  void _toggle() {
    if (_timer != null) {
      _timer?.cancel();
      setState(() => _timer = null);
      return;
    }
    _timer = Timer.periodic(const Duration(milliseconds: 780), (_) {
      _tick++;
      final report = AnalysisReport.demo(phase: _tick * .45);
      if (mounted) {
        setState(() {});
        widget.onReport(report);
      }
    });
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final report = _running
        ? AnalysisReport.demo(phase: _tick * .45)
        : widget.report;
    return _PageScroll(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const _Eyebrow('LIVE DEMONSTRATION'),
          const SizedBox(height: 10),
          const _PageTitle(
            desktop: 'Make the model\nlegible in seconds.',
            mobile: 'See the\ndecision move.',
          ),
          const SizedBox(height: 11),
          Text(
            MediaQuery.sizeOf(context).width < 768
                ? 'A reliable local walkthrough—no microphone audio is sent.'
                : 'This guided walkthrough visualises score updates without sending microphone audio. Use it as a reliable demo-day path, then connect the platform audio bridge for production streaming.',
          ),
          const SizedBox(height: 25),
          _LiveControlCard(running: _running, onToggle: _toggle),
          const SizedBox(height: 20),
          _VerdictCard(report: report),
          const SizedBox(height: 18),
          _TimelineCard(report: report),
          const SizedBox(height: 18),
          const _HonestyNote(),
        ],
      ),
    );
  }
}

class _LiveControlCard extends StatelessWidget {
  const _LiveControlCard({required this.running, required this.onToggle});
  final bool running;
  final VoidCallback onToggle;

  @override
  Widget build(BuildContext context) => Card(
    child: Padding(
      padding: EdgeInsets.all(MediaQuery.sizeOf(context).width < 768 ? 18 : 22),
      child: LayoutBuilder(
        builder: (context, constraints) {
          final compact = constraints.maxWidth < 520;
          final marker = Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: running ? AppColors.vermilionSoft : AppColors.mossSoft,
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(
              running ? Icons.stop_rounded : Icons.play_arrow_rounded,
              color: running ? AppColors.vermilion : AppColors.moss,
            ),
          );
          final copy = Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                running
                    ? 'Walkthrough running'
                    : 'Ready for the judge walkthrough',
                style: const TextStyle(
                  fontWeight: FontWeight.w800,
                  fontSize: 16,
                ),
              ),
              const SizedBox(height: 3),
              Text(
                running
                    ? 'The score stream is changing in controlled showcase mode.'
                    : 'Press play to show the risk timeline, evidence, and recommended action.',
                style: const TextStyle(
                  fontSize: 13,
                  color: AppColors.mutedInk,
                  height: 1.3,
                ),
              ),
            ],
          );
          final control = FilledButton.icon(
            onPressed: onToggle,
            icon: Icon(running ? Icons.stop_rounded : Icons.play_arrow_rounded),
            label: Text(running ? 'Stop walkthrough' : 'Start walkthrough'),
          );
          if (compact) {
            return Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Row(
                  children: [
                    marker,
                    const SizedBox(width: 14),
                    Expanded(child: copy),
                  ],
                ),
                const SizedBox(height: 18),
                control,
              ],
            );
          }
          return Row(
            children: [
              marker,
              const SizedBox(width: 14),
              Expanded(child: copy),
              const SizedBox(width: 12),
              control,
            ],
          );
        },
      ),
    ),
  );
}

class _ModelReviewPage extends StatelessWidget {
  const _ModelReviewPage({
    required this.endpointController,
    required this.health,
    required this.checkingHealth,
    required this.healthError,
    required this.onCheckHealth,
  });
  final TextEditingController endpointController;
  final BackendHealth? health;
  final bool checkingHealth;
  final String? healthError;
  final VoidCallback onCheckHealth;

  @override
  Widget build(BuildContext context) => _PageScroll(
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const _Eyebrow('MODEL & SYSTEM REVIEW'),
        const SizedBox(height: 10),
        const _PageTitle(
          desktop: 'The evidence behind\nthe risk band.',
          mobile: 'Inspect the\nevidence.',
        ),
        const SizedBox(height: 11),
        const Text(
          'This app deliberately distinguishes independent model evidence from the configured decision path. A result is useful only when its model, preprocessing, and calibrator match.',
        ),
        const SizedBox(height: 25),
        _BackendConnectionCard(
          controller: endpointController,
          health: health,
          checking: checkingHealth,
          healthError: healthError,
          onCheck: onCheckHealth,
        ),
        const SizedBox(height: 18),
        LayoutBuilder(
          builder: (context, constraints) {
            final wide = constraints.maxWidth > 880;
            final left = const _ModelCards();
            final right = const _DecisionMethodCard();
            return wide
                ? Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(child: left),
                      const SizedBox(width: 18),
                      Expanded(child: right),
                    ],
                  )
                : Column(children: [left, const SizedBox(height: 18), right]);
          },
        ),
        const SizedBox(height: 18),
        const _SafeguardsCard(),
        const SizedBox(height: 18),
        const _HonestyNote(),
      ],
    ),
  );
}

class _BackendConnectionCard extends StatelessWidget {
  const _BackendConnectionCard({
    required this.controller,
    required this.health,
    required this.checking,
    required this.healthError,
    required this.onCheck,
  });
  final TextEditingController controller;
  final BackendHealth? health;
  final bool checking;
  final String? healthError;
  final VoidCallback onCheck;

  @override
  Widget build(BuildContext context) => Card(
    child: Padding(
      padding: const EdgeInsets.all(22),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const _Eyebrow('BACKEND CONNECTION'),
          const SizedBox(height: 12),
          TextField(
            controller: controller,
            keyboardType: TextInputType.url,
            decoration: const InputDecoration(
              labelText: 'Backend URL',
              hintText: 'https://codequantum.in/sih',
              prefixIcon: Icon(Icons.link_rounded),
            ),
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              Text(
                '1-Tap Presets:',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: AppColors.ink.withValues(alpha: 0.65),
                ),
              ),
              ActionChip(
                avatar: const Icon(Icons.cloud_done_outlined, size: 16),
                label: const Text('Cloud VPS (Default)'),
                onPressed: checking
                    ? null
                    : () {
                        controller.text = VoiceIntegrityApi.defaultEndpoint;
                        onCheck();
                      },
              ),
              ActionChip(
                avatar: const Icon(Icons.laptop_chromebook_rounded, size: 16),
                label: const Text('Localhost:8000'),
                onPressed: checking
                    ? null
                    : () {
                        controller.text = 'http://127.0.0.1:8000';
                        onCheck();
                      },
              ),
            ],
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 12,
            runSpacing: 10,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              FilledButton.icon(
                onPressed: checking ? null : onCheck,
                icon: checking
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: AppColors.paper,
                        ),
                      )
                    : const Icon(Icons.health_and_safety_outlined),
                label: Text(checking ? 'Checking' : 'Check connection'),
              ),
              if (health != null)
                _ConnectionPill(connected: true, loading: false),
            ],
          ),
          if (healthError != null) ...[
            const SizedBox(height: 12),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AppColors.vermilion.withValues(alpha: 0.08),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(
                  color: AppColors.vermilion.withValues(alpha: 0.3),
                ),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(
                    Icons.error_outline_rounded,
                    color: AppColors.vermilion,
                    size: 18,
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      _errorExplanation(healthError!),
                      style: const TextStyle(
                        color: AppColors.vermilion,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        height: 1.4,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    ),
  );
}

class _ModelCards extends StatelessWidget {
  const _ModelCards();

  @override
  Widget build(BuildContext context) => Card(
    child: Padding(
      padding: const EdgeInsets.all(22),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const _Eyebrow('THE TWO EXPERTS'),
          const SizedBox(height: 8),
          const Text(
            'Each expert scores the same clean 16 kHz mono audio window. Their raw logits are not interchangeable; each gets its own Platt calibrator before a probability is displayed.',
          ),
          const SizedBox(height: 18),
          const _ModelExplanation(
            title: '1 · WavLM Base+',
            tag: 'SPEECH REPRESENTATION',
            body:
                'A frozen WavLM Base+ speech backbone with a classifier head. It contributes an independent synthetic-speech signal and is calibrated with platt_v5.',
          ),
          const Divider(height: 26, color: AppColors.line),
          const _ModelExplanation(
            title: '2 · LFCC-LCNN Hybrid',
            tag: 'SPECTRAL FORENSICS',
            body:
                'An LFCC-LCNN classifier fine-tuned with newer engine clips. The optional bandwidth-robust variant gates audio at 7 kHz to prevent the resampler from becoming an accidental label signal.',
          ),
        ],
      ),
    ),
  );
}

class _ModelExplanation extends StatelessWidget {
  const _ModelExplanation({
    required this.title,
    required this.tag,
    required this.body,
  });
  final String title;
  final String tag;
  final String body;

  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Text(
        tag,
        style: Theme.of(
          context,
        ).textTheme.labelSmall?.copyWith(fontSize: 9, color: AppColors.moss),
      ),
      const SizedBox(height: 5),
      Text(
        title,
        style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w800),
      ),
      const SizedBox(height: 5),
      Text(
        body,
        style: const TextStyle(
          color: AppColors.mutedInk,
          fontSize: 13,
          height: 1.38,
        ),
      ),
    ],
  );
}

class _DecisionMethodCard extends StatelessWidget {
  const _DecisionMethodCard();

  @override
  Widget build(BuildContext context) => Card(
    child: Padding(
      padding: const EdgeInsets.all(22),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const _Eyebrow('HOW A DECISION IS MADE'),
          const SizedBox(height: 8),
          const Text(
            'The deployed batch endpoint uses a probability-space weighted ensemble after each model has been calibrated.',
          ),
          const SizedBox(height: 18),
          const _FormulaLine(left: 'WavLM Base+', value: '50%'),
          const _FormulaLine(left: 'LFCC-LCNN Hybrid', value: '50%'),
          const SizedBox(height: 14),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: AppColors.mossSoft,
              borderRadius: BorderRadius.circular(13),
            ),
            child: const Text(
              'Decision Probability = 50% WavLM Base+ + 50% LFCC-LCNN Hybrid\nLow < 35% · Review 35–64% · High ≥ 65%',
              style: TextStyle(
                color: AppColors.moss,
                fontWeight: FontWeight.w800,
                height: 1.4,
              ),
            ),
          ),
          const SizedBox(height: 16),
          const Text(
            'For live streaming, the backend reports its active fusion mode through /health. The app reads that endpoint so the system page can identify the actual runtime path.',
            style: TextStyle(
              color: AppColors.mutedInk,
              fontSize: 12,
              height: 1.35,
            ),
          ),
        ],
      ),
    ),
  );
}

class _FormulaLine extends StatelessWidget {
  const _FormulaLine({required this.left, required this.value});
  final String left;
  final String value;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 7),
    child: Row(
      children: [
        Text(left, style: const TextStyle(fontWeight: FontWeight.w700)),
        const Spacer(),
        Text(
          value,
          style: const TextStyle(
            fontFamily: 'Georgia',
            fontSize: 20,
            fontWeight: FontWeight.w700,
            color: AppColors.moss,
          ),
        ),
      ],
    ),
  );
}

class _SafeguardsCard extends StatelessWidget {
  const _SafeguardsCard();

  @override
  Widget build(BuildContext context) => Card(
    child: Padding(
      padding: const EdgeInsets.all(22),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const _Eyebrow('NON-NEGOTIABLE SAFEGUARDS'),
          const SizedBox(height: 9),
          const Text(
            'The Flutter app never invents a “safe” answer. It preserves the backend’s unavailable state and shows the quality reason as a stop condition.',
          ),
          const SizedBox(height: 16),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: const [
              _Safeguard(
                label: 'Calibration',
                text: 'One calibrator per model and checkpoint.',
              ),
              _Safeguard(
                label: 'Preprocessing',
                text:
                    'Training and inference must use matching resampling and band-gate rules.',
              ),
              _Safeguard(
                label: 'Quality gate',
                text:
                    'Silence, clipping, decode failures, and dropped frames are unavailable.',
              ),
              _Safeguard(
                label: 'Policy',
                text:
                    'A risk score recommends verification; it never proves identity.',
              ),
            ],
          ),
        ],
      ),
    ),
  );
}

class _Safeguard extends StatelessWidget {
  const _Safeguard({required this.label, required this.text});
  final String label;
  final String text;

  @override
  Widget build(BuildContext context) => Container(
    width: 250,
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: AppColors.canvas,
      borderRadius: BorderRadius.circular(14),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(fontWeight: FontWeight.w800)),
        const SizedBox(height: 4),
        Text(
          text,
          style: const TextStyle(
            color: AppColors.mutedInk,
            fontSize: 12,
            height: 1.32,
          ),
        ),
      ],
    ),
  );
}

class _ShieldMark extends StatelessWidget {
  const _ShieldMark({this.compact = false});
  final bool compact;

  @override
  Widget build(BuildContext context) => Container(
    width: compact ? 34 : 38,
    height: compact ? 34 : 38,
    decoration: BoxDecoration(
      color: AppColors.ink,
      borderRadius: BorderRadius.circular(11),
    ),
    child: const Icon(Icons.shield_outlined, color: AppColors.paper, size: 22),
  );
}

class _ConnectionPill extends StatelessWidget {
  const _ConnectionPill({
    required this.connected,
    required this.loading,
    this.compact = false,
  });
  final bool connected;
  final bool loading;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final color = connected ? AppColors.moss : AppColors.amber;
    final label = loading
        ? 'Checking backend'
        : connected
        ? 'Backend ready'
        : 'Showcase mode';
    if (compact) {
      return Tooltip(
        message: label,
        child: Container(
          width: 36,
          height: 36,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: connected ? AppColors.mossSoft : AppColors.amberSoft,
            borderRadius: BorderRadius.circular(18),
          ),
          child: loading
              ? const SizedBox(
                  width: 14,
                  height: 14,
                  child: CircularProgressIndicator(
                    strokeWidth: 1.8,
                    color: AppColors.amber,
                  ),
                )
              : Container(
                  width: 9,
                  height: 9,
                  decoration: BoxDecoration(
                    color: color,
                    shape: BoxShape.circle,
                  ),
                ),
        ),
      );
    }
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: connected ? AppColors.mossSoft : AppColors.amberSoft,
        borderRadius: BorderRadius.circular(30),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (loading)
            const SizedBox(
              width: 11,
              height: 11,
              child: CircularProgressIndicator(
                strokeWidth: 1.8,
                color: AppColors.amber,
              ),
            )
          else
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle),
            ),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              color: color,
              fontSize: 11,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }
}

class _StatusTag extends StatelessWidget {
  const _StatusTag({required this.risk, this.compact = false});
  final RiskState risk;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final color = _riskColor(risk);
    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: compact ? 8 : 10,
        vertical: compact ? 5 : 6,
      ),
      decoration: BoxDecoration(
        color: _riskSoftColor(risk),
        borderRadius: BorderRadius.circular(30),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(_riskIcon(risk), color: color, size: compact ? 13 : 15),
          const SizedBox(width: 5),
          Text(
            risk.shortLabel.toUpperCase(),
            style: TextStyle(
              color: color,
              fontSize: compact ? 9 : 10,
              letterSpacing: .6,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }
}

class _TinyTag extends StatelessWidget {
  const _TinyTag({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
    decoration: BoxDecoration(
      border: Border.all(color: AppColors.line),
      borderRadius: BorderRadius.circular(20),
    ),
    child: Text(
      text,
      style: const TextStyle(
        color: AppColors.mutedInk,
        fontSize: 9,
        fontWeight: FontWeight.w800,
        letterSpacing: .6,
      ),
    ),
  );
}

Color _riskColor(RiskState risk) => switch (risk) {
  RiskState.low => AppColors.moss,
  RiskState.uncertain || RiskState.collecting => AppColors.amber,
  RiskState.high => AppColors.vermilion,
  RiskState.unavailable => AppColors.unavailable,
};

Color _riskSoftColor(RiskState risk) => switch (risk) {
  RiskState.low => AppColors.mossSoft,
  RiskState.uncertain || RiskState.collecting => AppColors.amberSoft,
  RiskState.high => AppColors.vermilionSoft,
  RiskState.unavailable => AppColors.line,
};

IconData _riskIcon(RiskState risk) => switch (risk) {
  RiskState.low => Icons.check_circle_outline_rounded,
  RiskState.uncertain => Icons.manage_search_outlined,
  RiskState.collecting => Icons.more_time_outlined,
  RiskState.high => Icons.warning_amber_rounded,
  RiskState.unavailable => Icons.do_not_disturb_alt_outlined,
};

String _errorExplanation(String raw) {
  if (raw.contains('Backend error:')) {
    return raw.replaceFirst('VoiceIntegrityApiException: ', '');
  }
  if (raw.contains('Connection refused') ||
      raw.contains('SocketException') ||
      raw.contains('Connection closed') ||
      raw.contains('ClientException') ||
      raw.contains('Failed host lookup') ||
      raw.contains('Broken pipe')) {
    return 'Backend offline or unreachable at this URL.\n• Cloud VPS (Default): Check your internet connection or verify the hosted service (https://codequantum.in/sih).\n• Localhost: Ensure realtime-backend is running locally (python server.py on port 8000).\n• For the Live Fraud Call Demo: the Relay Service is active on port 8001.';
  }
  if (raw.contains('Timed out') || raw.contains('TimeoutException')) {
    return 'The backend did not respond in time (8s timeout). Check network connectivity to the cloud VPS or model loading status in the terminal.';
  }
  return raw.replaceFirst('VoiceIntegrityApiException: ', '');
}

String _friendlyError(String raw) => _errorExplanation(raw);

extension<T> on List<T> {
  T? get singleOrNull => length == 1 ? single : null;
}
