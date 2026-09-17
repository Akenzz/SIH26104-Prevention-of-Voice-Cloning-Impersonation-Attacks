import 'dart:async';

import 'package:flutter/material.dart';

import '../../core/models/voice_models.dart';
import '../../core/services/live_call_service.dart';
import '../../core/services/voice_integrity_api.dart';
import '../live_call/live_call_screen.dart';

// ─────────────────────────────────────────────────────────────────────────────
// DARK OBSIDIAN / SLATE FORENSIC COLOR SYSTEM
// ─────────────────────────────────────────────────────────────────────────────

abstract final class _ObsidianTheme {
  static const canvas = Color(0xFFF8FAFC);
  static const panel = Color(0xFFFFFFFF);
  static const panelElevated = Color(0xFFF1F5F9);
  static const border = Color(0xFFE2E8F0);
  static const borderSubtle = Color(0xFFCBD5E1);

  static const textPrimary = Color(0xFF0F172A);
  static const textMuted = Color(0xFF475569);

  static const emerald = Color(0xFF059669);
  static const emeraldText = Color(0xFF065F46);
  static const emeraldSoft = Color(0xFFECFDF5);
  static const amber = Color(0xFFD97706);
  static const amberText = Color(0xFF92400E);
  static const amberSoft = Color(0xFFFFFBEB);
  static const red = Color(0xFFDC2626);
  static const redText = Color(0xFF991B1B);
  static const redSoft = Color(0xFFFEF2F2);
}

/// Simplified, single-mode shell hosting the real-time Live Call Defense Console.
/// Eliminates unused navigation tabs, synthetic sine wave graphs, and hardcoded showcase data.
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

  void _openSettings() {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => _SettingsSheet(
        endpointController: _endpointController,
        service: _liveCallService,
        health: _health,
        checkingHealth: _checkingHealth,
        healthError: _healthError,
        onRefreshHealth: _checkHealth,
        onEndpointApplied: (newEndpoint) {
          _checkHealth();
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Theme(
      data: ThemeData.light().copyWith(
        scaffoldBackgroundColor: _ObsidianTheme.canvas,
        canvasColor: _ObsidianTheme.canvas,
        cardColor: _ObsidianTheme.panel,
        dividerColor: _ObsidianTheme.border,
        colorScheme: const ColorScheme.light(
          surface: _ObsidianTheme.panel,
          onSurface: _ObsidianTheme.textPrimary,
          primary: _ObsidianTheme.emerald,
          outline: _ObsidianTheme.border,
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          margin: EdgeInsets.zero,
          color: _ObsidianTheme.panel,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
            side: const BorderSide(color: _ObsidianTheme.border, width: 1),
          ),
        ),
      ),
      child: Scaffold(
        backgroundColor: _ObsidianTheme.canvas,
        body: SafeArea(
          child: Column(
            children: [
              _AppHeader(
                isBackendConnected: _health != null,
                checkingHealth: _checkingHealth,
                onOpenSettings: _openSettings,
              ),
              Expanded(
                child: LiveCallScreen(service: _liveCallService),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// TOP BRANDING HEADER & CONNECTIVITY PILL
// ─────────────────────────────────────────────────────────────────────────────

class _AppHeader extends StatelessWidget {
  const _AppHeader({
    required this.isBackendConnected,
    required this.checkingHealth,
    required this.onOpenSettings,
  });

  final bool isBackendConnected;
  final bool checkingHealth;
  final VoidCallback onOpenSettings;

  @override
  Widget build(BuildContext context) {
    final phone = MediaQuery.sizeOf(context).width < 600;
    return Container(
      height: phone ? 60 : 66,
      padding: EdgeInsets.symmetric(horizontal: phone ? 16 : 24),
      decoration: const BoxDecoration(
        color: _ObsidianTheme.panel,
        border: Border(bottom: BorderSide(color: _ObsidianTheme.border, width: 1)),
      ),
      child: Row(
        children: [
          const _ShieldMark(compact: true),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'VOICE INTEGRITY',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontSize: phone ? 14 : 15,
                    fontWeight: FontWeight.w900,
                    letterSpacing: 1.2,
                    color: _ObsidianTheme.textPrimary,
                  ),
                ),
                const Text(
                  'Live Impersonation Defense Console',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w500,
                    color: _ObsidianTheme.textMuted,
                    letterSpacing: .3,
                  ),
                ),
              ],
            ),
          ),
          _ConnectionPill(
            connected: isBackendConnected,
            loading: checkingHealth,
            compact: phone,
          ),
          const SizedBox(width: 10),
          IconButton(
            tooltip: 'System & Model Configuration',
            style: IconButton.styleFrom(
              backgroundColor: _ObsidianTheme.panelElevated,
              side: const BorderSide(color: _ObsidianTheme.border),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10),
              ),
              padding: const EdgeInsets.all(8),
            ),
            onPressed: onOpenSettings,
            icon: const Icon(Icons.tune_rounded, size: 20, color: _ObsidianTheme.textPrimary),
          ),
        ],
      ),
    );
  }
}

class _ShieldMark extends StatelessWidget {
  const _ShieldMark({this.compact = false});

  final bool compact;

  @override
  Widget build(BuildContext context) {
    final size = compact ? 36.0 : 44.0;
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: _ObsidianTheme.emeraldSoft,
        borderRadius: BorderRadius.circular(compact ? 10 : 12),
        border: Border.all(
          color: _ObsidianTheme.emerald.withValues(alpha: 0.3),
          width: 1.2,
        ),
      ),
      child: Center(
        child: Icon(
          Icons.shield_rounded,
          color: _ObsidianTheme.emerald,
          size: compact ? 20 : 24,
        ),
      ),
    );
  }
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
    Color bg;
    Color border;
    Color dot;
    Color text;
    String label;

    if (loading) {
      bg = _ObsidianTheme.amberSoft;
      border = _ObsidianTheme.amber.withValues(alpha: 0.3);
      dot = _ObsidianTheme.amber;
      text = _ObsidianTheme.amberText;
      label = 'CHECKING';
    } else if (connected) {
      bg = _ObsidianTheme.emeraldSoft;
      border = _ObsidianTheme.emerald.withValues(alpha: 0.3);
      dot = _ObsidianTheme.emerald;
      text = _ObsidianTheme.emeraldText;
      label = 'BACKEND ONLINE';
    } else {
      bg = _ObsidianTheme.redSoft;
      border = _ObsidianTheme.red.withValues(alpha: 0.3);
      dot = _ObsidianTheme.red;
      text = _ObsidianTheme.redText;
      label = 'OFFLINE';
    }

    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: compact ? 8 : 10,
        vertical: compact ? 4 : 5,
      ),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: border, width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 7,
            height: 7,
            decoration: BoxDecoration(color: dot, shape: BoxShape.circle),
          ),
          if (!compact || !connected) ...[
            const SizedBox(width: 6),
            Text(
              label,
              style: TextStyle(
                fontFamily: 'monospace',
                fontSize: 10,
                fontWeight: FontWeight.w800,
                color: text,
                letterSpacing: 0.5,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLEAN SYSTEM SETTINGS SHEET
// ─────────────────────────────────────────────────────────────────────────────

class _SettingsSheet extends StatefulWidget {
  const _SettingsSheet({
    required this.endpointController,
    required this.service,
    required this.health,
    required this.checkingHealth,
    required this.healthError,
    required this.onRefreshHealth,
    required this.onEndpointApplied,
  });

  final TextEditingController endpointController;
  final LiveCallService service;
  final BackendHealth? health;
  final bool checkingHealth;
  final String? healthError;
  final VoidCallback onRefreshHealth;
  final ValueChanged<String> onEndpointApplied;

  @override
  State<_SettingsSheet> createState() => _SettingsSheetState();
}

class _SettingsSheetState extends State<_SettingsSheet> {
  late final TextEditingController _apiController;
  late final TextEditingController _wsHostController;
  late final TextEditingController _wsPortController;
  late bool _simulationMode;
  late bool _denoiserEnabled;

  @override
  void initState() {
    super.initState();
    _apiController = TextEditingController(text: widget.endpointController.text);
    _wsHostController = TextEditingController(text: widget.service.serverHost);
    _wsPortController = TextEditingController(text: '${widget.service.serverPort}');
    _simulationMode = widget.service.useSimulationMode;
    _denoiserEnabled = widget.service.captureService.denoiserEnabled;
  }

  @override
  void dispose() {
    _apiController.dispose();
    _wsHostController.dispose();
    _wsPortController.dispose();
    super.dispose();
  }

  void _applyPreset({
    required String api,
    required String wsHost,
    required int wsPort,
  }) {
    setState(() {
      _apiController.text = api;
      _wsHostController.text = wsHost;
      _wsPortController.text = '$wsPort';
    });
  }

  void _saveAndApply() {
    widget.endpointController.text = _apiController.text.trim();
    final host = _wsHostController.text.trim();
    final port = int.tryParse(_wsPortController.text.trim()) ?? 443;
    widget.service.setServerEndpoint(host, port);
    widget.service.setSimulationMode(_simulationMode);
    widget.service.captureService.denoiserEnabled = _denoiserEnabled;
    widget.onEndpointApplied(widget.endpointController.text);
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    final health = widget.health;
    final isConnected = health != null;

    return Container(
      decoration: const BoxDecoration(
        color: _ObsidianTheme.panel,
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      padding: EdgeInsets.only(
        left: 24,
        right: 24,
        top: 16,
        bottom: MediaQuery.of(context).viewInsets.bottom + 24,
      ),
      child: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 40,
                height: 4,
                margin: const EdgeInsets.only(bottom: 18),
                decoration: BoxDecoration(
                  color: _ObsidianTheme.borderSubtle,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'System Configuration',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                    color: _ObsidianTheme.textPrimary,
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.close_rounded),
                  onPressed: () => Navigator.of(context).pop(),
                ),
              ],
            ),
            const SizedBox(height: 6),
            const Text(
              'Configure REST backend endpoints, live WebSocket relay stream, and offline simulation mode.',
              style: TextStyle(fontSize: 12, color: _ObsidianTheme.textMuted),
            ),
            const SizedBox(height: 18),

            // Endpoint Presets
            Wrap(
              spacing: 8,
              runSpacing: 8,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                const Text(
                  'Presets:',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: _ObsidianTheme.textMuted,
                  ),
                ),
                ActionChip(
                  label: const Text('Cloud VPS (Default)'),
                  labelStyle: const TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: _ObsidianTheme.textPrimary,
                  ),
                  backgroundColor: _ObsidianTheme.panelElevated,
                  side: const BorderSide(color: _ObsidianTheme.border),
                  visualDensity: VisualDensity.compact,
                  onPressed: () => _applyPreset(
                    api: VoiceIntegrityApi.defaultEndpoint,
                    wsHost: 'codequantum.in',
                    wsPort: 443,
                  ),
                ),
                ActionChip(
                  label: const Text('Localhost Dev'),
                  labelStyle: const TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: _ObsidianTheme.textPrimary,
                  ),
                  backgroundColor: _ObsidianTheme.panelElevated,
                  side: const BorderSide(color: _ObsidianTheme.border),
                  visualDensity: VisualDensity.compact,
                  onPressed: () => _applyPreset(
                    api: 'http://127.0.0.1:8000',
                    wsHost: '127.0.0.1',
                    wsPort: 8001,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),

            // REST Backend URL
            TextField(
              controller: _apiController,
              decoration: const InputDecoration(
                labelText: 'REST Backend API Endpoint',
                hintText: 'https://codequantum.in/sih',
                prefixIcon: Icon(Icons.cloud_outlined, size: 18),
                isDense: true,
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),

            // WebSocket Relay Host & Port
            Row(
              children: [
                Expanded(
                  flex: 3,
                  child: TextField(
                    controller: _wsHostController,
                    decoration: const InputDecoration(
                      labelText: 'WebSocket Relay Host',
                      hintText: 'codequantum.in',
                      prefixIcon: Icon(Icons.stream_rounded, size: 18),
                      isDense: true,
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  flex: 2,
                  child: TextField(
                    controller: _wsPortController,
                    keyboardType: TextInputType.number,
                    decoration: const InputDecoration(
                      labelText: 'Port',
                      hintText: '443',
                      isDense: true,
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),

            // Simulation mode toggle
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              dense: true,
              title: const Text(
                'Offline Simulation Mode',
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: _ObsidianTheme.textPrimary,
                ),
              ),
              subtitle: const Text(
                'Generates realistic synthetic attack scores locally without live server connectivity.',
                style: TextStyle(fontSize: 11, color: _ObsidianTheme.textMuted),
              ),
              value: _simulationMode,
              activeThumbColor: _ObsidianTheme.emerald,
              onChanged: (val) => setState(() => _simulationMode = val),
            ),
            const SizedBox(height: 12),

            // Background Noise Cancellation & Fan Squelch toggle
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              dense: true,
              title: const Text(
                'Fan Squelch & Noise Cancellation',
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: _ObsidianTheme.textPrimary,
                ),
              ),
              subtitle: const Text(
                'Hardware telephony DSP + 120Hz HPF & adaptive noise gate. Suppresses fan hum and prevents false clone alerts when silent.',
                style: TextStyle(fontSize: 11, color: _ObsidianTheme.textMuted),
              ),
              value: _denoiserEnabled,
              activeThumbColor: _ObsidianTheme.emerald,
              onChanged: (val) => setState(() => _denoiserEnabled = val),
            ),
            const SizedBox(height: 14),

            // Health Status & Check Button
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: _ObsidianTheme.panelElevated,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: _ObsidianTheme.border),
              ),
              child: Row(
                children: [
                  Icon(
                    isConnected ? Icons.check_circle_rounded : Icons.info_outline_rounded,
                    size: 18,
                    color: isConnected ? _ObsidianTheme.emerald : _ObsidianTheme.amber,
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          isConnected
                              ? 'Model: ${health.decisionLabel} (${health.fusionMode})'
                              : (widget.checkingHealth
                                  ? 'Testing connection...'
                                  : 'Status: Offline / Standby'),
                          style: const TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                            color: _ObsidianTheme.textPrimary,
                          ),
                        ),
                        if (widget.healthError != null)
                          Text(
                            widget.healthError!,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(fontSize: 10, color: _ObsidianTheme.red),
                          ),
                      ],
                    ),
                  ),
                  OutlinedButton(
                    onPressed: widget.checkingHealth ? null : widget.onRefreshHealth,
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                      minimumSize: const Size(0, 32),
                      textStyle: const TextStyle(fontSize: 11, fontWeight: FontWeight.w700),
                    ),
                    child: const Text('Test Health'),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Save / Apply Button
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: _saveAndApply,
                icon: const Icon(Icons.check_rounded, size: 18),
                label: const Text('Apply Configuration'),
                style: FilledButton.styleFrom(
                  backgroundColor: _ObsidianTheme.emerald,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
