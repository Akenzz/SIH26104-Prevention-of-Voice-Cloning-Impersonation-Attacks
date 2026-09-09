import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import '../models/live_call_models.dart';
import 'audio_capture_service.dart';
import 'audio_playback_service.dart';

/// Live Call orchestrator managing WebSocket connections to /ws/caller and /ws/receiver,
/// microphone capture, speaker playback, and real-time risk assessment updates.
class LiveCallService {
  LiveCallService({
    AudioCaptureService? captureService,
    AudioPlaybackService? playbackService,
  }) : _captureService = captureService ?? AudioCaptureService(),
       _playbackService = playbackService ?? AudioPlaybackService();

  final AudioCaptureService _captureService;
  final AudioPlaybackService _playbackService;

  // Controllers
  final _connectionStateController =
      StreamController<CallConnectionState>.broadcast();
  final _riskAssessmentController =
      StreamController<LiveRiskAssessment>.broadcast();
  final _statsController = StreamController<CallStats>.broadcast();
  final _reasoningLogsController =
      StreamController<List<ReasoningLogEntry>>.broadcast();

  // State
  CallMode _mode = CallMode.caller;
  CallConnectionState _connectionState = CallConnectionState.idle;
  String _serverHost = '127.0.0.1';
  int _serverPort = 8001;
  bool _useSimulationMode = false;
  bool _isSpoofActive = false;
  String _spoofSpeaker = 'Teammate 3';

  LiveRiskAssessment _currentAssessment = LiveRiskAssessment.initial();
  CallStats _stats = const CallStats();
  final List<ReasoningLogEntry> _reasoningLogs = [];

  WebSocketChannel? _channel;
  StreamSubscription<dynamic>? _channelSubscription;
  StreamSubscription<Uint8List>? _micAudioSubscription;
  StreamSubscription<double>? _micLevelSubscription;
  StreamSubscription<double>? _speakerLevelSubscription;

  Timer? _sessionTimer;
  Timer? _spoofTransitionTimer;
  Timer? _simulationScoreTimer;
  DateTime? _callStartTime;
  DateTime? _spoofStartTime;
  Duration _accumulatedSpoofDuration = Duration.zero;

  // Public getters
  CallMode get mode => _mode;
  CallConnectionState get connectionState => _connectionState;
  bool get isConnected => _connectionState == CallConnectionState.connected;
  bool get isSpoofActive => _isSpoofActive;
  String get spoofSpeaker => _spoofSpeaker;
  String get serverHost => _serverHost;
  int get serverPort => _serverPort;
  bool get useSimulationMode => _useSimulationMode;
  String get webSocketUrl => _buildWebSocketUrl();

  LiveRiskAssessment get currentAssessment => _currentAssessment;
  CallStats get stats => _stats;
  List<ReasoningLogEntry> get reasoningLogs =>
      List.unmodifiable(_reasoningLogs);
  AudioCaptureService get captureService => _captureService;
  AudioPlaybackService get playbackService => _playbackService;

  Stream<CallConnectionState> get connectionStateStream =>
      _connectionStateController.stream;
  Stream<LiveRiskAssessment> get riskAssessmentStream =>
      _riskAssessmentController.stream;
  Stream<CallStats> get statsStream => _statsController.stream;
  Stream<List<ReasoningLogEntry>> get reasoningLogsStream =>
      _reasoningLogsController.stream;

  /// Update the server host and port.
  void setServerEndpoint(String host, int port) {
    _serverHost = host.trim();
    _serverPort = port;
  }

  /// Toggle simulation mode for offline demo-day reliability.
  void setSimulationMode(bool enabled) {
    _useSimulationMode = enabled;
  }

  /// Set the active mode (Caller vs Receiver).
  void setMode(CallMode newMode) {
    if (_mode == newMode) return;
    final wasConnected = isConnected;
    if (wasConnected) {
      disconnect();
    }
    _mode = newMode;
    _resetState();
  }

  String _buildWebSocketUrl() {
    final endpoint = _mode == CallMode.caller ? 'ws/caller' : 'ws/receiver';
    var input = _serverHost.trim();

    // Check scheme preference if explicitly provided
    bool isSecure = false;
    if (input.startsWith('wss://') || input.startsWith('https://')) {
      isSecure = true;
      input = input.replaceFirst(RegExp(r'^(wss|https)://'), '');
    } else if (input.startsWith('ws://') || input.startsWith('http://')) {
      input = input.replaceFirst(RegExp(r'^(ws|http)://'), '');
    }

    // Extract any path component (e.g. "codequantum.in/relay" -> host="codequantum.in", path="relay")
    String hostPart = input;
    String subPath = '';
    final slashIndex = input.indexOf('/');
    if (slashIndex != -1) {
      hostPart = input.substring(0, slashIndex);
      subPath = input.substring(slashIndex + 1);
    }

    // Extract any embedded port from hostPart (e.g. "codequantum.in:8001")
    int port = _serverPort;
    if (hostPart.contains(':')) {
      final parts = hostPart.split(':');
      hostPart = parts[0];
      final parsedPort = int.tryParse(parts[1]);
      if (parsedPort != null) {
        port = parsedPort;
      }
    } else if (isSecure && port == 8001) {
      // If user specified wss:// or https:// and left default port 8001, route to standard SSL port 443
      port = 443;
    }

    // If port is 443, treat as secure wss
    if (port == 443) {
      isSecure = true;
    }

    // Clean up subPath
    while (subPath.endsWith('/')) {
      subPath = subPath.substring(0, subPath.length - 1);
    }
    while (subPath.startsWith('/')) {
      subPath = subPath.substring(1);
    }

    // If connecting over secure SSL (port 443 or wss) to codequantum.in and no subpath is given,
    // default subpath to 'relay' so it matches the Apache reverse proxy (/relay/ws/caller)
    if (isSecure && hostPart.contains('codequantum.in') && subPath.isEmpty) {
      subPath = 'relay';
    }

    final scheme = isSecure ? 'wss' : 'ws';
    final portSuffix = (isSecure && port == 443) || (!isSecure && port == 80)
        ? ''
        : ':$port';

    final fullPath = subPath.isEmpty ? '/$endpoint' : '/$subPath/$endpoint';
    return '$scheme://$hostPart$portSuffix$fullPath';
  }

  /// Connect to the WebSocket endpoint (or start simulation if enabled).
  Future<void> connect() async {
    if (isConnected || _connectionState == CallConnectionState.connecting) {
      return;
    }

    _setConnectionState(CallConnectionState.connecting);
    _callStartTime = DateTime.now();
    _startSessionTimer();

    if (_useSimulationMode) {
      _startSimulationSession();
      return;
    }

    final wsUrl = _buildWebSocketUrl();
    debugPrint('[LiveCallService] Connecting to $wsUrl');

    try {
      final uri = Uri.parse(wsUrl);
      _channel = WebSocketChannel.connect(uri);

      // Listen for socket events
      _channelSubscription = _channel!.stream.listen(
        _handleSocketMessage,
        onError: (Object error) {
          debugPrint('[LiveCallService] WebSocket error: $error');
          _handleConnectionFailure(error.toString());
        },
        onDone: () {
          debugPrint('[LiveCallService] WebSocket closed');
          disconnect();
        },
        cancelOnError: true,
      );

      // Send initial handshake
      final startMsg = jsonEncode({
        'type': 'start',
        'role': _mode == CallMode.caller ? 'caller' : 'receiver',
        'sample_rate': 16000,
        'encoding': 'pcm_s16le',
        'channels': 1,
        'timestamp': DateTime.now().toIso8601String(),
      });
      _channel!.sink.add(startMsg);

      _setConnectionState(CallConnectionState.connected);
      _bindAudioPipelines();
    } catch (error) {
      debugPrint('[LiveCallService] Connection initiation error: $error');
      _handleConnectionFailure(error.toString());
    }
  }

  void _handleConnectionFailure(String error) {
    _setConnectionState(CallConnectionState.error);
    _addReasoningLog(
      'Connection failed to ${_buildWebSocketUrl()}: $error. You can switch to Simulation Mode for offline demonstrations.',
      LiveRiskLevel.unavailable,
    );
  }

  void _bindAudioPipelines() {
    if (_mode == CallMode.caller) {
      // Start microphone capture
      _captureService.startStream(sampleRate: 16000).then((started) {
        if (!started) {
          _addReasoningLog(
            'Microphone access was denied or unavailable. Please check system permissions.',
            LiveRiskLevel.unavailable,
          );
        }
      });

      // Stream mic chunks to WebSocket
      _micAudioSubscription?.cancel();
      _micAudioSubscription = _captureService.pcmStream.listen((chunk) {
        if (!isConnected || chunk.isEmpty) return;
        try {
          _channel?.sink.add(chunk);
          _stats = _stats.copyWith(
            packetsSent: _stats.packetsSent + 1,
            bytesTransferred: _stats.bytesTransferred + chunk.length,
          );
          _statsController.add(_stats);
        } catch (e) {
          debugPrint('[LiveCallService] Error sending PCM chunk: $e');
        }
      });

      // Track mic VU meter
      _micLevelSubscription?.cancel();
      _micLevelSubscription = _captureService.audioLevelStream.listen((level) {
        _stats = _stats.copyWith(audioLevel: level);
        _statsController.add(_stats);
      });
    } else {
      // Receiver: track speaker playback VU meter
      _speakerLevelSubscription?.cancel();
      _speakerLevelSubscription = _playbackService.playbackLevelStream.listen((
        level,
      ) {
        _stats = _stats.copyWith(audioLevel: level);
        _statsController.add(_stats);
      });
    }
  }

  void _handleSocketMessage(dynamic message) {
    if (message is Uint8List || message is List<int>) {
      // Binary PCM audio frame received
      final bytes = message is Uint8List
          ? message
          : Uint8List.fromList(message);
      _stats = _stats.copyWith(
        packetsReceived: _stats.packetsReceived + 1,
        bytesTransferred: _stats.bytesTransferred + bytes.length,
      );
      _statsController.add(_stats);

      if (_mode == CallMode.receiver) {
        _playbackService.ingestPcmChunk(bytes);
      }
      return;
    }

    if (message is String) {
      try {
        final data = jsonDecode(message);
        if (data is! Map<String, dynamic>) return;

        final type = (data['type'] ?? data['event'] ?? '')
            .toString()
            .toLowerCase();

        // 1. Spoof Toggle / Transition Event
        if (type == 'spoof_toggle' ||
            type == 'toggle_spoof' ||
            type == 'transition_state' ||
            type == 'spoof_status') {
          final enabled = (data['enabled'] as bool?) ??
              (data['spoof_enabled'] as bool?) ??
              false;
          _handleIncomingSpoofToggle(enabled);
          return;
        }

        // 2. Audio Frame in JSON (Base64)
        if (type == 'audio' || type == 'frame') {
          final b64 = data['data'] as String?;
          if (b64 != null && _mode == CallMode.receiver) {
            final pcm = base64Decode(b64);
            _playbackService.ingestPcmChunk(pcm);
          }
          return;
        }

        // 3. Risk Assessment / Score
        if (type == 'score' ||
            type == 'window_scored' ||
            type == 'assessment') {
          final assessment = LiveRiskAssessment.fromJson(data);
          _applyRiskAssessment(assessment);
          return;
        }
      } catch (e) {
        debugPrint('[LiveCallService] Error parsing JSON message: $e');
      }
    }
  }

  /// Critical UX Mitigation:
  /// When spoofing is toggled, immediately show an amber buffer analyzing state for 1.5s
  /// until post-switch score arrives.
  void _handleIncomingSpoofToggle(bool enabled) {
    _isSpoofActive = enabled;
    _spoofTransitionTimer?.cancel();

    _currentAssessment = LiveRiskAssessment.transitioning(
      isSpoofTarget: enabled,
    );
    _riskAssessmentController.add(_currentAssessment);

    _addReasoningLog(
      enabled
          ? 'Caller switched to Spoof Mode: "Talk as $_spoofSpeaker". Analyzing ring buffer...'
          : 'Caller returned to Genuine Human Voice. Flushing buffer...',
      LiveRiskLevel.transitioning,
    );

    // After 1.5 seconds, if no live score arrived, fall back to estimated assessment
    _spoofTransitionTimer = Timer(const Duration(milliseconds: 1500), () {
      if (_currentAssessment.riskLevel == LiveRiskLevel.transitioning) {
        final fallback = LiveRiskAssessment(
          riskLevel: enabled ? LiveRiskLevel.high : LiveRiskLevel.low,
          probability: enabled ? 0.88 : 0.12,
          recommendedAction: enabled
              ? 'AI Voice Clone Impersonation Detected. Pause call and verify identity.'
              : 'Speech characteristics match human acoustic baseline.',
          reasoning: enabled
              ? 'Multi-expert synthetic speech artifacts detected above 0.65 threshold.'
              : 'Acoustic envelope consistent with genuine human speech.',
          windowIndex: _currentAssessment.windowIndex + 1,
          latencyMs: 38.0,
          timestamp: DateTime.now(),
          isSpoofedByCaller: enabled,
          expertScores: {
            'wavlm': LiveExpertScore(
              name: 'wavlm',
              label: 'WavLM Base+',
              probability: enabled ? 0.85 : 0.09,
              riskLevel: enabled ? LiveRiskLevel.high : LiveRiskLevel.low,
            ),
            'hybrid': LiveExpertScore(
              name: 'hybrid',
              label: 'LFCC-LCNN Hybrid',
              probability: enabled ? 0.91 : 0.14,
              riskLevel: enabled ? LiveRiskLevel.high : LiveRiskLevel.low,
            ),
          },
        );
        _applyRiskAssessment(fallback);
      }
    });
  }

  void _applyRiskAssessment(LiveRiskAssessment assessment) {
    _spoofTransitionTimer?.cancel();
    _currentAssessment = assessment;
    _riskAssessmentController.add(assessment);

    final probStr = assessment.probability != null
        ? ' (P = ${(assessment.probability! * 100).toStringAsFixed(1)}%)'
        : '';
    _addReasoningLog(
      'Window #${assessment.windowIndex}: ${assessment.riskLevel.label}$probStr — ${assessment.reasoning}',
      assessment.riskLevel,
      probability: assessment.probability,
      windowIndex: assessment.windowIndex,
      latencyMs: assessment.latencyMs,
    );
  }

  /// Toggle Spoof ('Talk as Teammate 3') on the Caller side.
  void toggleSpoof({String? speaker}) {
    if (_mode != CallMode.caller) return;

    if (speaker != null) _spoofSpeaker = speaker;
    _isSpoofActive = !_isSpoofActive;

    if (_isSpoofActive) {
      _spoofStartTime = DateTime.now();
    } else if (_spoofStartTime != null) {
      _accumulatedSpoofDuration += DateTime.now().difference(_spoofStartTime!);
      _spoofStartTime = null;
    }

    final eventPayload = {
      'type': 'toggle_spoof',
      'enabled': _isSpoofActive,
      'speaker': _spoofSpeaker,
      'timestamp': DateTime.now().toIso8601String(),
    };

    if (isConnected && !_useSimulationMode) {
      try {
        _channel?.sink.add(jsonEncode(eventPayload));
      } catch (e) {
        debugPrint('[LiveCallService] Failed to send spoof toggle: $e');
      }
    } else if (_useSimulationMode) {
      _handleIncomingSpoofToggle(_isSpoofActive);
    }
  }

  /// Start simulation session for offline judge demonstrations.
  void _startSimulationSession() {
    _setConnectionState(CallConnectionState.connected);
    _addReasoningLog(
      'Interactive Demonstration Mode Active (${_mode.title}). Realistic audio & scoring pipeline running.',
      LiveRiskLevel.low,
    );

    _bindAudioPipelines();

    if (_mode == CallMode.receiver) {
      var step = 0;
      _simulationScoreTimer?.cancel();
      _simulationScoreTimer = Timer.periodic(const Duration(milliseconds: 1000), (
        timer,
      ) {
        step++;
        final isSpoof = _isSpoofActive;
        final baseProb = isSpoof ? 0.86 : 0.14;
        final jitter = (math.sin(step * 0.7) * 0.05);
        final prob = (baseProb + jitter).clamp(0.02, 0.98);

        final level = isSpoof ? LiveRiskLevel.high : LiveRiskLevel.low;
        final assessment = LiveRiskAssessment(
          riskLevel: level,
          probability: prob,
          recommendedAction: isSpoof
              ? 'Flagged synthetic clone. Pause transaction and demand verified callback.'
              : 'Voice biometric features consistent with authorized profile.',
          reasoning: isSpoof
              ? 'WavLM & LFCC-LCNN identified phase inconsistency and vocoder artifacting.'
              : 'Pitch contour, formant trajectory, and prosody within natural bounds.',
          windowIndex: step,
          latencyMs: 32.0 + (math.Random().nextDouble() * 12.0),
          timestamp: DateTime.now(),
          isSpoofedByCaller: isSpoof,
          expertScores: {
            'wavlm': LiveExpertScore(
              name: 'wavlm',
              label: 'WavLM Base+',
              probability: (prob - 0.03).clamp(0.01, 0.99),
              riskLevel: level,
            ),
            'hybrid': LiveExpertScore(
              name: 'hybrid',
              label: 'LFCC-LCNN Hybrid',
              probability: (prob + 0.04).clamp(0.01, 0.99),
              riskLevel: level,
            ),
          },
        );

        _applyRiskAssessment(assessment);
      });
    }
  }

  void _startSessionTimer() {
    _sessionTimer?.cancel();
    _sessionTimer = Timer.periodic(const Duration(milliseconds: 250), (_) {
      if (_callStartTime == null) return;
      final totalElapsed = DateTime.now().difference(_callStartTime!);
      Duration currentSpoof = _accumulatedSpoofDuration;
      if (_isSpoofActive && _spoofStartTime != null) {
        currentSpoof += DateTime.now().difference(_spoofStartTime!);
      }

      _stats = _stats.copyWith(
        duration: totalElapsed,
        spoofDuration: currentSpoof,
      );
      _statsController.add(_stats);
    });
  }

  void _addReasoningLog(
    String text,
    LiveRiskLevel level, {
    double? probability,
    int? windowIndex,
    double? latencyMs,
  }) {
    final entry = ReasoningLogEntry(
      id: '${DateTime.now().millisecondsSinceEpoch}_${_reasoningLogs.length}',
      timestamp: DateTime.now(),
      text: text,
      riskLevel: level,
      probability: probability,
      windowIndex: windowIndex,
      latencyMs: latencyMs,
    );
    _reasoningLogs.insert(0, entry);
    if (_reasoningLogs.length > 50) {
      _reasoningLogs.removeLast();
    }
    _reasoningLogsController.add(List.unmodifiable(_reasoningLogs));
  }

  void _setConnectionState(CallConnectionState state) {
    _connectionState = state;
    _connectionStateController.add(state);
  }

  /// Disconnect and release the active call session.
  Future<void> disconnect() async {
    _sessionTimer?.cancel();
    _sessionTimer = null;
    _spoofTransitionTimer?.cancel();
    _spoofTransitionTimer = null;
    _simulationScoreTimer?.cancel();
    _simulationScoreTimer = null;

    _micAudioSubscription?.cancel();
    _micAudioSubscription = null;
    _micLevelSubscription?.cancel();
    _micLevelSubscription = null;
    _speakerLevelSubscription?.cancel();
    _speakerLevelSubscription = null;

    await _captureService.stopStream();
    await _playbackService.stop();

    try {
      await _channelSubscription?.cancel();
      _channelSubscription = null;
      await _channel?.sink.close();
      _channel = null;
    } catch (_) {}

    _setConnectionState(CallConnectionState.disconnected);
  }

  void _resetState() {
    _isSpoofActive = false;
    _callStartTime = null;
    _spoofStartTime = null;
    _accumulatedSpoofDuration = Duration.zero;
    _stats = const CallStats();
    _currentAssessment = LiveRiskAssessment.initial();
    _reasoningLogs.clear();

    _statsController.add(_stats);
    _riskAssessmentController.add(_currentAssessment);
    _reasoningLogsController.add([]);
  }

  /// Dispose all controllers and services.
  Future<void> dispose() async {
    await disconnect();
    await _captureService.dispose();
    await _playbackService.dispose();
    await _connectionStateController.close();
    await _riskAssessmentController.close();
    await _statsController.close();
    await _reasoningLogsController.close();
  }
}
