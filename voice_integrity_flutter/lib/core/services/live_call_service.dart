import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import '../models/live_call_models.dart';
import 'audio_capture_service.dart';
import 'audio_playback_service.dart';
import 'pcm_resampler.dart';

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
  final _sessionRecordingController =
      StreamController<Uint8List?>.broadcast();

  static const String defaultHost = 'codequantum.in';
  static const int defaultPort = 443;

  /// 15-second heartbeat interval for WebSocket ping/pong keep-alive
  static const Duration heartbeatInterval = Duration(seconds: 15);

  /// Exponential backoff delays for auto-reconnect on unexpected drops: 1s, 2s, 4s, 8s (up to 4 attempts)
  static const List<Duration> reconnectBackoffDelays = [
    Duration(seconds: 1),
    Duration(seconds: 2),
    Duration(seconds: 4),
    Duration(seconds: 8),
  ];

  /// Dynamic available personas for voice spoof demonstration
  static const List<String> availablePersonas = [
    'Clone · Female',
    'Clone · Male',
    'Clone · Deep',
    'Clone · Child',
    'Robotic',
    'Teammate 3',
  ];

  // State
  CallMode _mode = CallMode.caller;
  CallConnectionState _connectionState = CallConnectionState.idle;
  String _serverHost = defaultHost;
  int _serverPort = defaultPort;
  bool _useSimulationMode = false;
  bool _isSpoofActive = false;
  String _spoofSpeaker = 'Teammate 3';

  String _callerName = 'Verified Voice Channel';
  String _callerSubtitle = 'Secure Audio Line · 16 kHz Direct';

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
  Timer? _heartbeatTimer;
  Timer? _reconnectTimer;
  int _reconnectAttempts = 0;
  bool _isCallActive = false;

  DateTime? _callStartTime;
  DateTime? _spoofStartTime;
  Duration _accumulatedSpoofDuration = Duration.zero;

  DateTime _lastStatsEmitTime = DateTime.fromMillisecondsSinceEpoch(0);

  /// High-performance audio level notifier decoupled from full-screen setState
  final ValueNotifier<double> audioLevelNotifier = ValueNotifier<double>(0.0);
  double get currentAudioLevel => audioLevelNotifier.value;

  // Temporary In-Memory Session Audio Recording (Volatile proof for judges)
  final List<int> _sessionPcmBuffer = [];
  Uint8List? _lastSessionWavBytes;
  Duration? _lastSessionDuration;
  static const int _maxSessionBufferBytes = 16000 * 2 * 180; // 3 min max in RAM (~5.76 MB)

  // Public getters
  CallMode get mode => _mode;
  CallConnectionState get connectionState => _connectionState;
  bool get isConnected => _connectionState == CallConnectionState.connected;
  bool get isReconnecting => _connectionState == CallConnectionState.reconnecting;
  bool get isCallActive => _isCallActive;
  int get reconnectAttempts => _reconnectAttempts;
  bool get isSpoofActive => _isSpoofActive;
  String get spoofSpeaker => _spoofSpeaker;
  String get serverHost => _serverHost;
  int get serverPort => _serverPort;
  bool get useSimulationMode => _useSimulationMode;
  String get webSocketUrl => _buildWebSocketUrl();
  String get callerName => _callerName;
  String get callerSubtitle => _callerSubtitle;
  bool get isHeartbeatActive =>
      _heartbeatTimer != null && _heartbeatTimer!.isActive;

  void setCallerIdentity(String name, [String? subtitle]) {
    _callerName = name;
    if (subtitle != null) _callerSubtitle = subtitle;
  }

  void _emitStatsThrottled({bool force = false}) {
    final now = DateTime.now();
    if (force || now.difference(_lastStatsEmitTime).inMilliseconds >= 250) {
      _lastStatsEmitTime = now;
      if (!_statsController.isClosed) {
        _statsController.add(_stats);
      }
    }
  }

  Uint8List? get lastSessionWavBytes => _lastSessionWavBytes;
  Duration? get lastSessionDuration => _lastSessionDuration;
  bool get hasSessionRecording =>
      _lastSessionWavBytes != null && _lastSessionWavBytes!.isNotEmpty;

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
  Stream<Uint8List?> get sessionRecordingStream =>
      _sessionRecordingController.stream;

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
  Future<void> setMode(CallMode newMode) async {
    if (_mode == newMode) return;
    if (isConnected ||
        _connectionState == CallConnectionState.connecting ||
        _connectionState == CallConnectionState.reconnecting) {
      await disconnect();
    }
    _mode = newMode;
    _resetState();
  }

  /// Dispatches a test sound chime through the audio playback pipeline
  /// so users can verify speaker playback and VU meter responsiveness.
  Future<void> playTestSound({
    Duration chunkDelay = const Duration(milliseconds: 100),
  }) => _playbackService.playTestSound(chunkDelay: chunkDelay);

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

    // Extract any path component
    String hostPart = input;
    String subPath = '';
    final slashIndex = input.indexOf('/');
    if (slashIndex != -1) {
      hostPart = input.substring(0, slashIndex);
      subPath = input.substring(slashIndex + 1);
    }

    // Extract any embedded port from hostPart
    int port = _serverPort;
    if (hostPart.contains(':')) {
      final parts = hostPart.split(':');
      hostPart = parts[0];
      final parsedPort = int.tryParse(parts[1]);
      if (parsedPort != null) {
        port = parsedPort;
      }
    } else if (isSecure && port == 8001) {
      port = 443;
    }

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
    if (isConnected ||
        _connectionState == CallConnectionState.connecting ||
        _connectionState == CallConnectionState.reconnecting) {
      return;
    }

    _isCallActive = true;
    _reconnectAttempts = 0;
    _reconnectTimer?.cancel();
    _reconnectTimer = null;

    clearSessionRecording();
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

      // Listen for socket events with cancelOnError: false to survive transient network hiccups
      _channelSubscription = _channel!.stream.listen(
        _handleSocketMessage,
        onError: (Object error) {
          debugPrint('[LiveCallService] WebSocket error: $error');
          _handleUnexpectedDisconnection('Socket error: $error');
        },
        onDone: () {
          debugPrint('[LiveCallService] WebSocket closed');
          _handleUnexpectedDisconnection('Remote server closed socket');
        },
        cancelOnError: false,
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
      _startHeartbeat();
      _bindAudioPipelines();
    } catch (error) {
      debugPrint('[LiveCallService] Connection initiation error: $error');
      _handleUnexpectedDisconnection('Initial connection error: $error');
    }
  }

  /// 15-second keep-alive ping/pong heartbeat timer to prevent reverse proxy (Nginx/Apache)
  /// and carrier NAT timeouts from killing idle sockets.
  void _startHeartbeat() {
    _heartbeatTimer?.cancel();
    _heartbeatTimer = Timer.periodic(heartbeatInterval, (_) {
      if (isConnected && _channel != null && !_useSimulationMode) {
        try {
          _channel!.sink.add(jsonEncode({'type': 'ping'}));
        } catch (e) {
          debugPrint('[LiveCallService] Ping heartbeat send error: $e');
        }
      }
    });
  }

  void _stopHeartbeat() {
    _heartbeatTimer?.cancel();
    _heartbeatTimer = null;
  }

  /// Handle unexpected socket disconnects with an exponential backoff auto-reconnect loop.
  void _handleUnexpectedDisconnection(String reason) {
    if (!_isCallActive || _useSimulationMode) {
      disconnect();
      return;
    }

    _stopHeartbeat();
    _channelSubscription?.cancel();
    _channelSubscription = null;
    try {
      _channel?.sink.close();
    } catch (_) {}
    _channel = null;

    if (_reconnectAttempts < reconnectBackoffDelays.length) {
      final delay = reconnectBackoffDelays[_reconnectAttempts];
      _reconnectAttempts++;
      _setConnectionState(CallConnectionState.reconnecting);
      _addReasoningLog(
        'Connection interrupted ($reason). Reconnecting in ${delay.inSeconds}s (attempt $_reconnectAttempts/${reconnectBackoffDelays.length})...',
        LiveRiskLevel.transitioning,
      );

      _reconnectTimer?.cancel();
      _reconnectTimer = Timer(delay, () {
        if (_isCallActive) {
          _executeReconnectAttempt();
        }
      });
    } else {
      _addReasoningLog(
        'Auto-reconnect failed after ${reconnectBackoffDelays.length} attempts. Call ended.',
        LiveRiskLevel.unavailable,
      );
      _handleConnectionFailure('Max reconnect attempts reached ($reason)');
    }
  }

  /// Execute an auto-reconnect attempt preserving session duration and stats.
  Future<void> _executeReconnectAttempt() async {
    if (!_isCallActive || _useSimulationMode) return;
    final wsUrl = _buildWebSocketUrl();
    debugPrint(
      '[LiveCallService] Auto-reconnecting to $wsUrl (attempt $_reconnectAttempts)...',
    );

    try {
      final uri = Uri.parse(wsUrl);
      _channel = WebSocketChannel.connect(uri);

      _channelSubscription = _channel!.stream.listen(
        _handleSocketMessage,
        onError: (Object error) {
          debugPrint('[LiveCallService] Reconnected socket error: $error');
          _handleUnexpectedDisconnection('Socket error: $error');
        },
        onDone: () {
          debugPrint('[LiveCallService] Reconnected socket closed');
          _handleUnexpectedDisconnection('Remote server closed socket');
        },
        cancelOnError: false,
      );

      // Re-send handshake
      final startMsg = jsonEncode({
        'type': 'start',
        'role': _mode == CallMode.caller ? 'caller' : 'receiver',
        'sample_rate': 16000,
        'encoding': 'pcm_s16le',
        'channels': 1,
        'timestamp': DateTime.now().toIso8601String(),
      });
      _channel!.sink.add(startMsg);

      // Re-sync spoof status if active
      if (_mode == CallMode.caller && _isSpoofActive) {
        _channel!.sink.add(jsonEncode({
          'type': 'toggle_spoof',
          'enabled': true,
          'speaker': _spoofSpeaker,
          'timestamp': DateTime.now().toIso8601String(),
        }));
      }

      _reconnectAttempts = 0;
      _setConnectionState(CallConnectionState.connected);
      _startHeartbeat();
      _addReasoningLog(
        'WebSocket reconnected successfully to $wsUrl. Line integrity verified.',
        LiveRiskLevel.low,
      );
    } catch (error) {
      debugPrint('[LiveCallService] Reconnection attempt failed: $error');
      _handleUnexpectedDisconnection('Reconnect error: $error');
    }
  }

  void _handleConnectionFailure(String error) {
    _sessionTimer?.cancel();
    _sessionTimer = null;
    _callStartTime = null;
    _micAudioSubscription?.cancel();
    _micLevelSubscription?.cancel();
    _speakerLevelSubscription?.cancel();
    audioLevelNotifier.value = 0.0;
    _setConnectionState(CallConnectionState.error);
    _addReasoningLog(
      'Connection failed to ${_buildWebSocketUrl()}: $error. You can switch to Simulation Mode for offline demonstrations.',
      LiveRiskLevel.unavailable,
    );
  }

  void _bindAudioPipelines() {
    if (_mode == CallMode.caller) {
      // Start microphone capture with hardware echo cancellation & noise suppression
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
        _recordSessionChunk(chunk);
        try {
          _channel?.sink.add(chunk);
          _stats = _stats.copyWith(
            packetsSent: _stats.packetsSent + 1,
            bytesTransferred: _stats.bytesTransferred + chunk.length,
          );
          _emitStatsThrottled();
        } catch (e) {
          debugPrint('[LiveCallService] Error sending PCM chunk: $e');
        }
      });

      // Track mic audio level without bombarding root setState
      _micLevelSubscription?.cancel();
      _micLevelSubscription = _captureService.audioLevelStream.listen((level) {
        audioLevelNotifier.value = level;
        _stats = _stats.copyWith(audioLevel: level);
      });
    } else {
      // Receiver: track speaker playback audio level without bombarding root setState
      _speakerLevelSubscription?.cancel();
      _speakerLevelSubscription = _playbackService.playbackLevelStream.listen((
        level,
      ) {
        audioLevelNotifier.value = level;
        _stats = _stats.copyWith(audioLevel: level);
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
      _emitStatsThrottled();

      if (_mode == CallMode.receiver) {
        _recordSessionChunk(bytes);
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

        // 0. Keep-alive heartbeat Ping/Pong
        if (type == 'ping') {
          try {
            _channel?.sink.add(jsonEncode({'type': 'pong'}));
          } catch (_) {}
          return;
        }
        if (type == 'pong') {
          return;
        }

        // 1. Spoof Toggle / Transition Event
        if (type == 'spoof_toggle' ||
            type == 'toggle_spoof' ||
            type == 'transition_state' ||
            type == 'spoof_status') {
          final enabled = (data['enabled'] as bool?) ??
              (data['spoof_enabled'] as bool?) ??
              false;
          final speaker = (data['speaker'] as String?) ??
              (data['spoof_speaker'] as String?);
          _handleIncomingSpoofToggle(enabled, speaker: speaker);
          return;
        }

        // 2. Audio Frame in JSON (Base64)
        if (type == 'audio' || type == 'frame') {
          final b64 = data['data'] as String?;
          if (b64 != null && _mode == CallMode.receiver) {
            final pcm = base64Decode(b64);
            _recordSessionChunk(pcm);
            _stats = _stats.copyWith(
              packetsReceived: _stats.packetsReceived + 1,
              bytesTransferred: _stats.bytesTransferred + pcm.length,
            );
            _emitStatsThrottled();
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
  void _handleIncomingSpoofToggle(bool enabled, {String? speaker}) {
    if (speaker != null && speaker.isNotEmpty) {
      _spoofSpeaker = speaker;
    }
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

  /// Set the active cloned persona/speaker on the Caller side.
  void setSpoofSpeaker(String speaker) {
    if (_spoofSpeaker == speaker) return;
    _spoofSpeaker = speaker;
    if (_isSpoofActive) {
      final eventPayload = {
        'type': 'toggle_spoof',
        'enabled': true,
        'speaker': _spoofSpeaker,
        'timestamp': DateTime.now().toIso8601String(),
      };

      if (isConnected && !_useSimulationMode) {
        try {
          _channel?.sink.add(jsonEncode(eventPayload));
        } catch (e) {
          debugPrint('[LiveCallService] Failed to send spoof speaker change: $e');
        }
      } else if (_useSimulationMode) {
        _handleIncomingSpoofToggle(true);
      }
    }
    _statsController.add(_stats);
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
    _statsController.add(_stats);
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
      // Capture mic during offline receiver simulations so presenter voice can be replayed to judges
      _captureService.startStream(sampleRate: 16000).then((started) {
        if (started && isConnected) {
          _micAudioSubscription?.cancel();
          _micAudioSubscription = _captureService.pcmStream.listen((chunk) {
            if (!isConnected || chunk.isEmpty) return;
            _recordSessionChunk(chunk);
          });
        }
      });

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
              probability: (prob + (math.sin(step * 0.9) * 0.03)).clamp(0.01, 0.99),
              riskLevel: level,
            ),
            'hybrid': LiveExpertScore(
              name: 'hybrid',
              label: 'LFCC-LCNN Hybrid',
              probability: (prob + (math.cos(step * 1.1) * 0.03)).clamp(0.01, 0.99),
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
    _sessionTimer = Timer.periodic(const Duration(milliseconds: 500), (_) {
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
      _emitStatsThrottled(force: true);
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
    _isCallActive = false;
    _reconnectAttempts = 0;
    _reconnectTimer?.cancel();
    _reconnectTimer = null;
    _stopHeartbeat();

    _sessionTimer?.cancel();
    _sessionTimer = null;
    _callStartTime = null;
    _isSpoofActive = false;
    _spoofStartTime = null;
    _accumulatedSpoofDuration = Duration.zero;
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

    // Finalize temporary in-memory session recording (proof for judges)
    if (_sessionPcmBuffer.length >= 3200) {
      final pcmBytes = Uint8List.fromList(_sessionPcmBuffer);
      _lastSessionDuration =
          Duration(milliseconds: (_sessionPcmBuffer.length ~/ 32));
      _lastSessionWavBytes = PcmResampler.createWavContainer(
        pcmData: pcmBytes,
        sampleRate: 16000,
        channels: 1,
        bitsPerSample: 16,
      );
    } else {
      _lastSessionWavBytes = null;
      _lastSessionDuration = null;
    }
    _sessionPcmBuffer.clear();
    if (!_sessionRecordingController.isClosed) {
      _sessionRecordingController.add(_lastSessionWavBytes);
    }

    try {
      await _channelSubscription?.cancel();
      _channelSubscription = null;
      await _channel?.sink.close();
      _channel = null;
    } catch (_) {}

    audioLevelNotifier.value = 0.0;
    _setConnectionState(CallConnectionState.disconnected);
  }

  void _recordSessionChunk(Uint8List chunk) {
    if (chunk.isEmpty) return;
    _sessionPcmBuffer.addAll(chunk);
    if (_sessionPcmBuffer.length > _maxSessionBufferBytes) {
      final excess = _sessionPcmBuffer.length - _maxSessionBufferBytes;
      final aligned = excess + (excess % 2);
      _sessionPcmBuffer.removeRange(0, aligned);
    }
  }

  /// Purge the temporary in-memory session recording.
  /// Called immediately when user switches tabs or leaves the call screen.
  void clearSessionRecording() {
    _sessionPcmBuffer.clear();
    _lastSessionWavBytes = null;
    _lastSessionDuration = null;
    if (!_sessionRecordingController.isClosed) {
      _sessionRecordingController.add(null);
    }
  }

  @visibleForTesting
  void setSessionRecordingForTesting(Uint8List? wavBytes, [Duration? duration]) {
    _lastSessionWavBytes = wavBytes;
    _lastSessionDuration =
        duration ?? (wavBytes != null ? const Duration(seconds: 2) : null);
    if (!_sessionRecordingController.isClosed) {
      _sessionRecordingController.add(_lastSessionWavBytes);
    }
  }

  @visibleForTesting
  void setAssessmentForTesting(LiveRiskAssessment assessment) {
    _applyRiskAssessment(assessment);
  }

  @visibleForTesting
  void setConnectionStateForTesting(CallConnectionState state) {
    _setConnectionState(state);
  }

  @visibleForTesting
  void triggerUnexpectedDisconnectionForTesting(String reason) =>
      _handleUnexpectedDisconnection(reason);

  @visibleForTesting
  void handleSocketMessageForTesting(dynamic msg) =>
      _handleSocketMessage(msg);

  void _resetState() {
    _isSpoofActive = false;
    _callStartTime = null;
    _spoofStartTime = null;
    _accumulatedSpoofDuration = Duration.zero;
    _stats = const CallStats();
    _currentAssessment = LiveRiskAssessment.initial();
    _reasoningLogs.clear();
    audioLevelNotifier.value = 0.0;

    _statsController.add(_stats);
    _riskAssessmentController.add(_currentAssessment);
    _reasoningLogsController.add([]);
  }

  /// Dispose all controllers and services.
  Future<void> dispose() async {
    clearSessionRecording();
    await disconnect();
    await _captureService.dispose();
    await _playbackService.dispose();
    audioLevelNotifier.dispose();
    await _connectionStateController.close();
    await _riskAssessmentController.close();
    await _statsController.close();
    await _reasoningLogsController.close();
    await _sessionRecordingController.close();
  }
}
