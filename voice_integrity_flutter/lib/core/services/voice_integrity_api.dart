import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../models/voice_models.dart';

class VoiceIntegrityApi {
  static const String defaultEndpoint = 'https://codequantum.in/sih';

  VoiceIntegrityApi([String baseUrl = defaultEndpoint])
      : _baseUri = Uri.parse(
          _normalise(baseUrl.trim().isEmpty ? defaultEndpoint : baseUrl),
        );

  final Uri _baseUri;

  Uri get baseUri => _baseUri;

  static String _normalise(String raw) =>
      raw.trim().replaceFirst(RegExp(r'/+$'), '');

  @visibleForTesting
  static String normalise(String raw) => _normalise(raw);

  Uri _uri(String path) {
    final cleanPath = path.startsWith('/') ? path : '/$path';
    final basePath = _baseUri.path.replaceFirst(RegExp(r'/+$'), '');
    final combinedPath = basePath.isEmpty ? cleanPath : '$basePath$cleanPath';
    return _baseUri.replace(path: combinedPath);
  }

  @visibleForTesting
  Uri uri(String path) => _uri(path);

  @visibleForTesting
  Uri buildUri(String path) => _uri(path);

  Future<BackendHealth> fetchHealth() async {
    final response = await http
        .get(_uri('/health'))
        .timeout(const Duration(seconds: 8));
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw VoiceIntegrityApiException(
        'Health check returned HTTP ${response.statusCode}.',
      );
    }
    return BackendHealth.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<AnalysisReport> analyzeAudio({
    required Uint8List bytes,
    required String filename,
  }) async {
    final request = http.MultipartRequest('POST', _uri('/predict-file'))
      ..files.add(
        http.MultipartFile.fromBytes('file', bytes, filename: filename),
      );
    final streamed = await request.send().timeout(const Duration(minutes: 3));
    if (streamed.statusCode < 200 || streamed.statusCode >= 300) {
      throw VoiceIntegrityApiException(
        'Audio analysis returned HTTP ${streamed.statusCode}.',
      );
    }
    final body = await streamed.stream.bytesToString();
    final events = _decodeSse(body);
    final windowEvents = events
        .where((event) => event['event'] == 'window_scored')
        .toList();
    final summary = events
        .where((event) => event['event'] == 'summary')
        .cast<Map<String, dynamic>>()
        .lastOrNull;
    if (summary == null) {
      throw VoiceIntegrityApiException(
        'The backend returned no analysis summary.',
      );
    }
    return _toReport(filename, summary, windowEvents);
  }

  List<Map<String, dynamic>> _decodeSse(String body) {
    final events = <Map<String, dynamic>>[];
    for (final frame in body.split(RegExp(r'\r?\n\r?\n'))) {
      final data = frame
          .split(RegExp(r'\r?\n'))
          .where((line) => line.startsWith('data:'))
          .map((line) => line.substring(5).trim())
          .join('\n');
      if (data.isEmpty) continue;
      final decoded = jsonDecode(data);
      if (decoded is Map<String, dynamic>) events.add(decoded);
    }
    return events;
  }

  AnalysisReport _toReport(
    String filename,
    Map<String, dynamic> summary,
    List<Map<String, dynamic>> events,
  ) {
    final expertRows = summary['experts'];
    final experts = <ExpertScore>[];
    if (expertRows is List) {
      for (final raw in expertRows.whereType<Map>()) {
        final entry = Map<String, dynamic>.from(raw);
        experts.add(
          ExpertScore(
            id: '${entry['name'] ?? ''}',
            label: '${entry['label'] ?? entry['name'] ?? 'Unknown model'}',
            probability: asDouble(entry['probability']),
            riskState: riskFromWire(entry['risk_state']),
          ),
        );
      }
    } else if (expertRows is Map) {
      expertRows.forEach((key, raw) {
        final entry = raw is Map
            ? Map<String, dynamic>.from(raw)
            : <String, dynamic>{};
        experts.add(
          ExpertScore(
            id: '$key',
            label: '${entry['label'] ?? key}',
            probability: asDouble(entry['probability']),
            riskState: riskFromWire(entry['risk_state']),
          ),
        );
      });
    }
    final windows = events.map((event) {
      final rawProbabilities =
          event['per_expert_probability'] as Map? ?? const {};
      return WindowScore(
        index: (event['window_index'] as num?)?.toInt() ?? 0,
        timeSeconds: asDouble(event['start_time_sec']) ?? 0,
        probability:
            asDouble(event['weighted_probability']) ??
            asDouble(event['calibrated_probability']),
        riskState: riskFromWire(event['risk_state']),
        expertProbabilities: rawProbabilities.map<String, double>(
          (key, value) => MapEntry('$key', asDouble(value) ?? 0),
        ),
      );
    }).toList();
    return AnalysisReport(
      sourceLabel: filename,
      riskState: riskFromWire(summary['overall_risk_state']),
      probability:
          asDouble(summary['final_smoothed_probability']) ??
          asDouble(summary['weighted_spoof_probability']),
      experts: experts,
      windows: windows,
      decisionExpert:
          '${summary['decision_expert'] ?? 'Weighted calibrated ensemble'}',
      agreement: '${summary['agreement'] ?? 'Not reported'}',
      confidence: '${summary['confidence_level'] ?? 'Not reported'}',
      suspiciousWindows: (summary['spoof_windows_count'] as num?)?.toInt() ?? 0,
      totalWindows:
          (summary['total_windows_count'] as num?)?.toInt() ?? windows.length,
      peakTimeSeconds: asDouble(summary['peak_time_sec']),
    );
  }
}

class VoiceIntegrityApiException implements Exception {
  VoiceIntegrityApiException(this.message);
  final String message;

  @override
  String toString() => message;
}

extension<T> on Iterable<T> {
  T? get lastOrNull => isEmpty ? null : last;
}
