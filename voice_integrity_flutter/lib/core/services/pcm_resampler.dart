import 'dart:math' as math;
import 'dart:typed_data';

/// Utility class for PCM audio processing, resampling, RMS level calculation,
/// and RIFF-WAV encapsulation.
class PcmResampler {
  const PcmResampler._();

  /// Resample and convert 16-bit PCM bytes to mono 16kHz PCM bytes.
  /// If already 16kHz mono, returns the input bytes directly.
  static Uint8List processTo16kMono({
    required Uint8List inputBytes,
    required int inputSampleRate,
    required int inputChannels,
    int targetSampleRate = 16000,
  }) {
    if (inputBytes.isEmpty) return Uint8List(0);

    // If already 16k mono 16-bit, return as-is
    if (inputSampleRate == targetSampleRate && inputChannels == 1) {
      return inputBytes;
    }

    final byteData = ByteData.sublistView(inputBytes);
    final totalSamples = inputBytes.length ~/ 2;
    final samplesPerChannel = totalSamples ~/ inputChannels;

    if (samplesPerChannel == 0) return Uint8List(0);

    // Step 1: Decode to mono Int16List
    final monoSamples = Int16List(samplesPerChannel);
    if (inputChannels == 1) {
      for (var i = 0; i < samplesPerChannel; i++) {
        monoSamples[i] = byteData.getInt16(i * 2, Endian.little);
      }
    } else {
      // Stereo/multichannel: average channels to mono
      for (var i = 0; i < samplesPerChannel; i++) {
        var sum = 0;
        for (var ch = 0; ch < inputChannels; ch++) {
          final sampleIdx = (i * inputChannels + ch) * 2;
          if (sampleIdx + 1 < inputBytes.length) {
            sum += byteData.getInt16(sampleIdx, Endian.little);
          }
        }
        monoSamples[i] = (sum ~/ inputChannels).clamp(-32768, 32767);
      }
    }

    // Step 2: Resample to targetSampleRate if rates differ
    if (inputSampleRate == targetSampleRate) {
      return Uint8List.view(monoSamples.buffer);
    }

    final ratio = inputSampleRate / targetSampleRate;
    final targetLength = (samplesPerChannel / ratio).floor();
    final resampled = Int16List(targetLength);

    for (var j = 0; j < targetLength; j++) {
      final sourcePos = j * ratio;
      final idx = sourcePos.floor();
      final frac = sourcePos - idx;

      if (idx + 1 < samplesPerChannel) {
        final s0 = monoSamples[idx];
        final s1 = monoSamples[idx + 1];
        final interpolated = s0 + (s1 - s0) * frac;
        resampled[j] = interpolated.round().clamp(-32768, 32767);
      } else if (idx < samplesPerChannel) {
        resampled[j] = monoSamples[idx];
      }
    }

    return Uint8List.view(resampled.buffer);
  }

  /// Calculates the normalized RMS level (0.0 to 1.0) of a 16-bit PCM buffer.
  static double calculateRmsLevel(Uint8List pcmBytes) {
    if (pcmBytes.length < 2) return 0.0;

    final byteData = ByteData.sublistView(pcmBytes);
    final sampleCount = pcmBytes.length ~/ 2;
    if (sampleCount == 0) return 0.0;

    var sumSquares = 0.0;
    for (var i = 0; i < sampleCount; i++) {
      final sample = byteData.getInt16(i * 2, Endian.little);
      sumSquares += sample * sample;
    }

    final rms = math.sqrt(sumSquares / sampleCount);
    // Normalize against max 16-bit int (32768) and amplify for UI visual responsiveness
    final normalized = (rms / 32768.0) * 3.5;
    return normalized.clamp(0.0, 1.0);
  }

  /// Encapsulates raw 16-bit PCM bytes with a standard 44-byte RIFF WAV header
  /// so that any platform audio player can decode and play the buffer.
  static Uint8List createWavContainer({
    required Uint8List pcmData,
    int sampleRate = 16000,
    int channels = 1,
    int bitsPerSample = 16,
  }) {
    final byteRate = sampleRate * channels * (bitsPerSample ~/ 8);
    final blockAlign = channels * (bitsPerSample ~/ 8);
    final totalDataLen = pcmData.length;
    final totalAudioLen = totalDataLen + 36;

    final header = Uint8List(44);
    final b = ByteData.view(header.buffer);

    // RIFF header
    b.setUint8(0, 0x52); // 'R'
    b.setUint8(1, 0x49); // 'I'
    b.setUint8(2, 0x46); // 'F'
    b.setUint8(3, 0x46); // 'F'
    b.setUint32(4, totalAudioLen, Endian.little);
    b.setUint8(8, 0x57); // 'W'
    b.setUint8(9, 0x41); // 'A'
    b.setUint8(10, 0x56); // 'V'
    b.setUint8(11, 0x45); // 'E'

    // fmt subchunk
    b.setUint8(12, 0x66); // 'f'
    b.setUint8(13, 0x6D); // 'm'
    b.setUint8(14, 0x74); // 't'
    b.setUint8(15, 0x20); // ' '
    b.setUint32(16, 16, Endian.little); // Subchunk1Size (16 for PCM)
    b.setUint16(20, 1, Endian.little); // AudioFormat (1 for PCM)
    b.setUint16(22, channels, Endian.little);
    b.setUint32(24, sampleRate, Endian.little);
    b.setUint32(28, byteRate, Endian.little);
    b.setUint16(32, blockAlign, Endian.little);
    b.setUint16(34, bitsPerSample, Endian.little);

    // data subchunk
    b.setUint8(36, 0x64); // 'd'
    b.setUint8(37, 0x61); // 'a'
    b.setUint8(38, 0x74); // 't'
    b.setUint8(39, 0x61); // 'a'
    b.setUint32(40, totalDataLen, Endian.little);

    final out = Uint8List(44 + totalDataLen);
    out.setRange(0, 44, header);
    out.setRange(44, 44 + totalDataLen, pcmData);
    return out;
  }
}
