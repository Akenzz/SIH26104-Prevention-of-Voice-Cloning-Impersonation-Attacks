import 'package:flutter/material.dart';

import 'core/theme/app_theme.dart';
import 'features/dashboard/voice_integrity_shell.dart';

class VoiceIntegrityApp extends StatelessWidget {
  const VoiceIntegrityApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Voice Integrity',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light(),
      home: const VoiceIntegrityShell(),
    );
  }
}
