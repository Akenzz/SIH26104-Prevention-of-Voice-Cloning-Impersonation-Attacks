import 'package:flutter/material.dart';

abstract final class AppColors {
  static const ink = Color(0xFF1E231D);
  static const mutedInk = Color(0xFF62695D);
  static const canvas = Color(0xFFF3EFE6);
  static const paper = Color(0xFFFBF8F1);
  static const line = Color(0xFFD7D1C5);
  static const moss = Color(0xFF285943);
  static const mossSoft = Color(0xFFDDECE3);
  static const amber = Color(0xFFB76A17);
  static const amberSoft = Color(0xFFFFEDCF);
  static const vermilion = Color(0xFFB53A2E);
  static const vermilionSoft = Color(0xFFF9DDD8);
  static const slate = Color(0xFF425266);
  static const unavailable = Color(0xFF6B6B68);
}

abstract final class AppTheme {
  static ThemeData light() {
    final scheme = ColorScheme.fromSeed(
      seedColor: AppColors.moss,
      brightness: Brightness.light,
      surface: AppColors.paper,
      onSurface: AppColors.ink,
    );
    return ThemeData(
      useMaterial3: true,
      colorScheme: scheme,
      scaffoldBackgroundColor: AppColors.canvas,
      fontFamily: 'Helvetica Neue',
      textTheme: const TextTheme(
        displaySmall: TextStyle(
          fontFamily: 'Georgia',
          fontSize: 38,
          height: 1.03,
          fontWeight: FontWeight.w700,
          color: AppColors.ink,
          letterSpacing: -1.3,
        ),
        headlineSmall: TextStyle(
          fontFamily: 'Georgia',
          fontSize: 28,
          height: 1.1,
          fontWeight: FontWeight.w700,
          color: AppColors.ink,
        ),
        titleLarge: TextStyle(
          fontSize: 20,
          fontWeight: FontWeight.w700,
          color: AppColors.ink,
        ),
        titleMedium: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w700,
          color: AppColors.ink,
        ),
        bodyLarge: TextStyle(
          fontSize: 16,
          height: 1.45,
          color: AppColors.mutedInk,
        ),
        bodyMedium: TextStyle(
          fontSize: 16,
          height: 1.42,
          color: AppColors.mutedInk,
        ),
        labelLarge: TextStyle(
          fontSize: 13,
          fontWeight: FontWeight.w700,
          letterSpacing: .2,
        ),
        labelSmall: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w700,
          letterSpacing: 1.1,
        ),
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: AppColors.canvas,
        foregroundColor: AppColors.ink,
        elevation: 0,
        surfaceTintColor: Colors.transparent,
      ),
      cardTheme: CardThemeData(
        elevation: 0,
        margin: EdgeInsets.zero,
        color: AppColors.paper,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
          side: const BorderSide(color: AppColors.line),
        ),
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: AppColors.ink,
          foregroundColor: AppColors.paper,
          minimumSize: const Size(0, 48),
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: AppColors.ink,
          minimumSize: const Size(0, 48),
          side: const BorderSide(color: AppColors.line),
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
        ),
      ),
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          minimumSize: const Size(44, 44),
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: AppColors.paper,
        border: OutlineInputBorder(
          borderSide: const BorderSide(color: AppColors.line),
          borderRadius: BorderRadius.circular(14),
        ),
        enabledBorder: OutlineInputBorder(
          borderSide: const BorderSide(color: AppColors.line),
          borderRadius: BorderRadius.circular(14),
        ),
        focusedBorder: OutlineInputBorder(
          borderSide: const BorderSide(color: AppColors.moss, width: 1.5),
          borderRadius: BorderRadius.circular(14),
        ),
      ),
    );
  }
}
