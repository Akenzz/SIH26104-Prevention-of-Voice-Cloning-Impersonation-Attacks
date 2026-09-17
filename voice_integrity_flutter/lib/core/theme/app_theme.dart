import 'package:flutter/material.dart';

/// Design tokens implementing the Impeccable design system (https://impeccable.style)
/// and anchored directly to the VoiceNow production forensic console (https://voicenow.vercel.app/).
///
/// Principles:
/// - Ban "AI beige" washed backgrounds and random Georgia serifs on tech/cybersecurity tools.
/// - Refuse gradient text, nested cards, and decorative glass/blur.
/// - Deliver high-contrast, authoritative, crisp visual craft.
/// - High-contrast neutral base: Obsidian (#0A0D14), Panel (#121722), Hairline borders (#1E293B / #263345).
/// - Semantic security tokens: Emerald (#10B981), Crimson (#EF4444), Amber (#F59E0B), Blue (#3B82F6).
/// - Modern typography: Inter / SF Pro sans-serif hierarchy; Monospace for forensic telemetry.
abstract final class AppColors {
  // ---------------------------------------------------------------------------
  // Production Web Reference Tokens (voicenow.vercel.app)
  // ---------------------------------------------------------------------------

  /// Primary Dark Obsidian Canvas background (#0A0D14).
  static const darkCanvas = Color(0xFF0A0D14);

  /// Primary Dark Card / Panel surface (#121722).
  static const darkPaper = Color(0xFF121722);

  /// Secondary elevated dark surface (#181F2E).
  static const darkPaperElevated = Color(0xFF181F2E);

  /// Hairline structural border (#1E293B - Slate 800).
  static const darkLine = Color(0xFF1E293B);

  /// Subtle secondary hairline divider (#263345).
  static const darkLineSubtle = Color(0xFF263345);

  /// Dark Primary Text: Zinc-100 (#F4F4F5).
  static const darkInk = Color(0xFFF4F4F5);

  /// Dark Secondary Text: Zinc-400 (#A1A1AA).
  static const darkSecondary = Color(0xFFA1A1AA);

  /// Dark Muted / Tertiary Text: Zinc-500 (#71717A).
  static const darkMutedInk = Color(0xFF71717A);

  // ---------------------------------------------------------------------------
  // Light Palette Tokens (High-Contrast Engineering Canvas)
  // ---------------------------------------------------------------------------

  /// Crisp engineering canvas (Slate 50 - #F8FAFC). Bans "AI beige" muddy wash.
  static const lightCanvas = Color(0xFFF8FAFC);

  /// Clean paper surface for cards and sheets (#FFFFFF).
  static const lightPaper = Color(0xFFFFFFFF);

  /// Subtle hairline border (Slate 200 - #E2E8F0).
  static const lightLine = Color(0xFFE2E8F0);

  /// Deep Ink / Obsidian primary text (#0A0D14).
  static const lightInk = Color(0xFF0A0D14);

  /// Slate 600 secondary text (#475569) — >7:1 contrast on white (WCAG AAA).
  static const lightMutedInk = Color(0xFF475569);

  // ---------------------------------------------------------------------------
  // Core Backward-Compatible Tokens (Used throughout current widgets)
  // ---------------------------------------------------------------------------

  /// Deep Ink / Obsidian: Primary authoritative text and key actions.
  static const ink = Color(0xFF0A0D14);
  static const obsidian = ink;

  /// Secondary neutral for supporting labels, subtitles, and metadata.
  static const mutedInk = Color(0xFF475569);

  /// Crisp canvas background.
  static const canvas = Color(0xFFF8FAFC);

  /// Clean paper surfaces for cards, modals, sheets.
  static const paper = Color(0xFFFFFFFF);

  /// Subtle hairline border.
  static const line = Color(0xFFE2E8F0);

  // ---------------------------------------------------------------------------
  // Semantic Security Tokens (voicenow.vercel.app standard)
  // ---------------------------------------------------------------------------

  /// Emerald-500 (#10B981): Verified Bonafide / Authentic Human Speech.
  static const moss = Color(0xFF10B981);
  static const emerald = moss;

  /// Emerald soft tint: High-clarity badge background that never washes out.
  static const mossSoft = Color(0xFFECFDF5);
  static const emeraldSoft = mossSoft;

  /// Red-500 (#EF4444): Spoof Attack / AI Voice Clone Impersonation Detected.
  static const vermilion = Color(0xFFEF4444);
  static const crimson = vermilion;

  /// Crimson soft tint: Clear alert banner tint.
  static const vermilionSoft = Color(0xFFFEF2F2);
  static const crimsonSoft = vermilionSoft;

  /// Amber-500 (#F59E0B): Uncertain / Evaluating / Buffer Transition.
  static const amber = Color(0xFFF59E0B);

  /// Amber soft tint: Cautionary pill background.
  static const amberSoft = Color(0xFFFFFBEB);

  /// Blue-500 (#3B82F6): Action / Tech Accent / Interactive Focus.
  static const blue = Color(0xFF3B82F6);
  static const action = blue;

  /// Refined Slate (#64748B): Technical telemetry, secondary indicators.
  static const slate = Color(0xFF64748B);
  static const slateSoft = Color(0xFFF1F5F9);

  /// Inactive / Unavailable (#94A3B8): Insufficient audio, disabled gate.
  static const unavailable = Color(0xFF94A3B8);

  // Dark semantic soft tints (for tinted pill/card backgrounds in dark mode)
  static const darkMossSoft = Color(0xFF064E3B);
  static const darkVermilionSoft = Color(0xFF450A0A);
  static const darkAmberSoft = Color(0xFF451A03);
  static const darkBlueSoft = Color(0xFF172554);
}

/// Impeccable theme definition for the Voice Integrity client.
abstract final class AppTheme {
  /// Font family hierarchy: Modern humanist/geometric sans-serif.
  /// Banned: Georgia serifs on cybersecurity/system tooling.
  static const String fontFamily = 'Inter';
  static const List<String> fontFallbacks = [
    '-apple-system',
    'BlinkMacSystemFont',
    'SF Pro Display',
    'SF Pro Text',
    'Segoe UI',
    'Roboto',
    'Helvetica Neue',
    'Arial',
    'sans-serif',
  ];

  /// Monospace font family for forensic logs, logits, spectrogram frequency bins.
  static const String monoFontFamily = 'JetBrains Mono';
  static const List<String> monoFontFallbacks = [
    'SF Mono',
    'Menlo',
    'Monaco',
    'Consolas',
    'Liberation Mono',
    'Courier New',
    'monospace',
  ];

  static TextStyle _font({
    required double fontSize,
    required FontWeight fontWeight,
    double? height,
    double? letterSpacing,
    Color? color,
  }) {
    return TextStyle(
      fontFamily: fontFamily,
      fontFamilyFallback: fontFallbacks,
      fontSize: fontSize,
      fontWeight: fontWeight,
      height: height,
      letterSpacing: letterSpacing,
      color: color,
    );
  }

  /// Monospace text style for forensic logits, audio chunks, and diagnostic data.
  static TextStyle mono({
    double fontSize = 13,
    FontWeight fontWeight = FontWeight.w400,
    double? height,
    double? letterSpacing,
    Color? color,
  }) {
    return TextStyle(
      fontFamily: monoFontFamily,
      fontFamilyFallback: monoFontFallbacks,
      fontSize: fontSize,
      fontWeight: fontWeight,
      height: height,
      letterSpacing: letterSpacing,
      color: color,
    );
  }

  static TextTheme _buildTextTheme(Color primaryColor, Color mutedColor) {
    return TextTheme(
      // Display: Tight, bold, authoritative headlines without oversized bloat.
      displayLarge: _font(
        fontSize: 34,
        fontWeight: FontWeight.w700,
        letterSpacing: -1.1,
        height: 1.15,
        color: primaryColor,
      ),
      displayMedium: _font(
        fontSize: 28,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.8,
        height: 1.2,
        color: primaryColor,
      ),
      displaySmall: _font(
        fontSize: 24,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.6,
        height: 1.2,
        color: primaryColor,
      ),

      // Headline: Structural section anchors.
      headlineLarge: _font(
        fontSize: 22,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.5,
        height: 1.25,
        color: primaryColor,
      ),
      headlineMedium: _font(
        fontSize: 20,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.4,
        height: 1.25,
        color: primaryColor,
      ),
      headlineSmall: _font(
        fontSize: 18,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.3,
        height: 1.3,
        color: primaryColor,
      ),

      // Title: Card headers, dialog banners, item titles.
      titleLarge: _font(
        fontSize: 17,
        fontWeight: FontWeight.w600,
        letterSpacing: -0.2,
        height: 1.35,
        color: primaryColor,
      ),
      titleMedium: _font(
        fontSize: 15,
        fontWeight: FontWeight.w600,
        letterSpacing: -0.1,
        height: 1.4,
        color: primaryColor,
      ),
      titleSmall: _font(
        fontSize: 13,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.0,
        height: 1.4,
        color: primaryColor,
      ),

      // Body: High-legibility running text and explanations.
      bodyLarge: _font(
        fontSize: 15,
        fontWeight: FontWeight.w400,
        letterSpacing: -0.1,
        height: 1.5,
        color: mutedColor,
      ),
      bodyMedium: _font(
        fontSize: 14,
        fontWeight: FontWeight.w400,
        letterSpacing: -0.05,
        height: 1.45,
        color: mutedColor,
      ),
      bodySmall: _font(
        fontSize: 12,
        fontWeight: FontWeight.w400,
        letterSpacing: 0.0,
        height: 1.4,
        color: mutedColor,
      ),

      // Label: Micro-copy, badge tags, uppercase chips, button labels.
      labelLarge: _font(
        fontSize: 13,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.1,
        height: 1.3,
        color: primaryColor,
      ),
      labelMedium: _font(
        fontSize: 12,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.2,
        height: 1.3,
        color: primaryColor,
      ),
      labelSmall: _font(
        fontSize: 11,
        fontWeight: FontWeight.w700,
        letterSpacing: 0.6,
        height: 1.25,
        color: mutedColor,
      ),
    );
  }

  /// Light theme — Crisp Obsidian and Emerald engineering console.
  static ThemeData light() {
    final textTheme = _buildTextTheme(AppColors.ink, AppColors.mutedInk);

    final scheme = const ColorScheme.light(
      primary: AppColors.ink,
      onPrimary: AppColors.paper,
      primaryContainer: AppColors.mossSoft,
      onPrimaryContainer: Color(0xFF065F46),
      secondary: Color(0xFF059669),
      onSecondary: AppColors.paper,
      secondaryContainer: AppColors.mossSoft,
      onSecondaryContainer: Color(0xFF065F46),
      tertiary: AppColors.amber,
      onTertiary: AppColors.paper,
      tertiaryContainer: AppColors.amberSoft,
      onTertiaryContainer: Color(0xFF92400E),
      error: AppColors.vermilion,
      onError: AppColors.paper,
      errorContainer: AppColors.vermilionSoft,
      onErrorContainer: Color(0xFF991B1B),
      surface: AppColors.paper,
      onSurface: AppColors.ink,
      outline: AppColors.line,
      outlineVariant: Color(0xFFF1F5F9),
      shadow: Color(0x0A0B0F17),
    );

    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.light,
      colorScheme: scheme,
      scaffoldBackgroundColor: AppColors.canvas,
      fontFamily: fontFamily,
      fontFamilyFallback: fontFallbacks,
      textTheme: textTheme,
      dividerColor: AppColors.line,

      // App Bar: Seamless canvas integration, zero elevation
      appBarTheme: AppBarTheme(
        backgroundColor: AppColors.canvas,
        foregroundColor: AppColors.ink,
        elevation: 0,
        scrolledUnderElevation: 0,
        surfaceTintColor: Colors.transparent,
        centerTitle: false,
        titleTextStyle: _font(
          fontSize: 18,
          fontWeight: FontWeight.w700,
          color: AppColors.ink,
          letterSpacing: -0.3,
        ),
      ),

      // Cards: Pure white surface, 1px crisp hairline border, zero blur
      cardTheme: const CardThemeData(
        elevation: 0,
        margin: EdgeInsets.zero,
        color: AppColors.paper,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.all(Radius.circular(14)),
          side: BorderSide(color: AppColors.line, width: 1.0),
        ),
      ),

      // Filled Buttons: Authoritative Obsidian with responsive states
      filledButtonTheme: FilledButtonThemeData(
        style: ButtonStyle(
          backgroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.disabled)) {
              return AppColors.line;
            }
            if (states.contains(WidgetState.pressed)) {
              return const Color(0xFF1E293B);
            }
            if (states.contains(WidgetState.hovered)) {
              return const Color(0xFF1E293B);
            }
            return AppColors.ink;
          }),
          foregroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.disabled)) {
              return AppColors.unavailable;
            }
            return AppColors.paper;
          }),
          elevation: WidgetStateProperty.all(0),
          overlayColor: WidgetStateProperty.all(Colors.white.withValues(alpha: 0.08)),
          padding: WidgetStateProperty.all(
            const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          ),
          minimumSize: WidgetStateProperty.all(const Size(0, 48)),
          shape: WidgetStateProperty.all(
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          ),
          textStyle: WidgetStateProperty.all(
            _font(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.1,
            ),
          ),
        ),
      ),

      // Elevated Buttons: Subtle hairline surface with clean responsive lift
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ButtonStyle(
          backgroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.disabled)) {
              return AppColors.canvas;
            }
            return AppColors.paper;
          }),
          foregroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.disabled)) {
              return AppColors.unavailable;
            }
            return AppColors.ink;
          }),
          elevation: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.pressed)) return 1.5;
            if (states.contains(WidgetState.hovered)) return 1.0;
            return 0; // Flat resting state for crisp craft
          }),
          shadowColor: WidgetStateProperty.all(AppColors.ink.withValues(alpha: 0.06)),
          overlayColor: WidgetStateProperty.all(AppColors.ink.withValues(alpha: 0.04)),
          padding: WidgetStateProperty.all(
            const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          ),
          minimumSize: WidgetStateProperty.all(const Size(0, 48)),
          side: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.focused)) {
              return const BorderSide(color: AppColors.ink, width: 1.5);
            }
            return const BorderSide(color: AppColors.line, width: 1.0);
          }),
          shape: WidgetStateProperty.all(
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          ),
          textStyle: WidgetStateProperty.all(
            _font(fontSize: 14, fontWeight: FontWeight.w600, letterSpacing: 0.1),
          ),
        ),
      ),

      // Outlined Buttons: 1px hairline border with crisp hover/active darkening
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: ButtonStyle(
          backgroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.hovered)) {
              return AppColors.canvas;
            }
            if (states.contains(WidgetState.pressed)) {
              return AppColors.line.withValues(alpha: 0.5);
            }
            return Colors.transparent;
          }),
          foregroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.disabled)) {
              return AppColors.unavailable;
            }
            return AppColors.ink;
          }),
          side: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.focused)) {
              return const BorderSide(color: AppColors.ink, width: 1.5);
            }
            if (states.contains(WidgetState.hovered)) {
              return const BorderSide(color: AppColors.mutedInk, width: 1.0);
            }
            return const BorderSide(color: AppColors.line, width: 1.0);
          }),
          padding: WidgetStateProperty.all(
            const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
          ),
          minimumSize: WidgetStateProperty.all(const Size(0, 48)),
          shape: WidgetStateProperty.all(
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          ),
          textStyle: WidgetStateProperty.all(
            _font(fontSize: 14, fontWeight: FontWeight.w600, letterSpacing: 0.1),
          ),
        ),
      ),

      // Text Buttons: Clean compact tap target
      textButtonTheme: TextButtonThemeData(
        style: ButtonStyle(
          foregroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.disabled)) {
              return AppColors.unavailable;
            }
            return AppColors.ink;
          }),
          overlayColor: WidgetStateProperty.all(AppColors.ink.withValues(alpha: 0.05)),
          minimumSize: WidgetStateProperty.all(const Size(44, 44)),
          padding: WidgetStateProperty.all(
            const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          ),
          shape: WidgetStateProperty.all(
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          ),
          textStyle: WidgetStateProperty.all(
            _font(fontSize: 14, fontWeight: FontWeight.w600),
          ),
        ),
      ),

      // Input Decoration: High-contrast focus rings, pure paper fill
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: AppColors.paper,
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        border: OutlineInputBorder(
          borderSide: const BorderSide(color: AppColors.line, width: 1.0),
          borderRadius: BorderRadius.circular(10),
        ),
        enabledBorder: OutlineInputBorder(
          borderSide: const BorderSide(color: AppColors.line, width: 1.0),
          borderRadius: BorderRadius.circular(10),
        ),
        focusedBorder: OutlineInputBorder(
          borderSide: const BorderSide(color: AppColors.moss, width: 2.0),
          borderRadius: BorderRadius.circular(10),
        ),
        errorBorder: OutlineInputBorder(
          borderSide: const BorderSide(color: AppColors.vermilion, width: 1.5),
          borderRadius: BorderRadius.circular(10),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderSide: const BorderSide(color: AppColors.vermilion, width: 2.0),
          borderRadius: BorderRadius.circular(10),
        ),
        labelStyle: _font(
          fontSize: 14,
          fontWeight: FontWeight.w500,
          color: AppColors.mutedInk,
        ),
        hintStyle: _font(
          fontSize: 14,
          fontWeight: FontWeight.w400,
          color: AppColors.slate,
        ),
        errorStyle: _font(
          fontSize: 12,
          fontWeight: FontWeight.w500,
          color: AppColors.vermilion,
        ),
      ),

      // Tab Bar: Crisp underline indicator, no blurry tab switches
      tabBarTheme: TabBarThemeData(
        labelColor: AppColors.ink,
        unselectedLabelColor: AppColors.mutedInk,
        labelStyle: _font(
          fontSize: 14,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.1,
        ),
        unselectedLabelStyle: _font(
          fontSize: 14,
          fontWeight: FontWeight.w500,
          letterSpacing: 0.1,
        ),
        indicatorSize: TabBarIndicatorSize.tab,
        indicator: const UnderlineTabIndicator(
          borderSide: BorderSide(color: AppColors.ink, width: 2.5),
        ),
        dividerColor: AppColors.line,
        dividerHeight: 1.0,
        overlayColor: WidgetStateProperty.all(AppColors.ink.withValues(alpha: 0.04)),
      ),

      // Navigation Bar: Crisp bottom/rail navigation with subtle active indicator
      navigationBarTheme: NavigationBarThemeData(
        backgroundColor: AppColors.paper,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        indicatorColor: AppColors.mossSoft,
        indicatorShape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
          side: const BorderSide(color: AppColors.line, width: 0.5),
        ),
        labelTextStyle: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return _font(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: AppColors.ink,
            );
          }
          return _font(
            fontSize: 12,
            fontWeight: FontWeight.w500,
            color: AppColors.mutedInk,
          );
        }),
        iconTheme: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return const IconThemeData(color: AppColors.moss, size: 22);
          }
          return const IconThemeData(color: AppColors.mutedInk, size: 22);
        }),
      ),

      // Divider: 1px hairline rule
      dividerTheme: const DividerThemeData(
        color: AppColors.line,
        thickness: 1,
        space: 1,
      ),

      // Chips: Crisp rounded tags with hairline border
      chipTheme: ChipThemeData(
        backgroundColor: AppColors.canvas,
        disabledColor: AppColors.canvas,
        selectedColor: AppColors.mossSoft,
        secondarySelectedColor: AppColors.mossSoft,
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
          side: const BorderSide(color: AppColors.line, width: 1.0),
        ),
        labelStyle: _font(
          fontSize: 12,
          fontWeight: FontWeight.w600,
          color: AppColors.ink,
        ),
        secondaryLabelStyle: _font(
          fontSize: 12,
          fontWeight: FontWeight.w600,
          color: AppColors.moss,
        ),
        elevation: 0,
        pressElevation: 0,
      ),

      // Dialog: Paper surface with crisp hairline outline
      dialogTheme: DialogThemeData(
        backgroundColor: AppColors.paper,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.all(Radius.circular(16)),
          side: BorderSide(color: AppColors.line, width: 1.0),
        ),
        titleTextStyle: _font(
          fontSize: 18,
          fontWeight: FontWeight.w700,
          color: AppColors.ink,
          letterSpacing: -0.3,
        ),
        contentTextStyle: _font(
          fontSize: 14,
          fontWeight: FontWeight.w400,
          color: AppColors.mutedInk,
          height: 1.45,
        ),
      ),

      // Tooltip: Crisp high-contrast obsidian badge
      tooltipTheme: TooltipThemeData(
        decoration: BoxDecoration(
          color: AppColors.ink,
          borderRadius: BorderRadius.circular(6),
        ),
        textStyle: _font(
          fontSize: 12,
          fontWeight: FontWeight.w500,
          color: AppColors.paper,
        ),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      ),

      // SnackBar: Authoritative floating notice
      snackBarTheme: SnackBarThemeData(
        backgroundColor: AppColors.ink,
        contentTextStyle: _font(
          fontSize: 14,
          fontWeight: FontWeight.w500,
          color: AppColors.paper,
        ),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  /// Dark theme — Exact Voicenow production console aesthetic (https://voicenow.vercel.app/).
  static ThemeData dark() {
    final textTheme = _buildTextTheme(AppColors.darkInk, AppColors.darkSecondary);

    final scheme = const ColorScheme.dark(
      primary: AppColors.darkInk,
      onPrimary: AppColors.darkCanvas,
      primaryContainer: AppColors.darkMossSoft,
      onPrimaryContainer: Color(0xFFA7F3D0),
      secondary: AppColors.moss,
      onSecondary: AppColors.darkCanvas,
      secondaryContainer: AppColors.darkMossSoft,
      onSecondaryContainer: Color(0xFFA7F3D0),
      tertiary: AppColors.amber,
      onTertiary: AppColors.darkCanvas,
      tertiaryContainer: AppColors.darkAmberSoft,
      onTertiaryContainer: Color(0xFFFDE68A),
      error: AppColors.vermilion,
      onError: Colors.white,
      errorContainer: AppColors.darkVermilionSoft,
      onErrorContainer: Color(0xFFFECACA),
      surface: AppColors.darkPaper,
      surfaceContainerHighest: AppColors.darkPaperElevated,
      onSurface: AppColors.darkInk,
      onSurfaceVariant: AppColors.darkSecondary,
      outline: AppColors.darkLine,
      outlineVariant: AppColors.darkLineSubtle,
      shadow: Colors.black,
    );

    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      colorScheme: scheme,
      scaffoldBackgroundColor: AppColors.darkCanvas,
      fontFamily: fontFamily,
      fontFamilyFallback: fontFallbacks,
      textTheme: textTheme,
      dividerColor: AppColors.darkLine,

      // App Bar: Seamless dark obsidian canvas
      appBarTheme: AppBarTheme(
        backgroundColor: AppColors.darkCanvas,
        foregroundColor: AppColors.darkInk,
        elevation: 0,
        scrolledUnderElevation: 0,
        surfaceTintColor: Colors.transparent,
        centerTitle: false,
        titleTextStyle: _font(
          fontSize: 18,
          fontWeight: FontWeight.w700,
          color: AppColors.darkInk,
          letterSpacing: -0.3,
        ),
      ),

      // Cards: Panel surface #121722 with hairline border #1E293B
      cardTheme: const CardThemeData(
        elevation: 0,
        margin: EdgeInsets.zero,
        color: AppColors.darkPaper,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.all(Radius.circular(14)),
          side: BorderSide(color: AppColors.darkLine, width: 1.0),
        ),
      ),

      // Filled Buttons: High-contrast Zinc-100 on dark canvas with subtle hover
      filledButtonTheme: FilledButtonThemeData(
        style: ButtonStyle(
          backgroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.disabled)) {
              return AppColors.darkLine;
            }
            if (states.contains(WidgetState.pressed)) {
              return const Color(0xFFE4E4E7);
            }
            if (states.contains(WidgetState.hovered)) {
              return Colors.white;
            }
            return AppColors.darkInk;
          }),
          foregroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.disabled)) {
              return AppColors.darkMutedInk;
            }
            return AppColors.darkCanvas;
          }),
          elevation: WidgetStateProperty.all(0),
          overlayColor: WidgetStateProperty.all(Colors.black.withValues(alpha: 0.1)),
          padding: WidgetStateProperty.all(
            const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          ),
          minimumSize: WidgetStateProperty.all(const Size(0, 48)),
          shape: WidgetStateProperty.all(
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          ),
          textStyle: WidgetStateProperty.all(
            _font(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.1,
            ),
          ),
        ),
      ),

      // Elevated Buttons: Dark elevated surface with 1px hairline border
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ButtonStyle(
          backgroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.disabled)) {
              return AppColors.darkCanvas;
            }
            if (states.contains(WidgetState.hovered)) {
              return const Color(0xFF1F293D);
            }
            return AppColors.darkPaperElevated;
          }),
          foregroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.disabled)) {
              return AppColors.darkMutedInk;
            }
            return AppColors.darkInk;
          }),
          elevation: WidgetStateProperty.all(0),
          padding: WidgetStateProperty.all(
            const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          ),
          minimumSize: WidgetStateProperty.all(const Size(0, 48)),
          side: WidgetStateProperty.all(
            const BorderSide(color: AppColors.darkLine, width: 1.0),
          ),
          shape: WidgetStateProperty.all(
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          ),
          textStyle: WidgetStateProperty.all(
            _font(fontSize: 14, fontWeight: FontWeight.w600, letterSpacing: 0.1),
          ),
        ),
      ),

      // Outlined Buttons: Hairline border #1E293B with hover lift
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: ButtonStyle(
          backgroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.hovered)) {
              return AppColors.darkPaperElevated;
            }
            return Colors.transparent;
          }),
          foregroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.disabled)) {
              return AppColors.darkMutedInk;
            }
            return AppColors.darkInk;
          }),
          side: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.focused)) {
              return const BorderSide(color: AppColors.blue, width: 1.5);
            }
            if (states.contains(WidgetState.hovered)) {
              return const BorderSide(color: AppColors.darkLineSubtle, width: 1.0);
            }
            return const BorderSide(color: AppColors.darkLine, width: 1.0);
          }),
          padding: WidgetStateProperty.all(
            const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
          ),
          minimumSize: WidgetStateProperty.all(const Size(0, 48)),
          shape: WidgetStateProperty.all(
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          ),
          textStyle: WidgetStateProperty.all(
            _font(fontSize: 14, fontWeight: FontWeight.w600, letterSpacing: 0.1),
          ),
        ),
      ),

      // Text Buttons: Clean compact tap target
      textButtonTheme: TextButtonThemeData(
        style: ButtonStyle(
          foregroundColor: WidgetStateProperty.resolveWith((states) {
            if (states.contains(WidgetState.disabled)) {
              return AppColors.darkMutedInk;
            }
            return AppColors.darkInk;
          }),
          overlayColor: WidgetStateProperty.all(Colors.white.withValues(alpha: 0.05)),
          minimumSize: WidgetStateProperty.all(const Size(44, 44)),
          padding: WidgetStateProperty.all(
            const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          ),
          shape: WidgetStateProperty.all(
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          ),
          textStyle: WidgetStateProperty.all(
            _font(fontSize: 14, fontWeight: FontWeight.w600),
          ),
        ),
      ),

      // Input Decoration: Dark panel fill #121722 with sharp Emerald focus ring
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: AppColors.darkPaper,
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        border: OutlineInputBorder(
          borderSide: const BorderSide(color: AppColors.darkLine, width: 1.0),
          borderRadius: BorderRadius.circular(10),
        ),
        enabledBorder: OutlineInputBorder(
          borderSide: const BorderSide(color: AppColors.darkLine, width: 1.0),
          borderRadius: BorderRadius.circular(10),
        ),
        focusedBorder: OutlineInputBorder(
          borderSide: const BorderSide(color: AppColors.moss, width: 2.0),
          borderRadius: BorderRadius.circular(10),
        ),
        errorBorder: OutlineInputBorder(
          borderSide: const BorderSide(color: AppColors.vermilion, width: 1.5),
          borderRadius: BorderRadius.circular(10),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderSide: const BorderSide(color: AppColors.vermilion, width: 2.0),
          borderRadius: BorderRadius.circular(10),
        ),
        labelStyle: _font(
          fontSize: 14,
          fontWeight: FontWeight.w500,
          color: AppColors.darkSecondary,
        ),
        hintStyle: _font(
          fontSize: 14,
          fontWeight: FontWeight.w400,
          color: AppColors.darkMutedInk,
        ),
        errorStyle: _font(
          fontSize: 12,
          fontWeight: FontWeight.w500,
          color: AppColors.vermilion,
        ),
      ),

      // Tab Bar: Crisp underline indicator, zinc typography
      tabBarTheme: TabBarThemeData(
        labelColor: AppColors.darkInk,
        unselectedLabelColor: AppColors.darkMutedInk,
        labelStyle: _font(
          fontSize: 14,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.1,
        ),
        unselectedLabelStyle: _font(
          fontSize: 14,
          fontWeight: FontWeight.w500,
          letterSpacing: 0.1,
        ),
        indicatorSize: TabBarIndicatorSize.tab,
        indicator: const UnderlineTabIndicator(
          borderSide: BorderSide(color: AppColors.blue, width: 2.5),
        ),
        dividerColor: AppColors.darkLine,
        dividerHeight: 1.0,
      ),

      // Navigation Bar: Dark panel dock with crisp active indicator
      navigationBarTheme: NavigationBarThemeData(
        backgroundColor: AppColors.darkPaper,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        indicatorColor: AppColors.darkMossSoft,
        indicatorShape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
          side: const BorderSide(color: AppColors.darkLineSubtle, width: 0.5),
        ),
        labelTextStyle: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return _font(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: AppColors.darkInk,
            );
          }
          return _font(
            fontSize: 12,
            fontWeight: FontWeight.w500,
            color: AppColors.darkSecondary,
          );
        }),
        iconTheme: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return const IconThemeData(color: AppColors.moss, size: 22);
          }
          return const IconThemeData(color: AppColors.darkSecondary, size: 22);
        }),
      ),

      // Divider: 1px hairline rule in #1E293B
      dividerTheme: const DividerThemeData(
        color: AppColors.darkLine,
        thickness: 1,
        space: 1,
      ),

      // Chips: Panel tags with hairline border
      chipTheme: ChipThemeData(
        backgroundColor: AppColors.darkPaper,
        disabledColor: AppColors.darkCanvas,
        selectedColor: AppColors.darkMossSoft,
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
          side: const BorderSide(color: AppColors.darkLine, width: 1.0),
        ),
        labelStyle: _font(
          fontSize: 12,
          fontWeight: FontWeight.w600,
          color: AppColors.darkInk,
        ),
        elevation: 0,
        pressElevation: 0,
      ),

      // Dialog: Dark panel modal with hairline border
      dialogTheme: DialogThemeData(
        backgroundColor: AppColors.darkPaper,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.all(Radius.circular(16)),
          side: BorderSide(color: AppColors.darkLine, width: 1.0),
        ),
        titleTextStyle: _font(
          fontSize: 18,
          fontWeight: FontWeight.w700,
          color: AppColors.darkInk,
          letterSpacing: -0.3,
        ),
        contentTextStyle: _font(
          fontSize: 14,
          fontWeight: FontWeight.w400,
          color: AppColors.darkSecondary,
          height: 1.45,
        ),
      ),

      // Tooltip: Elevated dark badge
      tooltipTheme: TooltipThemeData(
        decoration: BoxDecoration(
          color: AppColors.darkPaperElevated,
          borderRadius: BorderRadius.circular(6),
          border: Border.all(color: AppColors.darkLine, width: 1),
        ),
        textStyle: _font(
          fontSize: 12,
          fontWeight: FontWeight.w500,
          color: AppColors.darkInk,
        ),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      ),

      // SnackBar: Floating panel notice
      snackBarTheme: SnackBarThemeData(
        backgroundColor: AppColors.darkPaperElevated,
        contentTextStyle: _font(
          fontSize: 14,
          fontWeight: FontWeight.w500,
          color: AppColors.darkInk,
        ),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10),
          side: const BorderSide(color: AppColors.darkLine, width: 1),
        ),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }
}
