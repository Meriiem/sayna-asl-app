// import 'package:flutter/material.dart';
// import 'package:flutter/services.dart';
// import 'package:provider/provider.dart';
// import 'package:google_fonts/google_fonts.dart';
// import 'package:flutter_localizations/flutter_localizations.dart';
//
// import 'screens/splash_screen.dart';
// import 'screens/home_screen.dart';
// import 'screens/camera_screen.dart';
// import 'screens/history_screen.dart';
// import 'screens/settings_screen.dart';
// import 'providers/sign_detection_provider.dart';
// import 'providers/app_settings_provider.dart';
// import 'utils/app_theme.dart';
//
// void main() async {
//   WidgetsFlutterBinding.ensureInitialized();
//
//   // Set preferred orientations
//   await SystemChrome.setPreferredOrientations([
//     DeviceOrientation.portraitUp,
//     DeviceOrientation.portraitDown,
//   ]);
//
//   runApp(const ArabicSignLanguageApp());
// }
//
// class ArabicSignLanguageApp extends StatelessWidget {
//   const ArabicSignLanguageApp({super.key});
//
//   @override
//   Widget build(BuildContext context) {
//     return MultiProvider(
//       providers: [
//         ChangeNotifierProvider(create: (_) => SignDetectionProvider()),
//         ChangeNotifierProvider(create: (_) => AppSettingsProvider()),
//       ],
//       child: Consumer<AppSettingsProvider>(
//         builder: (context, settingsProvider, child) {
//           return MaterialApp(
//             title: 'Arabic Sign Language',
//             debugShowCheckedModeBanner: false,
//
//             // Theme configuration
//             theme: AppTheme.lightTheme,
//             darkTheme: AppTheme.darkTheme,
//             themeMode: settingsProvider.themeMode,
//
//             // Localization
//             locale: settingsProvider.locale,
//             supportedLocales: const [
//               Locale('en', 'US'),
//               Locale('ar', 'AE'),
//             ],
//             localizationsDelegates: const [
//               GlobalMaterialLocalizations.delegate,
//               GlobalWidgetsLocalizations.delegate,
//               GlobalCupertinoLocalizations.delegate,
//             ],
//
//             // Routes
//             initialRoute: '/',
//             routes: {
//               '/': (context) => const SplashScreen(),
//               '/home': (context) => const HomeScreen(),
//               '/camera': (context) => const CameraScreen(),
//               '/history': (context) => const HistoryScreen(),
//               '/settings': (context) => const SettingsScreen(),
//             },
//           );
//         },
//       ),
//     );
//   }
// }

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'firebase_options.dart';

import 'screens/splash_screen.dart';
import 'screens/home_screen.dart';
import 'screens/camera_screen.dart';
import 'screens/history_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/login_screen.dart';
import 'providers/sign_detection_provider.dart';
import 'providers/app_settings_provider.dart';
import 'utils/app_theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Lock orientation (your original setting)
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  // Init Firebase (one-time)
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  // Refresh local auth cache; if the user was deleted remotely, sign out.
  try {
    await FirebaseAuth.instance.currentUser?.reload();
  } catch (_) {
    await FirebaseAuth.instance.signOut();
  }

  runApp(const ArabicSignLanguageApp());
}

class ArabicSignLanguageApp extends StatelessWidget {
  const ArabicSignLanguageApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => SignDetectionProvider()),
        ChangeNotifierProvider(create: (_) => AppSettingsProvider()),
      ],
      child: Consumer<AppSettingsProvider>(
        builder: (context, settingsProvider, _) {
          return MaterialApp(
            title: 'SAYNA',
            debugShowCheckedModeBanner: false,

            // Theme
            theme: AppTheme.lightTheme,
            darkTheme: AppTheme.darkTheme,
            themeMode: settingsProvider.themeMode,

            // Localization
            locale: settingsProvider.locale,
            supportedLocales: const [
              Locale('en', 'US'),
              Locale('ar', 'AE'),
            ],
            localizationsDelegates: const [
              GlobalMaterialLocalizations.delegate,
              GlobalWidgetsLocalizations.delegate,
              GlobalCupertinoLocalizations.delegate,
            ],

            // Show Splash first, then decide where to go.
            home: const _SplashRouter(),

            // Keep your named routes
            routes: {
              '/home': (context) => const HomeScreen(),
              '/camera': (context) => const CameraScreen(),
              '/history': (context) => const HistoryScreen(),
              '/settings': (context) => const SettingsScreen(),
              '/login': (context) => const LoginScreen(),
            },
          );
        },
      ),
    );
  }
}

/// Shows SplashScreen for a minimum duration, then routes to Home/Login
class _SplashRouter extends StatefulWidget {
  const _SplashRouter({super.key});

  @override
  State<_SplashRouter> createState() => _SplashRouterState();
}

class _SplashRouterState extends State<_SplashRouter> {
  @override
  void initState() {
    super.initState();
    _boot();
  }

  Future<void> _boot() async {
    // Wait for both: a short splash delay AND a fresh user check
    await Future.wait([
      Future.delayed(const Duration(seconds: 2)), // minimum splash time
      _ensureFreshSession(),
    ]);

    final user = FirebaseAuth.instance.currentUser;
    if (!mounted) return;

    // Go to Home if signed in, otherwise Login
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(
        builder: (_) => user == null ? const LoginScreen() : const HomeScreen(),
      ),
    );
  }

  // If the user was deleted remotely, this signs them out locally
  Future<void> _ensureFreshSession() async {
    try {
      await FirebaseAuth.instance.currentUser?.reload();
    } catch (_) {
      await FirebaseAuth.instance.signOut();
    }
  }

  @override
  Widget build(BuildContext context) {
    // While booting, show your Splash screen
    return const SplashScreen();
  }
}
