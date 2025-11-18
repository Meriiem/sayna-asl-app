import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class AppSettingsProvider extends ChangeNotifier {
  static const String _themeKey = 'theme_mode';
  static const String _localeKey = 'locale';
  static const String _confidenceThresholdKey = 'confidence_threshold';
  static const String _soundEnabledKey = 'sound_enabled';
  static const String _vibrationEnabledKey = 'vibration_enabled';
  static const String _showArabicLabelsKey = 'show_arabic_labels';
  static const String _isLeftHandedKey = 'is_left_handed'; // <-- ADDED

  SharedPreferences? _prefs;

  // Settings variables
  ThemeMode _themeMode = ThemeMode.system;
  Locale _locale = const Locale('en', 'US');
  double _confidenceThreshold = 0.7;
  bool _soundEnabled = true;
  bool _vibrationEnabled = true;
  bool _showArabicLabels = true;
  bool _isLeftHanded = false; // <-- ADDED

  // Getters
  ThemeMode get themeMode => _themeMode;
  Locale get locale => _locale;
  double get confidenceThreshold => _confidenceThreshold;
  bool get soundEnabled => _soundEnabled;
  bool get vibrationEnabled => _vibrationEnabled;
  bool get showArabicLabels => _showArabicLabels;
  bool get isArabicLocale => _locale.languageCode == 'ar';
  bool get isLeftHanded => _isLeftHanded; // <-- ADDED

  // Initialize settings
  Future<void> initialize() async {
    _prefs = await SharedPreferences.getInstance();
    await _loadSettings();
  }

  // Load settings from shared preferences
  Future<void> _loadSettings() async {
    if (_prefs == null) return;

    // Load theme mode
    final themeIndex = _prefs!.getInt(_themeKey) ?? ThemeMode.system.index;
    _themeMode = ThemeMode.values[themeIndex];

    // Load locale
    final localeString = _prefs!.getString(_localeKey) ?? 'en_US';
    final localeParts = localeString.split('_');
    _locale = Locale(localeParts[0], localeParts.length > 1 ? localeParts[1] : '');

    // Load other settings
    _confidenceThreshold = _prefs!.getDouble(_confidenceThresholdKey) ?? 0.7;
    _soundEnabled = _prefs!.getBool(_soundEnabledKey) ?? true;
    _vibrationEnabled = _prefs!.getBool(_vibrationEnabledKey) ?? true;
    _showArabicLabels = _prefs!.getBool(_showArabicLabelsKey) ?? true;
    _isLeftHanded = _prefs!.getBool(_isLeftHandedKey) ?? false; // <-- ADDED

    notifyListeners();
  }

  // Set theme mode
  Future<void> setThemeMode(ThemeMode mode) async {
    _themeMode = mode;
    await _prefs?.setInt(_themeKey, mode.index);
    notifyListeners();
  }

  // Set locale
  Future<void> setLocale(Locale locale) async {
    _locale = locale;
    await _prefs?.setString(_localeKey, '${locale.languageCode}_${locale.countryCode}');
    notifyListeners();
  }

  // Set confidence threshold
  Future<void> setConfidenceThreshold(double threshold) async {
    _confidenceThreshold = threshold.clamp(0.1, 1.0);
    await _prefs?.setDouble(_confidenceThresholdKey, _confidenceThreshold);
    notifyListeners();
  }

  // Toggle sound
  Future<void> toggleSound() async {
    _soundEnabled = !_soundEnabled;
    await _prefs?.setBool(_soundEnabledKey, _soundEnabled);
    notifyListeners();
  }

  // Toggle vibration
  Future<void> toggleVibration() async {
    _vibrationEnabled = !_vibrationEnabled;
    await _prefs?.setBool(_vibrationEnabledKey, _vibrationEnabled);
    notifyListeners();
  }

  // Toggle Arabic labels
  Future<void> toggleArabicLabels() async {
    _showArabicLabels = !_showArabicLabels;
    await _prefs?.setBool(_showArabicLabelsKey, _showArabicLabels);
    notifyListeners();
  }

  // <-- ADDED THIS ENTIRE FUNCTION -->
  // Toggle Left-Handed Mode
  Future<void> toggleLeftHandedMode(bool isLeftHanded) async {
    _isLeftHanded = isLeftHanded;
    await _prefs?.setBool(_isLeftHandedKey, _isLeftHanded);
    notifyListeners();
  }
  // <-- END OF ADDED FUNCTION -->

  // Reset to defaults
  Future<void> resetToDefaults() async {
    _themeMode = ThemeMode.system;
    _locale = const Locale('en', 'US');
    _confidenceThreshold = 0.7;
    _soundEnabled = true;
    _vibrationEnabled = true;
    _showArabicLabels = true;
    _isLeftHanded = false; // <-- ADDED

    await _prefs?.clear();
    await _saveAllSettings();

    notifyListeners();
  }

  // Save all settings
  Future<void> _saveAllSettings() async {
    if (_prefs == null) return;

    await _prefs!.setInt(_themeKey, _themeMode.index);
    await _prefs!.setString(_localeKey, '${_locale.languageCode}_${_locale.countryCode}');
    await _prefs!.setDouble(_confidenceThresholdKey, _confidenceThreshold);
    await _prefs!.setBool(_soundEnabledKey, _soundEnabled);
    await _prefs!.setBool(_vibrationEnabledKey, _vibrationEnabled);
    await _prefs!.setBool(_showArabicLabelsKey, _showArabicLabels);
    await _prefs!.setBool(_isLeftHandedKey, _isLeftHanded); // <-- ADDED
  }

  // Get settings summary
  Map<String, dynamic> getSettingsSummary() {
    return {
      'themeMode': _themeMode.toString(),
      'locale': '${_locale.languageCode}_${_locale.countryCode}',
      'confidenceThreshold': _confidenceThreshold,
      'soundEnabled': _soundEnabled,
      'vibrationEnabled': _vibrationEnabled,
      'showArabicLabels': _showArabicLabels,
      'isLeftHanded': _isLeftHanded, // <-- ADDED
    };
  }
}