import 'package:flutter/foundation.dart';
import '../models/sign_model.dart';
import '../services/ml_service.dart';

enum DetectionState {
  idle,
  detecting,
  detected,
  error,
}

class SignDetectionProvider extends ChangeNotifier {
  final MLService _mlService = MLService();

  // State variables
  DetectionState _state = DetectionState.idle;
  SignModel? _currentSign;
  double _confidence = 0.0;
  String? _errorMessage;
  bool _isModelLoaded = false;
  List<SignModel> _detectionHistory = [];

  // === GAMIFICATION PROPERTIES ===
  int _userPoints = 0;
  int _correctDetections = 0;
  int _totalAttempts = 0;
  Map<String, int> _signStreaks = {};
  List<Map<String, dynamic>> _recentPoints = [];

  // Getters
  DetectionState get state => _state;
  SignModel? get currentSign => _currentSign;
  double get confidence => _confidence;
  String? get errorMessage => _errorMessage;
  bool get isModelLoaded => _isModelLoaded;
  List<SignModel> get detectionHistory => List.unmodifiable(_detectionHistory);

  // Gamification Getters
  int get userPoints => _userPoints;
  int get correctDetections => _correctDetections;
  int get totalAttempts => _totalAttempts;
  double get accuracy => _totalAttempts > 0 ? _correctDetections / _totalAttempts : 0.0;
  List<Map<String, dynamic>> get recentPoints => List.unmodifiable(_recentPoints);

  // Initialize the ML model
  Future<void> initializeModel() async {
    try {
      _setState(DetectionState.idle);
      await _mlService.loadModel();
      _isModelLoaded = true;
      print('✅ ML Model initialized successfully');
      notifyListeners();
    } catch (e) {
      print('❌ Failed to initialize ML model: $e');
      _setError('Failed to load ML model: ${e.toString()}');
    }
  }

  // Analyze a saved video file
  Future<void> analyzeRecordedVideo(
      String videoPath,
      double confidenceThreshold, {
        bool isLeftHanded = false,
      }) async {
    if (!_isModelLoaded) {
      _setError('ML Model not initialized. Please restart the app.');
      return;
    }

    try {
      _setState(DetectionState.detecting);

      final result = await _mlService.analyzeVideo(
        videoPath,
        isLeftHanded: isLeftHanded,
      );

      if (result == null) {
        _setError('Could not analyze the video or video too short.');
        return;
      }

      final int classId = result['classId'] as int;
      final double confidence = result['confidence'] as double;

      final sign = SignModel.getById(classId);
      if (sign == null) {
        _setError('Unknown sign detected (Class ID: $classId)');
        return;
      }

      if (confidence < confidenceThreshold) {
        _setError(
          'Sign not clear (Confidence: ${(confidence * 100).toStringAsFixed(0)}%)',
        );
        return;
      }

      final detectedSign = sign.copyWith(
        confidence: confidence,
        detectedAt: DateTime.now(),
      );

      setDetectionResult(detectedSign, confidence);

    } catch (e) {
      _setError('Analysis failed: $e');
    }
  }

  // Public method to set the detection result
  void setDetectionResult(SignModel sign, double confidence) {
    _currentSign = sign;
    _confidence = confidence;
    _setState(DetectionState.detected);

    _addToHistory(sign);
    _awardPoints(sign, confidence);

    notifyListeners();
  }

  // === GAMIFICATION METHODS ===
  void _awardPoints(SignModel sign, double confidence) {
    _totalAttempts++;
    int pointsEarned = 10;
    String bonusType = "Base";

    if (confidence >= 0.9) {
      pointsEarned += 8;
      bonusType = "High Accuracy";
    } else if (confidence >= 0.8) {
      pointsEarned += 5;
      bonusType = "Good Accuracy";
    }

    final signKey = sign.gestureId;
    _signStreaks[signKey] = (_signStreaks[signKey] ?? 0) + 1;
    final currentStreak = _signStreaks[signKey]!;

    if (currentStreak >= 3) {
      final streakBonus = currentStreak * 3;
      pointsEarned += streakBonus;
      bonusType = "Streak x$currentStreak";
    }

    _userPoints += pointsEarned;
    _correctDetections++;

    _recentPoints.insert(0, {
      'points': pointsEarned,
      'sign': sign.englishLabel,
      'confidence': confidence,
      'bonusType': bonusType,
      'timestamp': DateTime.now(),
    });

    if (_recentPoints.length > 10) {
      _recentPoints = _recentPoints.take(10).toList();
    }
    notifyListeners();
  }

  Map<String, dynamic> getUserProgress() {
    final level = (_userPoints / 100).floor() + 1;
    final nextLevelPoints = level * 100;
    final progress = (_userPoints % 100) / 100.0;

    return {
      'points': _userPoints,
      'correctDetections': _correctDetections,
      'totalAttempts': _totalAttempts,
      'accuracy': accuracy,
      'level': level,
      'nextLevelPoints': nextLevelPoints,
      'progress': progress,
      'pointsToNextLevel': nextLevelPoints - _userPoints,
    };
  }

  String getUserRank() {
    final points = _userPoints;
    if (points >= 1000) return "ASL Master";
    if (points >= 500) return "Expert";
    if (points >= 300) return "Advanced";
    if (points >= 150) return "Intermediate";
    if (points >= 50) return "Beginner";
    return "Newbie";
  }

  void resetGameData() {
    _userPoints = 0;
    _correctDetections = 0;
    _totalAttempts = 0;
    _signStreaks.clear();
    _recentPoints.clear();
    notifyListeners();
  }

  void _addToHistory(SignModel sign) {
    _detectionHistory.insert(0, sign);
    if (_detectionHistory.length > 50) {
      _detectionHistory = _detectionHistory.take(50).toList();
    }
    notifyListeners();
  }

  void clearHistory() {
    _detectionHistory.clear();
    notifyListeners();
  }

  void resetDetection() {
    _currentSign = null;
    _confidence = 0.0;
    _mlService.resetSequence();
    _setState(DetectionState.idle);
  }

  Map<String, dynamic> getStatistics() {
    final signCounts = <int, int>{};
    for (final detection in _detectionHistory) {
      signCounts[detection.id] = (signCounts[detection.id] ?? 0) + 1;
    }

    final mostDetectedEntry = signCounts.entries.isEmpty
        ? null
        : signCounts.entries.reduce((a, b) => a.value > b.value ? a : b);

    final mostDetectedSign = mostDetectedEntry != null
        ? SignModel.getById(mostDetectedEntry.key)
        : null;

    double avgConf = 0.0;
    if (_detectionHistory.isNotEmpty) {
      avgConf = _detectionHistory.map((s) => s.confidence).reduce((a, b) => a + b) / _detectionHistory.length;
    }

    return {
      'totalDetections': _detectionHistory.length,
      'uniqueSigns': signCounts.keys.length,
      'mostDetectedSign': mostDetectedSign,
      'averageConfidence': avgConf,
    };
  }

  void _setState(DetectionState newState) {
    _state = newState;
    if (newState != DetectionState.error) {
      _errorMessage = null;
    }
    notifyListeners();
  }

  void _setError(String error) {
    _state = DetectionState.error;
    _errorMessage = error;
    _currentSign = null;
    _confidence = 0.0;
    notifyListeners();
  }

  @override
  void dispose() {
    _mlService.dispose();
    super.dispose();
  }
}