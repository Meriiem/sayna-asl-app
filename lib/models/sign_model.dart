class SignModel {
  final int id;
  final String gestureId;
  final String englishLabel;
  final String arabicLabel;
  final String description;
  final String iconPath;
  final double confidence;
  final DateTime? detectedAt;
  final String? videoId;

  const SignModel({
    required this.id,
    required this.gestureId,
    required this.englishLabel,
    required this.arabicLabel,
    required this.description,
    required this.iconPath,
    this.confidence = 0.0,
    this.detectedAt,
    this.videoId,
  });

  // Static list of all supported signs
  static const List<SignModel> allSigns = [
    SignModel(
      id: 0,
      gestureId: 'G01',
      englishLabel: 'Hi',
      arabicLabel: 'اهلا',
      description: 'Greeting gesture',
      iconPath: 'assets/icons/hi.svg',
      videoId: '84w07z97RTk',
    ),
    SignModel(
      id: 1,
      gestureId: 'G02',
      englishLabel: 'Please',
      arabicLabel: 'من فضلك',
      description: 'Polite request gesture',
      iconPath: 'assets/icons/please.svg',
      videoId: 'cbVukiJaCf0',
    ),
    SignModel(
      id: 2,
      gestureId: 'G03',
      englishLabel: 'What?',
      arabicLabel: 'ماذا؟',
      description: 'Question gesture',
      iconPath: 'assets/icons/what.svg',
      videoId: 'B0y1ko9tprI',
    ),
    SignModel(
      id: 3,
      gestureId: 'G04',
      englishLabel: 'Arabic',
      arabicLabel: 'العربية',
      description: 'Language reference gesture',
      iconPath: 'assets/icons/arabic.svg',
      videoId: 'FwhUbqGAkac',
    ),
    SignModel(
      id: 4,
      gestureId: 'G05',
      englishLabel: 'University',
      arabicLabel: 'جامعة',
      description: 'Educational institution gesture',
      iconPath: 'assets/icons/university.svg',
      videoId: 'W4jvoirG9oM',
    ),
    SignModel(
      id: 5,
      gestureId: 'G06',
      englishLabel: 'You',
      arabicLabel: 'انت',
      description: 'Personal pronoun gesture',
      iconPath: 'assets/icons/you.svg',
      videoId: 'JTMavg80PTs',
    ),
    SignModel(
      id: 6,
      gestureId: 'G07',
      englishLabel: 'Eat',
      arabicLabel: 'كل',
      description: 'Eating action gesture',
      iconPath: 'assets/icons/eat.svg',
      videoId: 'dpwEHIMqX0A',
    ),
    SignModel(
      id: 7,
      gestureId: 'G08',
      englishLabel: 'Sleep',
      arabicLabel: 'نام',
      description: 'Sleeping action gesture',
      iconPath: 'assets/icons/sleep.svg',
      videoId: 'TcIVa-1Sdek',
    ),
    SignModel(
      id: 8,
      gestureId: 'G09',
      englishLabel: 'Go',
      arabicLabel: 'اذهب',
      description: 'Movement action gesture',
      iconPath: 'assets/icons/go.svg',
      videoId: '9JfP-HzDKYs',
    ),
    SignModel(
      id: 9,
      gestureId: 'G10',
      englishLabel: 'UAE',
      arabicLabel: 'الامارات العربية المتحدة',
      description: 'Country reference gesture',
      iconPath: 'assets/icons/uae.svg',
      videoId: 'MiMSUxWPYbA',
    ),
  ];

  // Get sign by ID
  static SignModel? getById(int id) {
    try {
      return allSigns.firstWhere((sign) => sign.id == id);
    } catch (e) {
      return null;
    }
  }

  // Get sign by gesture ID
  static SignModel? getByGestureId(String gestureId) {
    try {
      return allSigns.firstWhere((sign) => sign.gestureId == gestureId);
    } catch (e) {
      return null;
    }
  }

  // Create a copy with updated confidence and timestamp
  SignModel copyWith({
    int? id,
    String? gestureId,
    String? englishLabel,
    String? arabicLabel,
    String? description,
    String? iconPath,
    double? confidence,
    DateTime? detectedAt,
  }) {
    return SignModel(
      id: id ?? this.id,
      gestureId: gestureId ?? this.gestureId,
      englishLabel: englishLabel ?? this.englishLabel,
      arabicLabel: arabicLabel ?? this.arabicLabel,
      description: description ?? this.description,
      iconPath: iconPath ?? this.iconPath,
      confidence: confidence ?? this.confidence,
      detectedAt: detectedAt ?? this.detectedAt,
    );
  }

  // Convert to JSON
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'gestureId': gestureId,
      'englishLabel': englishLabel,
      'arabicLabel': arabicLabel,
      'description': description,
      'iconPath': iconPath,
      'confidence': confidence,
      'detectedAt': detectedAt?.toIso8601String(),
    };
  }

  // Create from JSON
  factory SignModel.fromJson(Map<String, dynamic> json) {
    return SignModel(
      id: json['id'],
      gestureId: json['gestureId'],
      englishLabel: json['englishLabel'],
      arabicLabel: json['arabicLabel'],
      description: json['description'],
      iconPath: json['iconPath'],
      confidence: json['confidence'] ?? 0.0,
      detectedAt: json['detectedAt'] != null
          ? DateTime.parse(json['detectedAt'])
          : null,
    );
  }

  @override
  String toString() {
    return 'SignModel(id: $id, gestureId: $gestureId, englishLabel: $englishLabel, arabicLabel: $arabicLabel, confidence: $confidence)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is SignModel && other.id == id;
  }

  @override
  int get hashCode => id.hashCode;
}