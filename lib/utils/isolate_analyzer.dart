// lib/utils/isolate_analyzer.dart

import 'dart:isolate';
import 'package:flutter/foundation.dart';
import '../providers/sign_detection_provider.dart';

// This is the function that will run on the separate isolate
// It must be a top-level or static function.
Future<void> runAnalysisInIsolate(Map<String, dynamic> args) async {
  // Pull necessary arguments out of the map
  final SignDetectionProvider provider = args['provider'];
  final String videoPath = args['videoPath'];
  final double confidenceThreshold = args['confidenceThreshold'];
  final bool isLeftHanded = args['isLeftHanded'];

  // Run the actual analysis function
  await provider.analyzeRecordedVideo(
    videoPath,
    confidenceThreshold,
    isLeftHanded: isLeftHanded,
  );
}