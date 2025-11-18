import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;

import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:ffmpeg_kit_flutter_new/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter_new/return_code.dart';

class MLService {
  // --- Model Constants ---
  static const String _modelPath = 'assets/best_model_quantized.tflite';
  static const int _imgSize = 224;
  static const int _nSegments = 3;
  static const int _targetFPS = 15;
  static const int _framesPerSegment = 25;
  static const int _numClasses = 10;

  tfl.Interpreter? _interpreter;
  bool _isModelLoaded = false;

  // Quantization Parameters (Loaded from model)
  double _inputScale = 0.0;
  int _inputZeroPoint = 0;
  double _outputScale = 0.0;
  int _outputZeroPoint = 0;

  bool get isModelLoaded => _isModelLoaded;

  Future<void> loadModel() async {
    try {
      print('🚀 Loading TFLite Quantized Model: $_modelPath');

      final options = tfl.InterpreterOptions();
      options.threads = 4;

      _interpreter = await tfl.Interpreter.fromAsset(_modelPath, options: options);

      // --- GET QUANTIZATION PARAMS ---
      // We must manually quantize inputs and dequantize outputs
      final inputTensor = _interpreter!.getInputTensor(0);
      final outputTensor = _interpreter!.getOutputTensor(0);

      _inputScale = inputTensor.params.scale;
      _inputZeroPoint = inputTensor.params.zeroPoint;

      _outputScale = outputTensor.params.scale;
      _outputZeroPoint = outputTensor.params.zeroPoint;

      print('✅ Model loaded successfully');
      print('     Input Shape: ${inputTensor.shape}');
      print('     Input Quantization: Scale=$_inputScale, ZP=$_inputZeroPoint');
      print('     Output Quantization: Scale=$_outputScale, ZP=$_outputZeroPoint');

      _isModelLoaded = true;
    } catch (e) {
      print('❌ Error loading model: $e');
      throw Exception('Failed to load model. Error: $e');
    }
  }

  Future<Map<String, dynamic>?> analyzeVideo(
      String videoPath, {
        bool isLeftHanded = false,
      }) async {
    if (!_isModelLoaded || _interpreter == null) {
      print('❌ Model not loaded');
      return null;
    }

    try {
      print('🧠 Starting analysis for: $videoPath');

      // 1. Extract Frames
      final frames = await _extractFramesFromVideo(videoPath, isLeftHanded: isLeftHanded);
      if (frames.length < 2) return null;

      // 2. Preprocess & QUANTIZE
      // Returns Int8List shaped [1, 3, 224, 224, 3]
      final modelInput = _preprocessAndQuantize(frames);

      // 3. Inference
      // Output buffer for INT8 model must be Int8List (not Float32)
      // Shape: [1, 10]
      var outputBuffer = List.filled(1 * _numClasses, 0).reshape([1, _numClasses]);

      _interpreter!.run(modelInput, outputBuffer);

      // 4. De-Quantize Results
      // Convert Int8 outputs back to Probabilities (0.0 - 1.0)
      // Formula: real_val = (quant_val - zero_point) * scale
      final rawOutput = outputBuffer[0] as List<dynamic>;
      final probabilities = <double>[];

      for (var val in rawOutput) {
        // val is int (likely -128 to 127)
        double prob = (val - _outputZeroPoint) * _outputScale;
        probabilities.add(prob);
      }

      print('     📈 Probabilities: ${probabilities.map((p) => p.toStringAsFixed(3)).toList()}');

      // 5. Find Max
      double maxConfidence = 0.0;
      int predictedClass = -1;

      for (int i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxConfidence) {
          maxConfidence = probabilities[i];
          predictedClass = i;
        }
      }

      if (predictedClass == -1) return null;

      return {
        'classId': predictedClass,
        'confidence': maxConfidence,
        'allProbabilities': probabilities,
      };
    } catch (e) {
      print('❌ Error during video analysis: $e');
      rethrow;
    }
  }

  Future<List<img.Image>> _extractFramesFromVideo(String videoPath, {required bool isLeftHanded}) async {
    final tempDir = await getTemporaryDirectory();
    final oldFiles = tempDir.listSync().where((e) => e.path.contains("frame_"));
    for (var f in oldFiles) f.deleteSync();

    final outputPath = '${tempDir.path}/frame_%04d.png';
    String filters = "fps=$_targetFPS,scale=$_imgSize:$_imgSize,format=gray";
    if (isLeftHanded) filters += ",hflip";

    final command = '-i "$videoPath" -vf "$filters" -vsync vfr "$outputPath"';
    final session = await FFmpegKit.execute(command);

    if (!ReturnCode.isSuccess(await session.getReturnCode())) {
      throw Exception('FFmpeg failed');
    }

    final files = tempDir.listSync()
        .where((e) => e.path.contains("frame_") && e.path.endsWith(".png"))
        .map((e) => File(e.path))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));

    List<img.Image> loadedFrames = [];
    for (var f in files) {
      final bytes = await f.readAsBytes();
      final image = img.decodeImage(bytes);
      if (image != null) loadedFrames.add(image);
      await f.delete();
    }
    return loadedFrames;
  }

  /// 1. Calculates SAD features (Float).
  /// 2. Quantizes them to Int8 immediately.
  Object _preprocessAndQuantize(List<img.Image> allFrames) {
    final numFrames = allFrames.length;
    final segmentLength = numFrames ~/ 2;
    final overlap = segmentLength ~/ 2;

    // Prepare final tensor buffer: [1, 3, 224, 224, 3]
    // MUST BE INT8 (Bytes)
    final totalElements = 1 * _nSegments * _imgSize * _imgSize * 3;
    final fullInputBuffer = Int8List(totalElements);
    int bufferOffset = 0;

    // Fallback for tiny videos
    if (segmentLength < 2) {
      // Fill with Zero Point (equivalent to 0.0 float)
      fullInputBuffer.fillRange(0, totalElements, _inputZeroPoint);
      return fullInputBuffer.reshape([1, _nSegments, _imgSize, _imgSize, 3]);
    }

    List<List<img.Image>> segments = [];
    segments.add(allFrames.sublist(0, segmentLength));
    int start2 = overlap;
    int end2 = math.min(segmentLength + overlap, numFrames);
    segments.add(allFrames.sublist(start2, end2));
    int start3 = numFrames - segmentLength;
    if (start3 < 0) start3 = 0;
    segments.add(allFrames.sublist(start3, numFrames));

    // Fill missing segments
    while (segments.length < _nSegments) segments.add(segments.last);
    segments = segments.sublist(0, _nSegments);

    for (var segmentFrames in segments) {
      List<img.Image> subsampled = [];

      if (segmentFrames.length < 2) {
        // Empty segment -> Fill with Zero Point
        int segSize = _imgSize * _imgSize * 3;
        for (int k = 0; k < segSize; k++) {
          fullInputBuffer[bufferOffset++] = _inputZeroPoint;
        }
        continue;
      }

      // Subsample
      int count = math.min(_framesPerSegment, segmentFrames.length);
      if (count < 2) count = segmentFrames.length;
      for (int i = 0; i < count; i++) {
        double val = (segmentFrames.length - 1) * i / (count - 1);
        subsampled.add(segmentFrames[val.round()]);
      }

      // Calculate Diffs & Threshold (Standard SAD logic)
      List<Float32List> diffs = [];
      List<double> allDiffValues = [];

      for (int i = 0; i < subsampled.length - 1; i++) {
        final frame1 = subsampled[i];
        final frame2 = subsampled[i+1];
        final diffImg = Float32List(_imgSize * _imgSize);
        int pIdx = 0;

        for (int y = 0; y < _imgSize; y++) {
          for (int x = 0; x < _imgSize; x++) {
            final p1 = frame1.getPixel(x, y).r;
            final p2 = frame2.getPixel(x, y).r;
            final d = (p1 - p2).abs().toDouble();
            diffImg[pIdx++] = d;
            allDiffValues.add(d);
          }
        }
        diffs.add(diffImg);
      }

      double threshold = 0.0;
      if (allDiffValues.isNotEmpty) {
        allDiffValues.sort();
        int threshIdx = (allDiffValues.length * 0.98).floor();
        if (threshIdx >= allDiffValues.length) threshIdx = allDiffValues.length - 1;
        threshold = allDiffValues[threshIdx];
      }

      // Create SAD Image (Float)
      final sadImage = Float32List(_imgSize * _imgSize);
      double maxVal = 0.0;

      for (var diff in diffs) {
        for (int i = 0; i < diff.length; i++) {
          double val = diff[i];
          if (val < threshold) val = 0.0;
          sadImage[i] += val;
        }
      }

      for (var val in sadImage) if (val > maxVal) maxVal = val;

      // --- QUANTIZATION STEP ---
      // Normalize (0-1) -> Quantize (Int8)
      for (int i = 0; i < sadImage.length; i++) {
        double normVal = 0.0;
        if (maxVal > 0) normVal = sadImage[i] / maxVal;

        // Quantize: q = (real / scale) + zero_point
        int quantizedVal = (normVal / _inputScale + _inputZeroPoint).round();

        // Clamp to Int8 range (-128 to 127)
        quantizedVal = quantizedVal.clamp(-128, 127);

        // Add 3 times (RGB)
        fullInputBuffer[bufferOffset++] = quantizedVal;
        fullInputBuffer[bufferOffset++] = quantizedVal;
        fullInputBuffer[bufferOffset++] = quantizedVal;
      }
    }

    return fullInputBuffer.reshape([1, _nSegments, _imgSize, _imgSize, 3]);
  }

  void resetSequence() {}

  void dispose() {
    _interpreter?.close();
  }
}