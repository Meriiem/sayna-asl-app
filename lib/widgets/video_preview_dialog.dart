import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:video_player/video_player.dart';
import '../providers/app_settings_provider.dart';
import '../providers/sign_detection_provider.dart';

class VideoPreviewDialog extends StatefulWidget {
  final File videoFile;
  final bool isArabic;

  const VideoPreviewDialog({
    super.key,
    required this.videoFile,
    required this.isArabic,
  });

  @override
  State<VideoPreviewDialog> createState() => _VideoPreviewDialogState();
}

class _VideoPreviewDialogState extends State<VideoPreviewDialog> {
  late VideoPlayerController _controller;
  bool _isInitializing = true;
  bool _isDetecting = false;
  String? _resultMessage;
  bool _detectionSuccess = false;

  @override
  void initState() {
    super.initState();
    _controller = VideoPlayerController.file(widget.videoFile)
      ..initialize().then((_) {
        if (mounted) setState(() => _isInitializing = false);
      });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _runDetection() async {
    setState(() {
      _isDetecting = true;
      _resultMessage = null;
    });

    try {
      final settings = context.read<AppSettingsProvider>();
      final detectionProvider = context.read<SignDetectionProvider>();

      // -------------------------------------------
      // FIX: Pass the user setting for left-handed
      // -------------------------------------------
      final isLeftHanded = settings.isLeftHanded;

      await detectionProvider.analyzeRecordedVideo(
        widget.videoFile.path,
        settings.confidenceThreshold,
        isLeftHanded: isLeftHanded,
      );

      // Check detection result state
      if (detectionProvider.state == DetectionState.error) {
        throw Exception(detectionProvider.errorMessage);
      }

      final currentSign = detectionProvider.currentSign;

      if (currentSign != null) {
        setState(() {
          _isDetecting = false;
          _detectionSuccess = true;
          _resultMessage = widget.isArabic
              ? 'تم الكشف عن الإشارة: ${currentSign.arabicLabel} (${(detectionProvider.confidence*100).toInt()}%)'
              : 'Detected sign: ${currentSign.englishLabel} (${(detectionProvider.confidence*100).toInt()}%)';
        });
      } else {
        setState(() {
          _isDetecting = false;
          _detectionSuccess = false;
          _resultMessage = widget.isArabic ? "لم يتم اكتشاف إشارة واضحة" : "No clear sign detected";
        });
      }

    } catch (e) {
      setState(() {
        _isDetecting = false;
        _detectionSuccess = false;
        _resultMessage = widget.isArabic
            ? 'حدث خطأ أثناء الكشف: $e'
            : 'Error during detection: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: Text(widget.isArabic ? 'معاينة الفيديو' : 'Video Preview'),
      content: SizedBox(
        width: MediaQuery.of(context).size.width * 0.8,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (_isInitializing)
              const Padding(
                padding: EdgeInsets.all(16),
                child: CircularProgressIndicator(),
              )
            else ...[
              AspectRatio(
                aspectRatio: _controller.value.aspectRatio,
                child: VideoPlayer(_controller),
              ),
              const SizedBox(height: 8),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  IconButton(
                    icon: Icon(
                      _controller.value.isPlaying
                          ? Icons.pause
                          : Icons.play_arrow,
                    ),
                    onPressed: () {
                      setState(() {
                        _controller.value.isPlaying
                            ? _controller.pause()
                            : _controller.play();
                      });
                    },
                  ),
                ],
              ),
            ],
            const SizedBox(height: 12),
            if (_isDetecting)
              Column(
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 10),
                  Text(widget.isArabic ? "جاري التحليل..." : "Analyzing...")
                ],
              )
            else if (_resultMessage != null) ...[
              Text(
                _resultMessage!,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: _detectionSuccess ? Colors.green : Colors.red,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ]
          ],
        ),
      ),
      actions: _buildActions(context),
    );
  }

  List<Widget> _buildActions(BuildContext context) {
    final closeText = widget.isArabic ? 'إغلاق' : 'Close';
    final detectText = widget.isArabic ? 'كشف' : 'Detect';
    final historyText = widget.isArabic ? 'عرض السجل' : 'View History';

    if (_resultMessage != null) {
      return [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: Text(closeText),
        ),
        if (_detectionSuccess)
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              Navigator.pushNamed(context, '/history');
            },
            child: Text(historyText),
          ),
      ];
    }

    return [
      TextButton(
        onPressed: _isDetecting ? null : () => Navigator.of(context).pop(),
        child: Text(closeText),
      ),
      ElevatedButton.icon(
        icon: Icon(Icons.analytics),
        onPressed: _isDetecting ? null : _runDetection,
        label: Text(detectText),
      ),
    ];
  }
}