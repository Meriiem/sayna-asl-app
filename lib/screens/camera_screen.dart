import 'dart:io';
import 'package:flutter/services.dart'; // <-- ADD THIS IMPORT
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:video_player/video_player.dart';
import 'package:path_provider/path_provider.dart';

import '../providers/sign_detection_provider.dart';

import '../providers/app_settings_provider.dart';

import '../models/sign_model.dart';

import 'dart:async';



class CameraScreen extends StatefulWidget {

  const CameraScreen({super.key});



  @override

  State<CameraScreen> createState() => _CameraScreenState();

}



class _CameraScreenState extends State<CameraScreen> with WidgetsBindingObserver {

  CameraController? _cameraController;

  VideoPlayerController? _videoController;

  bool _isRecording = false;

  bool _isPlaying = false;

  bool _isCameraInitialized = false;

  String? _recordedVideoPath;



  Timer? _recordingTimer;

  int _secondsElapsed = 0;

  static const int _minRecordTime = 2; // minimum 2 seconds

  static const int _maxRecordTime = 3; // <-- ADD THIS LINE



  @override

  void initState() {

    super.initState();

// --- ADD THIS BLOCK to force landscape ---

    SystemChrome.setPreferredOrientations([

      DeviceOrientation.landscapeLeft,

      DeviceOrientation.landscapeRight,

    ]);

// --- END BLOCK ---



    WidgetsBinding.instance.addObserver(this);

    _initializeCamera();

  }



  @override

  void dispose() {

// --- ADD THIS BLOCK to reset orientation ---

    SystemChrome.setPreferredOrientations([

      DeviceOrientation.portraitUp,

      DeviceOrientation.portraitDown,

    ]);

// --- END BLOCK ---



    WidgetsBinding.instance.removeObserver(this);

    _cameraController?.dispose();

    _videoController?.dispose();

    _recordingTimer?.cancel();

    super.dispose();

  }



  Future<void> _initializeCamera() async {

    try {

      final cameras = await availableCameras();



// --- 1. Use Back Camera ---

      final backCamera = cameras.firstWhere(

            (camera) => camera.lensDirection == CameraLensDirection.back, // <-- CHANGED

        orElse: () => cameras.first,

      );



      _cameraController = CameraController(

        backCamera, // <-- CHANGED

        ResolutionPreset.medium,

        enableAudio: false,

      );



      await _cameraController!.initialize();



// --- 2. Lock Sensor Orientation to Landscape ---

// This ensures the camera's output is horizontal

      await _cameraController!.lockCaptureOrientation(DeviceOrientation.landscapeLeft);



      setState(() {

        _isCameraInitialized = true;

      });

    } catch (e) {

      print('Error initializing camera: $e');

    }

  }



  Future<void> _startRecording() async {

    if (!_isCameraInitialized || _isRecording) return;



    try {

// --- NEW PERMISSION CHECK ---

      print("Checking microphone permission...");

      var micStatus = await Permission.microphone.status;

      if (!micStatus.isGranted) {

        micStatus = await Permission.microphone.request();

      }

      if (!micStatus.isGranted) {

        print("Microphone permission was denied.");

        ScaffoldMessenger.of(context).showSnackBar(

            const SnackBar(content: Text('Microphone permission is required to record video.'))

        );

        return;

      }

// --- (End of permission logic) ---



      print("Permissions OK. Starting recording...");

      final directory = await getTemporaryDirectory();

      final videoPath = '${directory.path}/${DateTime.now().millisecondsSinceEpoch}.mp4';



      await _cameraController!.startVideoRecording();



      setState(() {

        _isRecording = true;

        _secondsElapsed = 0;

      });



      _recordingTimer?.cancel();

      _recordingTimer = Timer.periodic(const Duration(seconds: 1), (timer) {

        if (!mounted) {

          timer.cancel();

          return;

        }



        setState(() {

          _secondsElapsed++;

        });



// --- ADD THIS CHECK ---

// Automatically stop recording when max time is reached

        if (_secondsElapsed >= _maxRecordTime) {

          print("Max record time (3s) reached. Stopping.");

          timer.cancel();

          _stopRecording(); // This will trigger the analysis

        }

// --- END CHECK ---

      });

    } catch (e) {

      print('Error starting recording: $e');

      ScaffoldMessenger.of(context).showSnackBar(

          SnackBar(content: Text('Error starting recording: ${e.toString()}'))

      );

    }

  }



  Future<void> _stopRecording() async {

    if (!_isRecording) return;



    _recordingTimer?.cancel();



    if (_secondsElapsed < _minRecordTime) {

      print("Recording is too short.");

      ScaffoldMessenger.of(context).showSnackBar(

          SnackBar(content: Text('Please record for at least $_minRecordTime seconds.'))

      );



// Abort recording

      await _cameraController!.stopVideoRecording(); // Stop but don't use the file

      setState(() {

        _isRecording = false;

        _secondsElapsed = 0;

      });

      return; // Stop execution

    }



    try {

      final file = await _cameraController!.stopVideoRecording();



      setState(() {

        _isRecording = false;

        _recordedVideoPath = file.path;

        _secondsElapsed = 0; // Reset timer

      });



// Initialize video player for playback

      _videoController = VideoPlayerController.file(File(file.path))

        ..initialize().then((_) {

          setState(() {});

        });



    } catch (e) {

      print('Error stopping recording: $e');

    }

  }



  void _playRecordedVideo() {

    if (_videoController != null && _videoController!.value.isInitialized) {

      setState(() {

        _isPlaying = true;

      });

      _videoController!.play();

    }

  }



  void _stopVideo() {

    if (_videoController != null) {

      setState(() {

        _isPlaying = false;

      });

      _videoController!.pause();

    }

  }



  void _resetVideo() {

    _recordingTimer?.cancel();

    setState(() {

      _recordedVideoPath = null;

      _isPlaying = false;

      _secondsElapsed = 0;

    });

    _videoController?.dispose();

    _videoController = null;

  }



  @override

  Widget build(BuildContext context) {

    final theme = Theme.of(context);

    final isArabic = context.watch<AppSettingsProvider>().isArabicLocale;



    return Scaffold(

      backgroundColor: Colors.black,

      appBar: AppBar(

        title: Text(isArabic ? 'تسجيل الفيديو' : 'Video Recording'),

        backgroundColor: Colors.black,

        foregroundColor: Colors.white,

        actions: [

          Consumer<SignDetectionProvider>(

            builder: (context, provider, child) {

              return Padding(

                padding: const EdgeInsets.symmetric(horizontal: 16),

                child: Row(

                  children: [

                    Icon(Icons.emoji_events, color: Colors.yellow, size: 20),

                    SizedBox(width: 4),

                    Text(

                      '${provider.userPoints}',

                      style: TextStyle(

                        color: Colors.white,

                        fontWeight: FontWeight.bold,

                        fontSize: 16,

                      ),

                    ),

                  ],

                ),

              );

            },

          ),

        ],

      ),



      body: Consumer<SignDetectionProvider>(

        builder: (context, provider, child) {

          if (provider.state == DetectionState.error && provider.errorMessage != null) {

            return _buildErrorView(provider.errorMessage!, isArabic);

          }



          return Stack(

            children: [

// Camera preview or video player

              _buildMediaPreview(),



// Detection overlay (shows current detection if any)

              _buildDetectionOverlay(context, provider, isArabic),



// Control buttons

              _buildControlButtons(context, provider, isArabic),

            ],

          );

        },

      ),

    );

  }



  Widget _buildMediaPreview() {

    if (_recordedVideoPath != null && _videoController != null && _videoController!.value.isInitialized) {

// Show recorded video

      return AspectRatio(

        aspectRatio: _videoController!.value.aspectRatio,

        child: VideoPlayer(_videoController!),

      );

    } else if (_isCameraInitialized && _cameraController != null) {

// Show camera preview

      return CameraPreview(_cameraController!);

    } else {

// Show loading/placeholder

      return Center(

        child: Column(

          mainAxisAlignment: MainAxisAlignment.center,

          children: [

            CircularProgressIndicator(color: Colors.white),

            SizedBox(height: 16),

            Text(

              'جاري تحضير الكاميرا...',

              style: TextStyle(color: Colors.white, fontSize: 16),

            ),

          ],

        ),

      );

    }

  }



  Widget _buildErrorView(String error, bool isArabic) {

    return Center(

      child: Padding(

        padding: const EdgeInsets.all(24.0),

        child: Column(

          mainAxisAlignment: MainAxisAlignment.center,

          children: [

            Icon(Icons.error_outline, color: Colors.red, size: 64),

            SizedBox(height: 16),

            Text(

              isArabic ? 'خطأ' : 'Error',

              style: TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold),

            ),

            SizedBox(height: 8),

            Text(

              error,

              style: TextStyle(color: Colors.white70, fontSize: 16),

              textAlign: TextAlign.center,

            ),

            SizedBox(height: 24),

            ElevatedButton.icon(

              onPressed: () {

// 1. Reset the provider's state back to 'idle'

                context.read<SignDetectionProvider>().resetDetection();



// 2. Re-initialize the camera

                _initializeCamera();

              },

              icon: Icon(Icons.refresh),

              label: Text(isArabic ? 'إعادة المحاولة' : 'Retry'),

            ),

          ],

        ),

      ),

    );

  }



  Widget _buildDetectionOverlay(

      BuildContext context,

      SignDetectionProvider provider,

      bool isArabic,

      ) {

    return Positioned(

      top: 0,

      left: 0,

      right: 0,

      child: Container(

        padding: EdgeInsets.all(16),

        decoration: BoxDecoration(

          gradient: LinearGradient(

            begin: Alignment.topCenter,

            end: Alignment.bottomCenter,

            colors: [Colors.black.withOpacity(0.7), Colors.transparent],

          ),

        ),

        child: Column(

          children: [

// Recording status

            _buildRecordingStatus(isArabic),

            SizedBox(height: 16),

// Current detection result (if any from previous detections)

            if (provider.currentSign != null)

              _buildDetectionResult(provider.currentSign!, provider.confidence, isArabic),

          ],

        ),

      ),

    );

  }



  Widget _buildRecordingStatus(bool isArabic) {

    String statusText;

    Color statusColor;

    IconData statusIcon;



    if (_isRecording) {

      statusText = isArabic ? 'جاري التسجيل...' : 'Recording...';

      statusColor = _secondsElapsed < _minRecordTime ? Colors.orange : Colors.red;

      statusIcon = _secondsElapsed < _minRecordTime ? Icons.timer_outlined : Icons.fiber_manual_record;

    } else if (_recordedVideoPath != null) {

      statusText = isArabic ? 'تم التسجيل' : 'Recorded';

      statusColor = Colors.green;

      statusIcon = Icons.check_circle;

    } else {

      statusText = isArabic ? 'جاهز للتسجيل' : 'Ready to Record';

      statusColor = Colors.blue;

      statusIcon = Icons.videocam;

    }



    return Container(

      padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),

      decoration: BoxDecoration(

        color: statusColor.withOpacity(0.2),

        borderRadius: BorderRadius.circular(20),

        border: Border.all(color: statusColor, width: 1),

      ),

      child: Row(

        mainAxisSize: MainAxisSize.min,

        children: [

          Icon(statusIcon, color: statusColor, size: 20),

          SizedBox(width: 8),

          Text(

            statusText,

            style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600),

          ),



          if (_isRecording)

            Text(

              ' ($_secondsElapsed' + (isArabic ? ' ث' : 's') + ')',

              style: TextStyle(

                  color: _secondsElapsed < _minRecordTime ? Colors.orange : Colors.white,

                  fontWeight: FontWeight.bold

              ),

            ),



          if (_isRecording && _secondsElapsed < _minRecordTime) ...[

            SizedBox(width: 8),

            SizedBox(

              width: 12,

              height: 12,

              child: CircularProgressIndicator(

                strokeWidth: 2,

                valueColor: AlwaysStoppedAnimation<Color>(statusColor),

              ),

            ),

          ],

        ],

      ),

    );

  }



  Widget _buildDetectionResult(SignModel sign, double confidence, bool isArabic) {

    final showArabicLabels = context.watch<AppSettingsProvider>().showArabicLabels;



    return Container(

      width: double.infinity,

      padding: EdgeInsets.all(16),

      decoration: BoxDecoration(

        color: Colors.white.withOpacity(0.9),

        borderRadius: BorderRadius.circular(12),

        boxShadow: [

          BoxShadow(

            color: Colors.black.withOpacity(0.2),

            blurRadius: 10,

            offset: Offset(0, 4),

          ),

        ],

      ),

      child: Column(

        children: [

// Points earned badge

          Consumer<SignDetectionProvider>(

            builder: (context, provider, child) {

              if (provider.recentPoints.isNotEmpty) {

                final recentPoints = provider.recentPoints.first;

                return Container(

                  padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),

                  decoration: BoxDecoration(

                    color: Colors.green.withOpacity(0.2),

                    borderRadius: BorderRadius.circular(20),

                    border: Border.all(color: Colors.green),

                  ),

                  child: Row(

                    mainAxisSize: MainAxisSize.min,

                    children: [

                      Icon(Icons.emoji_events, size: 16, color: Colors.green),

                      SizedBox(width: 4),

                      Text(

                        '+${recentPoints['points']}',

                        style: TextStyle(

                          color: Colors.green,

                          fontWeight: FontWeight.bold,

                          fontSize: 14,

                        ),

                      ),

                    ],

                  ),

                );

              }

              return SizedBox();

            },

          ),

          SizedBox(height: 12),



// Sign icon

          Container(

            padding: EdgeInsets.all(12),

            decoration: BoxDecoration(

              color: Theme.of(context).colorScheme.primary.withOpacity(0.1),

              shape: BoxShape.circle,

            ),

            child: Icon(

              Icons.sign_language,

              size: 32,

              color: Theme.of(context).colorScheme.primary,

            ),

          ),



          SizedBox(height: 12),



// Sign labels

          if (showArabicLabels && isArabic) ...[

            Text(

              sign.arabicLabel,

              style: TextStyle(

                fontSize: 24,

                fontWeight: FontWeight.bold,

                color: Colors.black87,

                fontFamily: 'Amiri',

              ),

              textAlign: TextAlign.center,

            ),

            SizedBox(height: 4),

            Text(

              sign.englishLabel,

              style: TextStyle(

                fontSize: 16,

                color: Colors.black54,

              ),

              textAlign: TextAlign.center,

            ),

          ] else ...[

            Text(

              sign.englishLabel,

              style: TextStyle(

                fontSize: 24,

                fontWeight: FontWeight.bold,

                color: Colors.black87,

              ),

              textAlign: TextAlign.center,

            ),

            if (showArabicLabels) ...[

              SizedBox(height: 4),

              Text(

                sign.arabicLabel,

                style: TextStyle(

                  fontSize: 16,

                  color: Colors.black54,

                  fontFamily: 'Amiri',

                ),

                textAlign: TextAlign.center,

              ),

            ],

          ],



          SizedBox(height: 12),



// Confidence indicator

          Row(

            mainAxisAlignment: MainAxisAlignment.center,

            children: [

              Icon(

                Icons.precision_manufacturing,

                size: 16,

                color: _getConfidenceColor(confidence),

              ),

              SizedBox(width: 4),

              Text(

                '${(confidence * 100).toStringAsFixed(1)}%',

                style: TextStyle(

                  fontSize: 14,

                  fontWeight: FontWeight.w600,

                  color: _getConfidenceColor(confidence),

                ),

              ),

              SizedBox(width: 8),

              Text(

                isArabic ? 'دقة' : 'Confidence',

                style: TextStyle(

                  fontSize: 14,

                  color: Colors.black54,

                ),

              ),

            ],

          ),

        ],

      ),

    );

  }



  Widget _buildControlButtons(

      BuildContext context,

      SignDetectionProvider provider,

      bool isArabic,

      ) {

// Get settings for confidence threshold

    final settings = context.watch<AppSettingsProvider>();



    return Positioned(

      bottom: 50,

      left: 0,

      right: 0,

      child: Column(

        children: [

// Loading indicator when analyzing

          if (provider.state == DetectionState.detecting)

            const Center(

              child: SizedBox(

                width: 80,

                height: 80,

                child: CircularProgressIndicator(

                  color: Colors.white,

                  strokeWidth: 6,

                ),

              ),

            )

// Main record/play/analyze buttons

          else if (_recordedVideoPath == null)

// Show Record Button

            FloatingActionButton.large(

              onPressed: _isRecording ? _stopRecording : _startRecording,

              backgroundColor: _isRecording ? Colors.red : Colors.green,

              heroTag: 'record_button',

              child: Icon(

                _isRecording ? Icons.stop : Icons.videocam,

                size: 32,

              ),

            )

          else

// Show Analyze Button

            FloatingActionButton.large(

              onPressed: () {

                if (_recordedVideoPath != null) {

// --- FIX 3: Read left-handed setting from provider ---

                  final isLeftHanded = context.read<AppSettingsProvider>().isLeftHanded;



                  provider.analyzeRecordedVideo(

                    _recordedVideoPath!,

                    settings.confidenceThreshold,

                    isLeftHanded: isLeftHanded, // <-- Pass it here

                  );

                }

              },

              backgroundColor: Colors.blue,

              heroTag: 'analyze_button',

              child: const Icon(

                Icons.analytics,

                size: 32,

              ),

            ),



          const SizedBox(height: 20),



// Secondary buttons

          Row(

            mainAxisAlignment: MainAxisAlignment.spaceEvenly,

            children: [

// Reset button

              if (_recordedVideoPath != null && provider.state != DetectionState.detecting)

                FloatingActionButton(

                  onPressed: () {

                    _resetVideo();

                    provider.resetDetection(); // Also reset provider state

                  },

                  backgroundColor: Colors.red,

                  heroTag: 'reset',

                  child: const Icon(Icons.refresh),

                ),



// Play button (only if video is recorded and not analyzing)

              if (_recordedVideoPath != null && provider.state != DetectionState.detecting)

                FloatingActionButton(

                  onPressed: _isPlaying ? _stopVideo : _playRecordedVideo,

                  backgroundColor: _isPlaying ? Colors.orange : Colors.green,

                  heroTag: 'play',

                  child: Icon(_isPlaying ? Icons.stop : Icons.play_arrow),

                ),



// Points info button

              FloatingActionButton(

                onPressed: () => _showPointsDialog(context, provider, isArabic),

                backgroundColor: Colors.orange,

                heroTag: 'points_info',

                child: const Icon(Icons.emoji_events),

              ),



// History button

              FloatingActionButton(

                onPressed: () {

                  Navigator.pushNamed(context, '/history');

                },

                backgroundColor: Colors.purple,

                heroTag: 'history',

                child: const Icon(Icons.history),

              ),

            ],

          ),

        ],

      ),

    );

  }



  Color _getConfidenceColor(double confidence) {

    if (confidence >= 0.8) return Colors.green;

    if (confidence >= 0.6) return Colors.orange;

    return Colors.red;

  }



  void _showPointsDialog(BuildContext context, SignDetectionProvider provider, bool isArabic) {

    final progress = provider.getUserProgress();



    showDialog(

      context: context,

      builder: (context) => AlertDialog(

        title: Text(isArabic ? 'النقاط والمستوى' : 'Points & Level'),

        content: Column(

          mainAxisSize: MainAxisSize.min,

          children: [

            Icon(Icons.emoji_events, size: 64, color: Colors.amber),

            SizedBox(height: 16),

            Text(

              '${progress['points']} ${isArabic ? 'نقطة' : 'Points'}',

              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),

            ),

            SizedBox(height: 8),

            Text('${isArabic ? 'المستوى' : 'Level'} ${progress['level']}'),

            SizedBox(height: 8),

            LinearProgressIndicator(

              value: progress['progress'],

              backgroundColor: Colors.grey[300],

              color: Colors.green,

            ),

            SizedBox(height: 8),

            Text(

              '${progress['pointsToNextLevel']} ${isArabic ? 'نقطة للمستوى التالي' : 'points to next level'}',

              style: TextStyle(fontSize: 12, color: Colors.grey),

            ),

            SizedBox(height: 16),

            Row(

              mainAxisAlignment: MainAxisAlignment.spaceAround,

              children: [

                Column(

                  children: [

                    Text(

                      '${progress['correctDetections']}',

                      style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),

                    ),

                    Text(isArabic ? 'صحيح' : 'Correct', style: TextStyle(fontSize: 12)),

                  ],

                ),

                Column(

                  children: [

                    Text(

                      '${(progress['accuracy'] * 100).toStringAsFixed(1)}%',

                      style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),

                    ),

                    Text(isArabic ? 'دقة' : 'Accuracy', style: TextStyle(fontSize: 12)),

                  ],

                ),

              ],

            ),

          ],

        ),

        actions: [

          TextButton(

            onPressed: () => Navigator.of(context).pop(),

            child: Text(isArabic ? 'إغلاق' : 'Close'),

          ),

        ],

      ),

    );

  }

}