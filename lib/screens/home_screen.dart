import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/sign_detection_provider.dart';
import '../providers/app_settings_provider.dart';
import '../models/sign_model.dart';
import '../widgets/sign_card.dart';
import '../widgets/stats_card.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:image_picker/image_picker.dart';
import 'package:video_player/video_player.dart';
import 'dart:io';
import '../widgets/video_preview_dialog.dart';


class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

  class _HomeScreenState extends State<HomeScreen> {

    final ImagePicker _picker = ImagePicker(); // keep this in the class

    Future<void> _pickVideoForDetection(BuildContext context) async {
      try {
        final XFile? picked = await _picker.pickVideo(
          source: ImageSource.gallery,
        );

        if (picked == null) {
          // User cancelled
          return;
        }

        final file = File(picked.path);

        // 👇 instead of detecting immediately, show preview dialog
        final isArabic = context
            .read<AppSettingsProvider>()
            .isArabicLocale;
        _showVideoPreviewDialog(context, file, isArabic);
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to pick video: $e')),
        );
      }
    }


    Widget build(BuildContext context) {
      final theme = Theme.of(context);
      final isArabic = context
          .watch<AppSettingsProvider>()
          .isArabicLocale;

      return Scaffold(
        appBar: AppBar(
          title: Text(
              isArabic ? 'لغة الإشارة العربية' : 'Arabic Sign Language'),
          actions: [
            // Add points display in app bar - NEW
            Consumer<SignDetectionProvider>(
              builder: (context, provider, child) {
                return Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 8),
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
            IconButton(
              icon: const Icon(Icons.history),
              onPressed: () => Navigator.pushNamed(context, '/history'),
              tooltip: isArabic ? 'السجل' : 'History',
            ),
            IconButton(
              icon: const Icon(Icons.settings),
              onPressed: () => Navigator.pushNamed(context, '/settings'),
              tooltip: isArabic ? 'الإعدادات' : 'Settings',
            ),
          ],
        ),

        body: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Welcome section
              _buildWelcomeSection(context, isArabic),

              const SizedBox(height: 24),

              // Quick stats
              _buildStatsSection(context, isArabic),

              const SizedBox(height: 24),

              // Gamification stats - NEW SECTION
              _buildGamificationSection(context, isArabic),

              const SizedBox(height: 24),

              // Available signs section
              _buildSignsSection(context, isArabic),

              const SizedBox(height: 24),

              // Quick actions
              _buildQuickActions(context, isArabic),
            ],
          ),
        ),

        floatingActionButton: FloatingActionButton.extended(
          onPressed: () => _showStartDetectionSheet(context, isArabic),
          icon: const Icon(Icons.videocam),
          label: Text(isArabic ? 'بدء الكشف' : 'Start Detection'),
        ),

      );
    }

    Widget _buildWelcomeSection(BuildContext context, bool isArabic) {
      final theme = Theme.of(context);
      final userPoints = context
          .watch<SignDetectionProvider>()
          .userPoints;
      final userRank = context.watch<SignDetectionProvider>().getUserRank();

      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              theme.colorScheme.primary,
              theme.colorScheme.secondary,
            ],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.sign_language,
                  size: 40,
                  color: theme.colorScheme.onPrimary,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        isArabic ? 'مرحباً بك!' : 'Welcome!',
                        style: theme.textTheme.headlineMedium?.copyWith(
                          color: theme.colorScheme.onPrimary,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        isArabic
                            ? 'اكتشف الإشارات العربية بالذكاء الاصطناعي'
                            : 'Discover Arabic signs with AI',
                        style: theme.textTheme.bodyLarge?.copyWith(
                          color: theme.colorScheme.onPrimary.withOpacity(0.9),
                        ),
                      ),
                    ],
                  ),
                ),
                // Add points display - NEW
                Container(
                  padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: theme.colorScheme.onPrimary.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Column(
                    children: [
                      Row(
                        children: [
                          Icon(Icons.emoji_events, size: 16,
                              color: Colors.yellow),
                          SizedBox(width: 4),
                          Text(
                            '$userPoints',
                            style: TextStyle(
                              color: theme.colorScheme.onPrimary,
                              fontWeight: FontWeight.bold,
                              fontSize: 16,
                            ),
                          ),
                        ],
                      ),
                      SizedBox(height: 2),
                      Text(
                        userRank,
                        style: TextStyle(
                          color: theme.colorScheme.onPrimary,
                          fontSize: 10,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ),
      );
    }

    Widget _buildStatsSection(BuildContext context, bool isArabic) {
      return Consumer<SignDetectionProvider>(
        builder: (context, provider, child) {
          final stats = provider.getStatistics();

          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                isArabic ? 'الإحصائيات' : 'Statistics',
                style: Theme
                    .of(context)
                    .textTheme
                    .headlineSmall
                    ?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),

              const SizedBox(height: 12),

              Row(
                children: [
                  Expanded(
                    child: StatsCard(
                      title: isArabic ? 'إجمالي الكشوفات' : 'Total Detections',
                      value: stats['totalDetections'].toString(),
                      icon: Icons.analytics,
                      color: Colors.blue,
                    ),
                  ),

                  const SizedBox(width: 12),

                  Expanded(
                    child: StatsCard(
                      title: isArabic ? 'الإشارات المكتشفة' : 'Unique Signs',
                      value: stats['uniqueSigns'].toString(),
                      icon: Icons.gesture,
                      color: Colors.green,
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 12),

              Row(
                children: [
                  Expanded(
                    child: StatsCard(
                      title: isArabic ? 'متوسط الدقة' : 'Avg Accuracy',
                      value: '${(stats['averageConfidence'] * 100)
                          .toStringAsFixed(1)}%',
                      icon: Icons.precision_manufacturing,
                      color: Colors.orange,
                    ),
                  ),

                  const SizedBox(width: 12),

                  Expanded(
                    child: StatsCard(
                      title: isArabic ? 'الأكثر كشفاً' : 'Most Detected',
                      value: stats['mostDetectedSign']?.englishLabel ?? 'None',
                      icon: Icons.star,
                      color: Colors.purple,
                    ),
                  ),
                ],
              ),
            ],
          );
        },
      );
    }

    // NEW SECTION: Gamification Stats
    Widget _buildGamificationSection(BuildContext context, bool isArabic) {
      return Consumer<SignDetectionProvider>(
        builder: (context, provider, child) {
          final progress = provider.getUserProgress();

          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    isArabic ? 'التقدم والمستويات' : 'Progress & Levels',
                    style: Theme
                        .of(context)
                        .textTheme
                        .headlineSmall
                        ?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  IconButton(
                    icon: Icon(Icons.emoji_events, color: Colors.amber),
                    onPressed: () =>
                        _showDetailedProgress(context, provider, isArabic),
                    tooltip: isArabic ? 'تفاصيل التقدم' : 'Progress Details',
                  ),
                ],
              ),

              const SizedBox(height: 12),

              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    children: [
                      // Level and progress
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                isArabic ? 'المستوى' : 'Level',
                                style: Theme
                                    .of(context)
                                    .textTheme
                                    .bodySmall
                                    ?.copyWith(
                                  color: Theme
                                      .of(context)
                                      .colorScheme
                                      .onSurface
                                      .withOpacity(0.7),
                                ),
                              ),
                              Text(
                                '${progress['level']}',
                                style: Theme
                                    .of(context)
                                    .textTheme
                                    .headlineMedium
                                    ?.copyWith(
                                  fontWeight: FontWeight.bold,
                                  color: Theme
                                      .of(context)
                                      .colorScheme
                                      .primary,
                                ),
                              ),
                            ],
                          ),
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.end,
                            children: [
                              Text(
                                isArabic ? 'النقاط' : 'Points',
                                style: Theme
                                    .of(context)
                                    .textTheme
                                    .bodySmall
                                    ?.copyWith(
                                  color: Theme
                                      .of(context)
                                      .colorScheme
                                      .onSurface
                                      .withOpacity(0.7),
                                ),
                              ),
                              Text(
                                '${progress['points']}',
                                style: Theme
                                    .of(context)
                                    .textTheme
                                    .headlineMedium
                                    ?.copyWith(
                                  fontWeight: FontWeight.bold,
                                  color: Colors.amber,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),

                      const SizedBox(height: 12),

                      // Progress bar
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(
                                isArabic
                                    ? 'التقدم للمستوى التالي'
                                    : 'Progress to next level',
                                style: Theme
                                    .of(context)
                                    .textTheme
                                    .bodySmall,
                              ),
                              Text(
                                '${progress['pointsToNextLevel']} ${isArabic
                                    ? 'نقطة'
                                    : 'points'}',
                                style: Theme
                                    .of(context)
                                    .textTheme
                                    .bodySmall
                                    ?.copyWith(
                                  color: Theme
                                      .of(context)
                                      .colorScheme
                                      .primary,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 8),
                          LinearProgressIndicator(
                            value: progress['progress'],
                            backgroundColor: Colors.grey[300],
                            color: Theme
                                .of(context)
                                .colorScheme
                                .primary,
                            minHeight: 8,
                            borderRadius: BorderRadius.circular(4),
                          ),
                        ],
                      ),

                      const SizedBox(height: 8),

                      // Additional stats
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceAround,
                        children: [
                          Column(
                            children: [
                              Text(
                                '${progress['correctDetections']}',
                                style: TextStyle(
                                  fontWeight: FontWeight.bold,
                                  fontSize: 16,
                                  color: Colors.green,
                                ),
                              ),
                              Text(
                                isArabic ? 'صحيح' : 'Correct',
                                style: Theme
                                    .of(context)
                                    .textTheme
                                    .bodySmall,
                              ),
                            ],
                          ),
                          Column(
                            children: [
                              Text(
                                '${(progress['accuracy'] * 100).toStringAsFixed(
                                    1)}%',
                                style: TextStyle(
                                  fontWeight: FontWeight.bold,
                                  fontSize: 16,
                                  color: Colors.blue,
                                ),
                              ),
                              Text(
                                isArabic ? 'دقة' : 'Accuracy',
                                style: Theme
                                    .of(context)
                                    .textTheme
                                    .bodySmall,
                              ),
                            ],
                          ),
                          Column(
                            children: [
                              Text(
                                '${progress['totalAttempts']}',
                                style: TextStyle(
                                  fontWeight: FontWeight.bold,
                                  fontSize: 16,
                                  color: Colors.orange,
                                ),
                              ),
                              Text(
                                isArabic ? 'محاولة' : 'Attempts',
                                style: Theme
                                    .of(context)
                                    .textTheme
                                    .bodySmall,
                              ),
                            ],
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ],
          );
        },
      );
    }

    Widget _buildSignsSection(BuildContext context, bool isArabic) {
      final showArabicLabels = context
          .watch<AppSettingsProvider>()
          .showArabicLabels;

      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                isArabic ? 'الإشارات المتاحة' : 'Available Signs',
                style: Theme
                    .of(context)
                    .textTheme
                    .headlineSmall
                    ?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),

              Text(
                '${SignModel.allSigns.length} ${isArabic ? 'إشارة' : 'signs'}',
                style: Theme
                    .of(context)
                    .textTheme
                    .bodyMedium
                    ?.copyWith(
                  color: Theme
                      .of(context)
                      .colorScheme
                      .primary,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),

          const SizedBox(height: 12),

          GridView.builder(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2,
              childAspectRatio: 1.2,
              crossAxisSpacing: 12,
              mainAxisSpacing: 12,
            ),
            itemCount: SignModel.allSigns.length,
            itemBuilder: (context, index) {
              final sign = SignModel.allSigns[index];
              return SignCard(
                sign: sign,
                showArabicLabel: showArabicLabels,
                onTap: () => _showSignDetails(context, sign, isArabic),
              );
            },
          ),
        ],
      );
    }



    Widget _buildQuickActions(BuildContext context, bool isArabic) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            isArabic ? 'إجراءات سريعة' : 'Quick Actions',
            style: Theme.of(context).textTheme.headlineSmall?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),

          const SizedBox(height: 12),

          // NEW ROW → With Upload button included
          Row(
            children: [
              Expanded(
                child: _buildActionCard(
                  context,
                  title: isArabic ? 'رفع فيديو' : 'Upload Video',
                  subtitle: isArabic ? 'استخدم ملف الفيديو' : 'Use video file',
                  icon: Icons.upload_file,
                  color: Colors.green,
                  onTap: () => _pickVideoForDetection(context),
                ),
              ),

              const SizedBox(width: 12),

              Expanded(
                child: _buildActionCard(
                  context,
                  title: isArabic ? 'كشف مباشر' : 'Live Detection',
                  subtitle: isArabic ? 'استخدم الكاميرا' : 'Use camera',
                  icon: Icons.videocam,
                  color: Colors.red,
                  onTap: () => Navigator.pushNamed(context, '/camera'),
                ),
              ),
            ],
          ),

          const SizedBox(height: 12),

          // SECOND ROW → History alone (or you can add more later)
          Row(
            children: [
              Expanded(
                child: _buildActionCard(
                  context,
                  title: isArabic ? 'السجل' : 'History',
                  subtitle: isArabic ? 'عرض النتائج' : 'View results',
                  icon: Icons.history,
                  color: Colors.indigo,
                  onTap: () => Navigator.pushNamed(context, '/history'),
                ),
              ),
            ],
          ),
        ],
      );
    }



    Widget _buildActionCard(BuildContext context, {
      required String title,
      required String subtitle,
      required IconData icon,
      required Color color,
      required VoidCallback onTap,
    }) {
      return Card(
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(12),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.1),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    icon,
                    color: color,
                    size: 24,
                  ),
                ),

                const SizedBox(height: 8),

                Text(
                  title,
                  style: Theme
                      .of(context)
                      .textTheme
                      .titleMedium
                      ?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                  textAlign: TextAlign.center,
                ),

                Text(
                  subtitle,
                  style: Theme
                      .of(context)
                      .textTheme
                      .bodySmall
                      ?.copyWith(
                    color: Theme
                        .of(context)
                        .colorScheme
                        .onSurface
                        .withOpacity(0.7),
                  ),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      );
    }

    // NEW METHOD: Show detailed progress dialog
    void _showDetailedProgress(BuildContext context,
        SignDetectionProvider provider, bool isArabic) {
      final progress = provider.getUserProgress();
      final userRank = provider.getUserRank();

      showDialog(
        context: context,
        builder: (context) =>
            AlertDialog(
              title: Text(isArabic ? 'تفاصيل التقدم' : 'Progress Details'),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.emoji_events, size: 64, color: Colors.amber),
                  SizedBox(height: 16),

                  Text(
                    userRank,
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Theme
                          .of(context)
                          .colorScheme
                          .primary,
                    ),
                  ),

                  SizedBox(height: 8),

                  Text(
                    '${progress['points']} ${isArabic ? 'نقطة' : 'Points'}',
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                  ),

                  SizedBox(height: 16),

                  LinearProgressIndicator(
                    value: progress['progress'],
                    backgroundColor: Colors.grey[300],
                    color: Colors.green,
                    minHeight: 10,
                  ),

                  SizedBox(height: 8),

                  Text(
                    '${progress['pointsToNextLevel']} ${isArabic
                        ? 'نقطة للمستوى'
                        : 'points to level'} ${progress['level'] + 1}',
                    style: TextStyle(fontSize: 12, color: Colors.grey),
                  ),

                  SizedBox(height: 16),

                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      Column(
                        children: [
                          Text('${progress['correctDetections']}',
                              style: TextStyle(
                                  fontWeight: FontWeight.bold, fontSize: 18)),
                          Text(isArabic ? 'صحيح' : 'Correct',
                              style: TextStyle(fontSize: 12)),
                        ],
                      ),
                      Column(
                        children: [
                          Text('${(progress['accuracy'] * 100).toStringAsFixed(
                              1)}%', style: TextStyle(
                              fontWeight: FontWeight.bold, fontSize: 18)),
                          Text(isArabic ? 'دقة' : 'Accuracy',
                              style: TextStyle(fontSize: 12)),
                        ],
                      ),
                      Column(
                        children: [
                          Text('${progress['totalAttempts']}', style: TextStyle(
                              fontWeight: FontWeight.bold, fontSize: 18)),
                          Text(isArabic ? 'محاولة' : 'Attempts',
                              style: TextStyle(fontSize: 12)),
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


    Future<void> _openYoutubeVideo(String videoId, BuildContext context) async {
      final url = Uri.parse(
          'https://youtu.be/$videoId'); // or https://www.youtube.com/watch?v=$videoId

      if (!await launchUrl(url, mode: LaunchMode.externalApplication)) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Could not open YouTube.')),
        );
      }
    }


    void _showSignDetails(BuildContext context, SignModel sign, bool isArabic) {
      showDialog(
        context: context,
        builder: (ctx) {
          return AlertDialog(
            title: Text(
              isArabic ? sign.arabicLabel : sign.englishLabel,
              textAlign: TextAlign.center,
            ),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const SizedBox(height: 8),
                Icon(Icons.pan_tool, size: 48, color: Theme
                    .of(context)
                    .colorScheme
                    .primary),
                const SizedBox(height: 12),
                Text(
                  isArabic ? 'العربية: ${sign.arabicLabel}' : 'Arabic: ${sign
                      .arabicLabel}',
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 8),
                Text(
                  sign.description,
                  textAlign: TextAlign.center,
                ),
              ],
            ),
            actionsAlignment: MainAxisAlignment.spaceBetween,
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx),
                child: const Text('Close'),
              ),

              if (sign.videoId != null)
                ElevatedButton.icon(
                  icon: const Icon(Icons.play_circle_fill),
                  label: const Text('Learn the sign'),
                  onPressed: () {
                    //debugPrint('Learn the sign pressed for: ${sign.videoId}');
                    // Close the info dialog first
                    Navigator.pop(ctx);

                    // Open YouTube app or browser
                    _openYoutubeVideo(sign.videoId!, context);
                  },
                ),

            ],
          );
        },
      );
    }


    void _showVideoPreviewDialog(BuildContext context,
        File videoFile,
        bool isArabic,) {
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (ctx) =>
            VideoPreviewDialog(
              videoFile: videoFile,
              isArabic: isArabic,
            ),
      );
    }
    void _showStartDetectionSheet(BuildContext context, bool isArabic) {
      showModalBottomSheet(
        context: context,
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
        ),
        builder: (ctx) {
          return SafeArea(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                ListTile(
                  leading: const Icon(Icons.videocam),
                  title: Text(
                    isArabic ? 'كشف مباشر' : 'Live Detection',
                  ),
                  subtitle: Text(
                    isArabic ? 'استخدم الكاميرا' : 'Use camera',
                  ),
                  onTap: () {
                    Navigator.of(ctx).pop();
                    Navigator.pushNamed(context, '/camera');
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.upload_file),
                  title: Text(
                    isArabic ? 'كشف من فيديو' : 'Detect from Video',
                  ),
                  subtitle: Text(
                    isArabic ? 'اختر ملف فيديو من الجهاز' : 'Choose video file from device',
                  ),
                  onTap: () {
                    Navigator.of(ctx).pop();
                    _pickVideoForDetection(context); // 👈 opens your preview dialog
                  },
                ),
              ],
            ),
          );
        },
      );
    }

  }

