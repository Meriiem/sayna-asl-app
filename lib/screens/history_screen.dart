import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/sign_detection_provider.dart';
import '../providers/app_settings_provider.dart';
import '../models/sign_model.dart';

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final isArabic = context.watch<AppSettingsProvider>().isArabicLocale;

    return Scaffold(
      appBar: AppBar(
        title: Text(isArabic ? 'سجل الكشوفات' : 'Detection History'),
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
          Consumer<SignDetectionProvider>(
            builder: (context, provider, child) {
              if (provider.detectionHistory.isEmpty) return const SizedBox();

              return IconButton(
                icon: const Icon(Icons.clear_all),
                onPressed: () => _showClearDialog(context, provider, isArabic),
                tooltip: isArabic ? 'مسح الكل' : 'Clear All',
              );
            },
          ),
        ],
      ),

      body: Consumer<SignDetectionProvider>(
        builder: (context, provider, child) {
          if (provider.detectionHistory.isEmpty) {
            return _buildEmptyState(context, isArabic);
          }

          return Column(
            children: [
              // Statistics summary
              _buildStatsSummary(context, provider, isArabic),

              // History list
              Expanded(
                child: _buildHistoryList(context, provider, isArabic),
              ),
            ],
          );
        },
      ),
    );
  }

  Widget _buildEmptyState(BuildContext context, bool isArabic) {
    final theme = Theme.of(context);

    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.history,
            size: 80,
            color: theme.colorScheme.onSurface.withOpacity(0.3),
          ),

          const SizedBox(height: 16),

          Text(
            isArabic ? 'لا توجد كشوفات بعد' : 'No detections yet',
            style: theme.textTheme.headlineSmall?.copyWith(
              color: theme.colorScheme.onSurface.withOpacity(0.6),
            ),
          ),

          const SizedBox(height: 8),

          Text(
            isArabic
                ? 'ابدأ بكشف الإشارات لرؤية النتائج هنا'
                : 'Start detecting signs to see results here',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurface.withOpacity(0.5),
            ),
            textAlign: TextAlign.center,
          ),

          const SizedBox(height: 32),

          ElevatedButton.icon(
            onPressed: () => Navigator.pushNamed(context, '/camera'),
            icon: const Icon(Icons.camera_alt),
            label: Text(isArabic ? 'بدء الكشف' : 'Start Detection'),
          ),
        ],
      ),
    );
  }

  Widget _buildStatsSummary(
      BuildContext context,
      SignDetectionProvider provider,
      bool isArabic,
      ) {
    final theme = Theme.of(context);
    final stats = provider.getStatistics();
    final progress = provider.getUserProgress(); // NEW: Get gamification progress

    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: theme.colorScheme.primaryContainer,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          // Gamification stats - NEW SECTION
          Row(
            children: [
              Expanded(
                child: _buildStatItem(
                  context,
                  title: isArabic ? 'النقاط' : 'Points',
                  value: '${progress['points']}',
                  icon: Icons.emoji_events,
                  color: Colors.amber,
                ),
              ),

              Container(
                width: 1,
                height: 40,
                color: theme.colorScheme.onPrimaryContainer.withOpacity(0.2),
              ),

              Expanded(
                child: _buildStatItem(
                  context,
                  title: isArabic ? 'المستوى' : 'Level',
                  value: '${progress['level']}',
                  icon: Icons.star,
                  color: Colors.orange,
                ),
              ),

              Container(
                width: 1,
                height: 40,
                color: theme.colorScheme.onPrimaryContainer.withOpacity(0.2),
              ),

              Expanded(
                child: _buildStatItem(
                  context,
                  title: isArabic ? 'الدقة' : 'Accuracy',
                  value: '${(progress['accuracy'] * 100).toStringAsFixed(0)}%',
                  icon: Icons.precision_manufacturing,
                  color: Colors.blue,
                ),
              ),
            ],
          ),

          const SizedBox(height: 12),

          // Original stats
          Row(
            children: [
              Expanded(
                child: _buildStatItem(
                  context,
                  title: isArabic ? 'المجموع' : 'Total',
                  value: stats['totalDetections'].toString(),
                  icon: Icons.analytics,
                  color: theme.colorScheme.onPrimaryContainer,
                ),
              ),

              Container(
                width: 1,
                height: 40,
                color: theme.colorScheme.onPrimaryContainer.withOpacity(0.2),
              ),

              Expanded(
                child: _buildStatItem(
                  context,
                  title: isArabic ? 'الإشارات' : 'Signs',
                  value: stats['uniqueSigns'].toString(),
                  icon: Icons.gesture,
                  color: theme.colorScheme.onPrimaryContainer,
                ),
              ),

              Container(
                width: 1,
                height: 40,
                color: theme.colorScheme.onPrimaryContainer.withOpacity(0.2),
              ),

              Expanded(
                child: _buildStatItem(
                  context,
                  title: isArabic ? 'متوسط الثقة' : 'Avg Confidence',
                  value: '${(stats['averageConfidence'] * 100).toStringAsFixed(0)}%',
                  icon: Icons.psychology,
                  color: theme.colorScheme.onPrimaryContainer,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildStatItem(
      BuildContext context, {
        required String title,
        required String value,
        required IconData icon,
        required Color color,
      }) {
    final theme = Theme.of(context);

    return Column(
      children: [
        Icon(
          icon,
          color: color,
          size: 20,
        ),

        const SizedBox(height: 4),

        Text(
          value,
          style: theme.textTheme.titleLarge?.copyWith(
            color: color,
            fontWeight: FontWeight.bold,
          ),
        ),

        Text(
          title,
          style: theme.textTheme.bodySmall?.copyWith(
            color: color.withOpacity(0.8),
          ),
          textAlign: TextAlign.center,
        ),
      ],
    );
  }

  Widget _buildHistoryList(
      BuildContext context,
      SignDetectionProvider provider,
      bool isArabic,
      ) {
    final showArabicLabels = context.watch<AppSettingsProvider>().showArabicLabels;

    return ListView.builder(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      itemCount: provider.detectionHistory.length,
      itemBuilder: (context, index) {
        final detection = provider.detectionHistory[index];
        return _buildHistoryItem(
          context,
          detection,
          index,
          isArabic,
          showArabicLabels,
        );
      },
    );
  }

  Widget _buildHistoryItem(
      BuildContext context,
      SignModel detection,
      int index,
      bool isArabic,
      bool showArabicLabels,
      ) {
    final theme = Theme.of(context);
    final timeAgo = _getTimeAgo(detection.detectedAt!, isArabic);

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        leading: Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: _getConfidenceColor(detection.confidence).withOpacity(0.1),
            shape: BoxShape.circle,
          ),
          child: Icon(
            Icons.sign_language,
            color: _getConfidenceColor(detection.confidence),
            size: 24,
          ),
        ),

        title: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (showArabicLabels && isArabic) ...[
                    Text(
                      detection.arabicLabel,
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w600,
                        fontFamily: 'Amiri',
                      ),
                    ),
                    Text(
                      detection.englishLabel,
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.onSurface.withOpacity(0.7),
                      ),
                    ),
                  ] else ...[
                    Text(
                      detection.englishLabel,
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    if (showArabicLabels)
                      Text(
                        detection.arabicLabel,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onSurface.withOpacity(0.7),
                          fontFamily: 'Amiri',
                        ),
                      ),
                  ],
                ],
              ),
            ),

            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                // Points earned for this detection - NEW
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: Colors.green.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.emoji_events, size: 10, color: Colors.green),
                      SizedBox(width: 2),
                      Text(
                        '+${_calculatePointsForDetection(detection.confidence)}',
                        style: TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                          color: Colors.green,
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 4),

                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: _getConfidenceColor(detection.confidence).withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    '${(detection.confidence * 100).toStringAsFixed(1)}%',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: _getConfidenceColor(detection.confidence),
                    ),
                  ),
                ),

                const SizedBox(height: 4),

                Text(
                  timeAgo,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurface.withOpacity(0.5),
                  ),
                ),
              ],
            ),
          ],
        ),

        onTap: () => _showDetectionDetails(context, detection, isArabic),
      ),
    );
  }

  // NEW: Calculate points for a detection based on confidence
  int _calculatePointsForDetection(double confidence) {
    int basePoints = 10;
    if (confidence >= 0.9) return basePoints + 8;
    if (confidence >= 0.8) return basePoints + 5;
    if (confidence >= 0.7) return basePoints + 2;
    return basePoints;
  }

  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.8) return Colors.green;
    if (confidence >= 0.6) return Colors.orange;
    return Colors.red;
  }

  String _getTimeAgo(DateTime dateTime, bool isArabic) {
    final now = DateTime.now();
    final difference = now.difference(dateTime);

    if (difference.inDays > 0) {
      return isArabic
          ? '${difference.inDays} ${difference.inDays == 1 ? 'يوم' : 'أيام'}'
          : '${difference.inDays}d ago';
    } else if (difference.inHours > 0) {
      return isArabic
          ? '${difference.inHours} ${difference.inHours == 1 ? 'ساعة' : 'ساعات'}'
          : '${difference.inHours}h ago';
    } else if (difference.inMinutes > 0) {
      return isArabic
          ? '${difference.inMinutes} ${difference.inMinutes == 1 ? 'دقيقة' : 'دقائق'}'
          : '${difference.inMinutes}m ago';
    } else {
      return isArabic ? 'الآن' : 'Now';
    }
  }

  void _showDetectionDetails(BuildContext context, SignModel detection, bool isArabic) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(
          isArabic ? 'تفاصيل الكشف' : 'Detection Details',
          textAlign: TextAlign.center,
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: _getConfidenceColor(detection.confidence).withOpacity(0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.sign_language,
                size: 48,
                color: _getConfidenceColor(detection.confidence),
              ),
            ),

            const SizedBox(height: 16),

            // Points earned - NEW
            Container(
              padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.green.withOpacity(0.1),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Colors.green),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.emoji_events, size: 16, color: Colors.green),
                  SizedBox(width: 4),
                  Text(
                    '+${_calculatePointsForDetection(detection.confidence)} ${isArabic ? 'نقطة' : 'Points'}',
                    style: TextStyle(
                      color: Colors.green,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 16),

            Text(
              isArabic ? detection.arabicLabel : detection.englishLabel,
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
                fontFamily: isArabic ? 'Amiri' : null,
              ),
              textAlign: TextAlign.center,
            ),

            const SizedBox(height: 8),

            Text(
              isArabic ? detection.englishLabel : detection.arabicLabel,
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                color: Theme.of(context).colorScheme.onSurface.withOpacity(0.7),
                fontFamily: isArabic ? null : 'Amiri',
              ),
              textAlign: TextAlign.center,
            ),

            const SizedBox(height: 16),

            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                Column(
                  children: [
                    Text(
                      '${(detection.confidence * 100).toStringAsFixed(1)}%',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: _getConfidenceColor(detection.confidence),
                      ),
                    ),
                    Text(
                      isArabic ? 'الدقة' : 'Confidence',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ],
                ),

                Column(
                  children: [
                    Text(
                      _getTimeAgo(detection.detectedAt!, isArabic),
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      isArabic ? 'الوقت' : 'Time',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
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

  void _showClearDialog(
      BuildContext context,
      SignDetectionProvider provider,
      bool isArabic,
      ) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(isArabic ? 'مسح السجل' : 'Clear History'),
        content: Text(
          isArabic
              ? 'هل أنت متأكد من أنك تريد مسح جميع الكشوفات؟'
              : 'Are you sure you want to clear all detection history?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: Text(isArabic ? 'إلغاء' : 'Cancel'),
          ),
          TextButton(
            onPressed: () {
              provider.clearHistory();
              Navigator.of(context).pop();
            },
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: Text(isArabic ? 'مسح' : 'Clear'),
          ),
        ],
      ),
    );
  }
}