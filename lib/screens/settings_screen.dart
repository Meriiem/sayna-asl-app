import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/app_settings_provider.dart';
import '../providers/sign_detection_provider.dart'; // NEW: Import for gamification
import 'package:firebase_auth/firebase_auth.dart'; //to log out

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final isArabic = context.watch<AppSettingsProvider>().isArabicLocale;

    return Scaffold(
      appBar: AppBar(
        title: Text(isArabic ? 'الإعدادات' : 'Settings'),
      ),

      body: Consumer<AppSettingsProvider>(
        builder: (context, settings, child) {
          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              // Gamification section - NEW
              _buildGamificationSection(context, isArabic),

              const SizedBox(height: 24),

              // Appearance section
              _buildSectionHeader(context, isArabic ? 'المظهر' : 'Appearance'),
              _buildThemeSelector(context, settings, isArabic),
              _buildLanguageSelector(context, settings, isArabic),

              const SizedBox(height: 24),

              // Detection section
              _buildSectionHeader(context, isArabic ? 'الكشف' : 'Detection'),
              _buildConfidenceSlider(context, settings, isArabic),
              _buildArabicLabelsToggle(context, settings, isArabic),

              const SizedBox(height: 24),

              // Feedback section
              _buildSectionHeader(context, isArabic ? 'التنبيهات' : 'Feedback'),
              _buildSoundToggle(context, settings, isArabic),
              _buildVibrationToggle(context, settings, isArabic),

              const SizedBox(height: 24),

              // About section
              _buildSectionHeader(context, isArabic ? 'حول التطبيق' : 'About'),
              _buildAboutCard(context, isArabic),

              const SizedBox(height: 16),

              // Reset buttons
              _buildResetButtons(context, settings, isArabic), // UPDATED
            ],
          );
        },
      ),
    );
  }

  // NEW: Gamification section
  Widget _buildGamificationSection(BuildContext context, bool isArabic) {
    return Consumer<SignDetectionProvider>(
      builder: (context, provider, child) {
        final progress = provider.getUserProgress();
        final userRank = provider.getUserRank();

        return Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(
                      Icons.emoji_events,
                      color: Colors.amber,
                    ),
                    const SizedBox(width: 12),
                    Text(
                      isArabic ? 'التقدم والنجوم' : 'Progress & Achievements',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 12),

                // Level and points
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    Column(
                      children: [
                        Text(
                          isArabic ? 'المستوى' : 'Level',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                        Text(
                          '${progress['level']}',
                          style: TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: Theme.of(context).colorScheme.primary,
                          ),
                        ),
                      ],
                    ),
                    Column(
                      children: [
                        Text(
                          isArabic ? 'الرتبة' : 'Rank',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                        Text(
                          userRank,
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Colors.amber,
                          ),
                        ),
                      ],
                    ),
                    Column(
                      children: [
                        Text(
                          isArabic ? 'النقاط' : 'Points',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                        Text(
                          '${progress['points']}',
                          style: TextStyle(
                            fontSize: 24,
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
                          isArabic ? 'التقدم للمستوى التالي' : 'Progress to next level',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                        Text(
                          '${progress['pointsToNextLevel']} ${isArabic ? 'نقطة' : 'points'}',
                          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: Theme.of(context).colorScheme.primary,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    LinearProgressIndicator(
                      value: progress['progress'],
                      backgroundColor: Colors.grey[300],
                      color: Theme.of(context).colorScheme.primary,
                      minHeight: 8,
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ],
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildSectionHeader(BuildContext context, String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(
        title,
        style: Theme.of(context).textTheme.titleLarge?.copyWith(
          fontWeight: FontWeight.bold,
          color: Theme.of(context).colorScheme.primary,
        ),
      ),
    );
  }

  Widget _buildThemeSelector(
      BuildContext context,
      AppSettingsProvider settings,
      bool isArabic,
      ) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.palette,
                  color: Theme.of(context).colorScheme.primary,
                ),
                const SizedBox(width: 12),
                Text(
                  isArabic ? 'المظهر' : 'Theme',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),

            const SizedBox(height: 12),

            Row(
              children: [
                Expanded(
                  child: _buildThemeOption(
                    context,
                    title: isArabic ? 'فاتح' : 'Light',
                    icon: Icons.light_mode,
                    isSelected: settings.themeMode == ThemeMode.light,
                    onTap: () => settings.setThemeMode(ThemeMode.light),
                  ),
                ),

                const SizedBox(width: 8),

                Expanded(
                  child: _buildThemeOption(
                    context,
                    title: isArabic ? 'داكن' : 'Dark',
                    icon: Icons.dark_mode,
                    isSelected: settings.themeMode == ThemeMode.dark,
                    onTap: () => settings.setThemeMode(ThemeMode.dark),
                  ),
                ),

                const SizedBox(width: 8),

                Expanded(
                  child: _buildThemeOption(
                    context,
                    title: isArabic ? 'تلقائي' : 'Auto',
                    icon: Icons.auto_mode,
                    isSelected: settings.themeMode == ThemeMode.system,
                    onTap: () => settings.setThemeMode(ThemeMode.system),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildThemeOption(
      BuildContext context, {
        required String title,
        required IconData icon,
        required bool isSelected,
        required VoidCallback onTap,
      }) {
    final theme = Theme.of(context);

    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 12),
        decoration: BoxDecoration(
          color: isSelected
              ? theme.colorScheme.primary.withOpacity(0.1)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: isSelected
                ? theme.colorScheme.primary
                : theme.colorScheme.outline.withOpacity(0.3),
          ),
        ),
        child: Column(
          children: [
            Icon(
              icon,
              color: isSelected
                  ? theme.colorScheme.primary
                  : theme.colorScheme.onSurface.withOpacity(0.6),
            ),
            const SizedBox(height: 4),
            Text(
              title,
              style: TextStyle(
                fontSize: 12,
                color: isSelected
                    ? theme.colorScheme.primary
                    : theme.colorScheme.onSurface.withOpacity(0.6),
                fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLanguageSelector(
      BuildContext context,
      AppSettingsProvider settings,
      bool isArabic,
      ) {
    return Card(
      child: ListTile(
        leading: Icon(
          Icons.language,
          color: Theme.of(context).colorScheme.primary,
        ),
        title: Text(isArabic ? 'اللغة' : 'Language'),
        subtitle: Text(isArabic ? 'العربية' : 'English'),
        trailing: Switch(
          value: isArabic,
          onChanged: (value) {
            final newLocale = value
                ? const Locale('ar', 'AE')
                : const Locale('en', 'US');
            settings.setLocale(newLocale);
          },
        ),
      ),
    );
  }

  Widget _buildConfidenceSlider(
      BuildContext context,
      AppSettingsProvider settings,
      bool isArabic,
      ) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.precision_manufacturing,
                  color: Theme.of(context).colorScheme.primary,
                ),
                const SizedBox(width: 12),
                Text(
                  isArabic ? 'حد الثقة' : 'Confidence Threshold',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),

            const SizedBox(height: 8),

            Text(
              isArabic
                  ? 'الحد الأدنى للثقة لقبول الكشف'
                  : 'Minimum confidence to accept detection',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurface.withOpacity(0.7),
              ),
            ),

            const SizedBox(height: 12),

            Row(
              children: [
                Text('10%'),
                Expanded(
                  child: Slider(
                    value: settings.confidenceThreshold,
                    min: 0.1,
                    max: 1.0,
                    divisions: 9,
                    label: '${(settings.confidenceThreshold * 100).toInt()}%',
                    onChanged: settings.setConfidenceThreshold,
                  ),
                ),
                Text('100%'),
              ],
            ),

            Center(
              child: Text(
                '${(settings.confidenceThreshold * 100).toInt()}%',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: Theme.of(context).colorScheme.primary,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildArabicLabelsToggle(
      BuildContext context,
      AppSettingsProvider settings,
      bool isArabic,
      ) {
    return Card(
      child: SwitchListTile(
        secondary: Icon(
          Icons.translate,
          color: Theme.of(context).colorScheme.primary,
        ),
        title: Text(isArabic ? 'عرض النصوص العربية' : 'Show Arabic Labels'),
        subtitle: Text(
          isArabic
              ? 'إظهار النصوص العربية مع الإنجليزية'
              : 'Display Arabic text alongside English',
        ),
        value: settings.showArabicLabels,
        onChanged: (_) => settings.toggleArabicLabels(),
      ),
    );
  }

  Widget _buildSoundToggle(
      BuildContext context,
      AppSettingsProvider settings,
      bool isArabic,
      ) {
    return Card(
      child: SwitchListTile(
        secondary: Icon(
          settings.soundEnabled ? Icons.volume_up : Icons.volume_off,
          color: Theme.of(context).colorScheme.primary,
        ),
        title: Text(isArabic ? 'الصوت' : 'Sound'),
        subtitle: Text(
          isArabic
              ? 'تشغيل الأصوات عند الكشف'
              : 'Play sounds on detection',
        ),
        value: settings.soundEnabled,
        onChanged: (_) => settings.toggleSound(),
      ),
    );
  }

  Widget _buildVibrationToggle(
      BuildContext context,
      AppSettingsProvider settings,
      bool isArabic,
      ) {
    return Card(
      child: SwitchListTile(
        secondary: Icon(
          settings.vibrationEnabled ? Icons.vibration : Icons.phone_android,
          color: Theme.of(context).colorScheme.primary,
        ),
        title: Text(isArabic ? 'الاهتزاز' : 'Vibration'),
        subtitle: Text(
          isArabic
              ? 'اهتزاز الهاتف عند الكشف'
              : 'Vibrate phone on detection',
        ),
        value: settings.vibrationEnabled,
        onChanged: (_) => settings.toggleVibration(),
      ),
    );
  }

  Widget _buildAboutCard(BuildContext context, bool isArabic) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.info,
                  color: Theme.of(context).colorScheme.primary,
                ),
                const SizedBox(width: 12),
                Text(
                  isArabic ? 'معلومات التطبيق' : 'App Information',
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),

            const SizedBox(height: 12),

            _buildInfoRow(
              context,
              label: isArabic ? 'الإصدار' : 'Version',
              value: '1.0.0',
            ),

            _buildInfoRow(
              context,
              label: isArabic ? 'المطور' : 'Developer',
              value: 'MLR511 Team',
            ),

            _buildInfoRow(
              context,
              label: isArabic ? 'الجامعة' : 'University',
              value: isArabic ? 'جامعة الإمارات' : 'UAE University',
            ),

            _buildInfoRow(
              context,
              label: isArabic ? 'المقرر' : 'Course',
              value: 'Mobile Apps & ML',
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(BuildContext context, {required String label, required String value}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              color: Theme.of(context).colorScheme.onSurface.withOpacity(0.7),
            ),
          ),
          Text(
            value,
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  // UPDATED: Combined reset buttons
  Widget _buildResetButtons(BuildContext context, AppSettingsProvider settings, bool isArabic) {
    return Column(
      children: [
        SizedBox(
          width: double.infinity,
          child: OutlinedButton.icon(
              onPressed: () => _showLogoutDialog(context, isArabic),
            icon: Icon(Icons.emoji_events),
            label: Text(isArabic ? 'إعادة تعيين التقدم' : 'Reset Game Progress'),
            style: OutlinedButton.styleFrom(
              foregroundColor: Colors.orange,
              side: BorderSide(color: Colors.orange),
            ),
          ),

        ),
        SizedBox(
          width: double.infinity,
          child: OutlinedButton.icon(
            onPressed: () => _showResetDialog(context, settings, isArabic),
            icon: const Icon(Icons.restore),
            label: Text(isArabic ? 'إعادة تعيين الإعدادات' : 'Reset Settings'),
            style: OutlinedButton.styleFrom(
              foregroundColor: Colors.red,
              side: const BorderSide(color: Colors.red),
            ),
          ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          child: OutlinedButton.icon(
            onPressed: () => _showLogoutDialog(context, isArabic),
            icon: Icon(Icons.logout),
            label: Text(isArabic ? 'تسجيل الخروج' : 'Log Out'),
            style: OutlinedButton.styleFrom(
              foregroundColor: Colors.blueGrey,
              side: BorderSide(color: Colors.blueGrey),
            ),
          ),
        ),

        const SizedBox(height: 12),


      ],
    );
  }

  // NEW: Reset game progress dialog
  void _showResetGameDialog(BuildContext context, bool isArabic) {
    final provider = Provider.of<SignDetectionProvider>(context, listen: false);

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(isArabic ? 'إعادة تعيين التقدم' : 'Reset Game Progress'),
        content: Text(
          isArabic
              ? 'هل أنت متأكد من أنك تريد إعادة تعيين جميع النقاط والمستويات؟'
              : 'Are you sure you want to reset all points and levels?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: Text(isArabic ? 'إلغاء' : 'Cancel'),
          ),
          TextButton(
            onPressed: () {
              provider.resetGameData();
              Navigator.of(context).pop();
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text(
                    isArabic
                        ? 'تم إعادة تعيين التقدم بنجاح'
                        : 'Game progress reset successfully',
                  ),
                ),
              );
            },
            style: TextButton.styleFrom(foregroundColor: Colors.orange),
            child: Text(isArabic ? 'إعادة تعيين' : 'Reset'),
          ),
        ],
      ),
    );
  }

  void _showResetDialog(
      BuildContext context,
      AppSettingsProvider settings,
      bool isArabic,
      ) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(isArabic ? 'إعادة تعيين الإعدادات' : 'Reset Settings'),
        content: Text(
          isArabic
              ? 'هل أنت متأكد من أنك تريد إعادة تعيين جميع الإعدادات إلى القيم الافتراضية؟'
              : 'Are you sure you want to reset all settings to default values?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: Text(isArabic ? 'إلغاء' : 'Cancel'),
          ),
          TextButton(
            onPressed: () {
              settings.resetToDefaults();
              Navigator.of(context).pop();
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text(
                    isArabic
                        ? 'تم إعادة تعيين الإعدادات بنجاح'
                        : 'Settings reset successfully',
                  ),
                ),
              );
            },
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: Text(isArabic ? 'إعادة تعيين' : 'Reset'),
          ),
        ],
      ),
    );
  }

  Future<void> _showLogoutDialog(BuildContext context, bool isArabic) async {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(isArabic ? 'تسجيل الخروج' : 'Log out'),
        content: Text(
          isArabic
              ? 'هل أنت متأكد أنك تريد تسجيل الخروج من الحساب؟'
              : 'Are you sure you want to log out?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: Text(isArabic ? 'إلغاء' : 'Cancel'),
          ),
          TextButton(
            onPressed: () async {
              Navigator.of(context).pop(); // close dialog

              try {
                await FirebaseAuth.instance.signOut();

                // ✅ If you use AuthGate with authStateChanges() as home,
                // you DON'T need navigation, it will go to login automatically.

                // ❕ If you are using named routes and a LoginScreen, you can also do:
                Navigator.of(context).pushNamedAndRemoveUntil(
                   '/login',
                   (route) => false,
                 );
              } catch (e) {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text(
                      isArabic
                          ? 'حدث خطأ أثناء تسجيل الخروج'
                          : 'Error while logging out',
                    ),
                  ),
                );
              }
            },
            child: Text(isArabic ? 'تسجيل الخروج' : 'Log out'),
          ),
        ],
      ),
    );
  }

}