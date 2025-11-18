# Sayna - Arabic Sign Language Recognition App
**Project for MLR 511: Mobile Apps & ML**
**By:** Meriem Aoudia, Salma Saad, & Hafsa Al-Zubaizi

## 1. Project Overview
**Sayna** is a Flutter application designed to recognize **10 common Arabic sign language (ArSL) gestures** from a live camera feed or a pre-recorded video. This project combines a high-performance deep learning model with a cross-platform mobile app, featuring real-time analysis, user-friendly controls, and gamification to encourage learning.

The application uses an **InceptionV3-based Time-Distributed CNN** model, quantized to **INT8**, to achieve efficient on-device inference.

### Key Features
* **Real-time Detection:** Analyzes video from the device camera (max 3 seconds).
* **Video Upload:** Allows users to pick a video from their gallery for analysis.
* **Gamification:** Users earn points, levels, and streaks for correct detections, viewable on a dedicated history/stats page.
* **History & Stats:** A persistent history screen tracks all past detections, average confidence, and most-detected signs.
* **Bilingual UI:** Supports both English and Arabic.

## 2. Tech Stack
* **Mobile Framework:** Flutter
* **Machine Learning:** TensorFlow Lite
* **ML Model:** `sayna_full_model_INT8.tflite` (InceptionV3 + T-CNN Pooling)
* **Preprocessing:** `ffmpeg_kit_flutter_new` (for video-to-frame conversion) and `image` (for SAD calculation).
* **State Management:** Provider
* **Backend:** Firebase (for authentication, though model is local)

## 3. Environment & Setup
To run this project, your local development environment must match the following versions:

| Software | Required Version |
| :--- | :--- |
| **Flutter** | 3.35.2 |
| **Dart** | 3.9.0 |
| **Android Studio** | 2025.1.2 |
| **Java** | 22.0.2 |

## 4. How to Run This Project
Follow these steps exactly to get the project running.

### Step 1: Clone the Repository
```bash
git clone [https://github.com/your-username/sayna-project.git](https://github.com/your-username/sayna-project.git)
cd sayna-project
```

### Step 2: Add the TFLite Model
This repository does not include the trained model file. You must add it manually.

1. Create a folder named `assets` in the root of the project.
2. Place your trained model file inside it and name it exactly: `assets/sayna_full_model_INT8.tflite`

### Step 3: Get Flutter Dependencies
```bash
flutter pub get
```

### Step 4: Clean and Run
You must run flutter clean after changing dependencies or configuration.

```bash
flutter clean
flutter run
```
The app should now build and launch on your connected Android device or emulator.

## 5. ML Model Details
The model architecture and training process are detailed in the `python/` folder.

* **Architecture:** **InceptionV3** (as a feature extractor) + **Time-Distributed Global Pooling** (for temporal sequence analysis).
* **Preprocessing (SAD):** The app does not feed raw video frames to the model. Instead, it:
    1. Extracts frames at ~15 FPS.
    2. Divides the video into 3 overlapping temporal segments.
    3. Calculates the Sum of Absolute Differences (SAD) for each segment, creating a single image that represents motion.
    4. These 3 SAD images are stacked and fed to the model.

## 6. Project File Structure
```text
.
├── android/                  # Native Android files (build.gradle.kts is here)
├── assets/
│   └── sayna_full_model_INT8.tflite  # (You must add this file)
├── lib/
│   ├── main.dart             # App entry point, providers, and routes
│   ├── models/               # Dart data models (sign_model.dart)
│   ├── providers/            # State management (app_settings, sign_detection)
│   ├── screens/              # UI for each page (splash, home, camera, history)
│   ├── services/             # Core logic (ml_service.dart)
│   └── widgets/              # Reusable UI components (video_preview_dialog)
├── python/                   # Python scripts for training and quantization
│   └── full_training.py
└── pubspec.yaml              # Project dependencies
```
