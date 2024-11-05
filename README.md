# WBG: Words by Gestures - Emergency Line for Hearing Impaired Community

This project presents **WBG (Words by Gestures)**, a novel deep-learning-based system designed to facilitate communication between the deaf community and emergency responders. WBG translates between Portuguese Sign Language (LGP) and Portuguese Language (LP) in both directions, enabling emergency lines to assist hearing-impaired individuals without needing a physical interpreter.

## Table of Contents
- [Abstract](#abstract)
- [Project Motivation](#project-motivation)
- [Methodology](#methodology)
  - [LGP to LP Translation](#lgp-to-lp-translation)
  - [LP to LGP Translation](#lp-to-lgp-translation)
  - [Intonation Classification](#intonation-classification)
- [Usage](#usage)
  - [Folder Structure](#folder-structure)
  - [Running the System Test](#running-the-system-test)
- [Results](#results)
- [Future Work](#future-work)
- [Demo](#demo)
- [Report](#report)

## Abstract

WBG provides a two-way translation system:
1. **LGP to LP**: Processes video of sign language gestures using a Bi-LSTM model that outputs corresponding text in LP.
2. **LP to LGP**: Converts spoken LP to LGP, with a CNN model classifying intonation and a text-to-speech API producing sign language gestures.

## Project Motivation

With more than 1.5 billion individuals globally experiencing hearing loss, emergency responders face significant communication challenges. Current solutions are limited by the need for physical interpreters, which WBG addresses by using AI-based translation.

## Methodology

The system architecture comprises two main pipelines:

### LGP to LP Translation
1. **3D Keypoint Detection**: Extracts body, hand, and facial keypoints using MediaPipe Holistic.
2. **Sequence Models**: Classifies gesture sequences into text using models like Bi-LSTM, LSTM, RNN, and GRU.
3. **Text-to-Speech**: Uses gTTS API to convert LP text to spoken language for emergency responders.

### LP to LGP Translation
1. **Speech to Text**: Speech is transcribed using the SpeechRecognition API.
2. **Intonation Classification**: Uses MFCCs to classify intonation as interrogative, declarative, or exclamative, processed with a CNN.
3. **Gesture Generation**: Maps transcribed sentences to pre-recorded skeleton gestures from a database.

### Intonation Classification

Intonation is classified based on MFCC features, which distinguish between different sentence types. This is essential for correct gesture interpretation as some sentences vary by punctuation in sign language.

## Usage

### Folder Structure

- **all_models/**: Contains pre-trained models used in translation processes.
  - **audio_model/**: Audio model for classifying intonation in LP sentences.
  - **bi_lstm_model/**: Bi-LSTM model for gesture recognition.
  - **lstm_model/**, **rnn_model/**, **gru_model/**: Alternative models for recognizing gestures in LGP.

- **LGP_to_LP/**: Contains code and resources for translating from LGP to LP.
  - **Keypoints_extractors/**: Scripts for extracting 3D keypoints from video, which are essential for recognizing LGP gestures.
  - **real_time_LGP/**: Scripts and setup for real-time translation of LGP to LP.
  - **training/**: Training resources, including model scripts for LGP-to-LP translation.

- **LP_to_LGP/**: Contains code and resources for translating from LP to LGP.
  - **real_time_LP/**: Implements real-time LP-to-LGP translation, using intonation classification.
  - **train_audio/**: Contains scripts and resources for training the audio model that classifies intonation in LP.

### Running the System Test

To test the entire system, use the `test.py` script if provided. This will help verify that all components are functioning as expected and that translation occurs seamlessly between LGP and LP.

## Results

- **LGP to LP**: Achieved 99% accuracy in real-time translation of LGP gestures to Portuguese text.
- **LP to LGP**: Achieved 87% accuracy in intonation classification, enabling accurate gesture selection.

## Future Work

- **Expand the Dataset**: Add recordings from LGP speakers for greater gesture accuracy.
- **Optimize for Mobile Devices**: Enable processing on smartphones to improve accessibility.
- **Enhanced Gesture Context**: Refine the model to better capture sentence context in LGP.

## Demo

The project demonstration video is available in the `demo/` folder. It provides a real-world application of the WBG system, showcasing translation from LGP to LP and vice versa.

## Report

For detailed insights into the project, refer to the `report/` folder, which contains the full project report.

## References

1. [MediaPipe Holistic](https://ai.googleblog.com/2020/12/mediapipe-holistic-simultaneous-face.html)
2. [gTTS Documentation](https://gtts.readthedocs.io/en/latest/)
3. [SpeechRecognition API](https://pypi.org/project/SpeechRecognition/)
4. [Project Paper](link-to-paper-if-published)

---

This project provides a complete, real-time communication tool for hearing-impaired individuals, bridging the gap between sign language and spoken language in emergency settings.

