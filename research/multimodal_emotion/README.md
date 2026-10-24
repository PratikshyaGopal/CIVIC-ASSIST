# Multimodal Emotion Detection Research Prototypes

This directory contains standalone research prototypes for multimodal emotion detection combining:
- **Audio processing** (MFCC extraction via `librosa`)
- **Speech-to-text** (multilingual Google STT via `SpeechRecognition`)
- **EEG signal processing** (LSTM modeling)
- **NLP** (TF-IDF vectorization and word embeddings + LSTM)
- **Computer Vision** (OpenCV Haar cascades and Face CNN)

---

## 📁 Files

1. **`emotion_nlp_tfidf.py`**: Audio (MFCC) + EEG + NLP (TF-IDF vectorizer + LSTM) with Tkinter GUI.
2. **`emotion_nlp_embedding.py`**: Audio (MFCC/LSTM) + EEG (LSTM) + NLP (Embedding + LSTM) with Tkinter GUI.
3. **`emotion_nlp_camera.py`**: Audio + EEG + NLP + Live Camera (CNN Face detection) with Tkinter GUI.

---

## ⚙️ Requirements & Installation

These research scripts require specialized machine learning and signal processing packages:

```bash
pip install -r requirements-ml.txt
```

### Dependencies
- Python 3.9 - 3.11
- `tensorflow`
- `scikit-learn`
- `librosa`
- `opencv-python`
- `speechrecognition`
- `pandas`
- `numpy`
- `matplotlib`

---

## 🚀 Running the Prototypes

To launch any of the desktop interfaces, run:

```bash
python emotion_nlp_tfidf.py
```
or
```bash
python emotion_nlp_camera.py
```
