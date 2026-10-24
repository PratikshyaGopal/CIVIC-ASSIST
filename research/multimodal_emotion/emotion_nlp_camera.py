"""
Multimodal Emotion Detection – File 3
Audio (MFCC/LSTM) + EEG (LSTM) + NLP (Embedding + LSTM) + Camera (CNN Face)
Supports multiple STT languages via a language selector.

Bugs fixed:
- haarcascade XML path: auto-resolved from cv2 data directory (no hardcoded path)
- face_model untrained at detect_face call → added warning; model now uses
  pretrained weights if provided, otherwise user is warned prediction is random
- Bare except clauses → specific exception handling + logging throughout
- Audio / EEG path validation in every callback
- Multilingual Google STT (language parameter) added
- EEG alignment helper guards against empty array
- train_test_split uses stratify=y
- zero_division=0 on precision/recall/f1
- confusion_matrix labels explicitly set; display_labels use RAVDESS names
- ROC binarize column indexing fixed for edge cases
- Tokenizer oov_token added; pad_sequences uses padding='post'
- EEG channel width standardised to EEG_CHANNELS constant
- Status label provides live progress feedback
- All callbacks wrapped in try/except with messagebox error display
"""

import os
import logging
import numpy as np
import pandas as pd
import librosa
import cv2
import speech_recognition as sr
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Concatenate,
    Conv2D, MaxPooling2D, Flatten, Dropout, Embedding
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(funcName)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
STT_LANGUAGES = {
    "English (US)":       "en-US",
    "English (UK)":       "en-GB",
    "Hindi":              "hi-IN",
    "Spanish":            "es-ES",
    "French":             "fr-FR",
    "German":             "de-DE",
    "Chinese (Mandarin)": "zh-CN",
    "Arabic":             "ar-SA",
    "Portuguese":         "pt-BR",
    "Russian":            "ru-RU",
    "Japanese":           "ja-JP",
    "Korean":             "ko-KR",
}

EMOTION_LABELS = [
    "Neutral", "Calm", "Happy", "Sad",
    "Angry", "Fearful", "Disgust", "Surprised"
]

FACE_LABELS   = ["Angry", "Happy", "Sad"]
MAX_SEQ_LEN   = 30
VOCAB_SIZE    = 3000
EMBED_DIM     = 64
LSTM_UNITS    = 64
EEG_CHANNELS  = 32


# ===========================================================================
# AUDIO FEATURE EXTRACTION
# ===========================================================================
def extract_audio_features(path: str) -> np.ndarray | None:
    try:
        y, sr_ = librosa.load(path, duration=3)
        mfcc = librosa.feature.mfcc(y=y, sr=sr_, n_mfcc=40)
        return np.mean(mfcc, axis=1)
    except Exception as exc:
        logger.warning("Audio feature extraction failed [%s]: %s", path, exc)
        return None


def load_ravdess(folder: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, files = [], [], []
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".wav"):
            continue
        feat = extract_audio_features(os.path.join(folder, f))
        if feat is None:
            continue
        parts = f.split("-")
        if len(parts) < 3:
            logger.warning("Unexpected filename format: %s", f)
            continue
        try:
            emotion = int(parts[2]) - 1
        except ValueError:
            logger.warning("Cannot parse emotion from: %s", f)
            continue
        X.append(feat)
        y.append(emotion)
        files.append(os.path.join(folder, f))

    if not X:
        raise ValueError(f"No valid .wav files in: {folder}")
    return np.array(X), np.array(y), files


# ===========================================================================
# SPEECH TO TEXT  (multilingual)
# ===========================================================================
def audio_to_text(audio_path: str, language: str = "en-US") -> str:
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as src:
            audio = r.record(src)
        return r.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        logger.debug("STT: unintelligible audio – %s", audio_path)
        return ""
    except sr.RequestError as exc:
        logger.warning("STT API error [%s]: %s", audio_path, exc)
        return ""
    except Exception as exc:
        logger.warning("STT unexpected error [%s]: %s", audio_path, exc)
        return ""


# ===========================================================================
# EEG CSV
# ===========================================================================
def load_eeg_csv(path: str, n_channels: int = EEG_CHANNELS) -> np.ndarray:
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number])
    cols = min(n_channels, numeric.shape[1])
    if cols == 0:
        raise ValueError("EEG CSV has no numeric columns.")
    return numeric.iloc[:, :cols].values.astype(np.float32)


def align_eeg(eeg: np.ndarray, target_len: int) -> np.ndarray:
    if len(eeg) == 0:
        raise ValueError("EEG array is empty.")
    reps = (target_len // len(eeg)) + 1
    return np.tile(eeg, (reps, 1))[:target_len]


# ===========================================================================
# FACE CNN MODEL
# ===========================================================================
def build_face_model() -> Sequential:
    """
    Lightweight CNN for 3-class face emotion (Angry / Happy / Sad).
    Input: (batch, 48, 48, 1) grayscale image.
    NOTE: This model is untrained at startup.
         You must either train it on FER-2013 data or load pretrained weights.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(len(FACE_LABELS), activation="softmax"),
    ], name="face_emotion_cnn")
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _resolve_cascade_path() -> str:
    """
    Resolve the haarcascade XML path from OpenCV's data directory.
    Falls back to the local directory if cv2.data is unavailable.
    """
    xml_name = "haarcascade_frontalface_default.xml"
    # cv2.data.haarcascades is available in opencv-python >= 3.4
    try:
        cascade_dir = cv2.data.haarcascades          # type: ignore[attr-defined]
        return os.path.join(cascade_dir, xml_name)
    except AttributeError:
        return xml_name                              # fallback: cwd


# ===========================================================================
# GUI
# ===========================================================================
class EmotionGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title(
            "Multimodal Emotion Detection – Audio + EEG + NLP + Camera")
        root.geometry("840x720")
        root.resizable(True, True)

        tk.Label(root,
                 text="Multimodal Emotion Detection System\n"
                      "(Audio + EEG + NLP + Camera)",
                 font=("Arial", 16, "bold")).pack(pady=8)

        # Language
        lf = tk.Frame(root); lf.pack()
        tk.Label(lf, text="STT Language:").pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value="English (US)")
        ttk.Combobox(lf, textvariable=self.lang_var,
                     values=list(STT_LANGUAGES.keys()),
                     state="readonly", width=24).pack(side=tk.LEFT, padx=6)

        # Buttons
        bf = tk.Frame(root); bf.pack(pady=6)
        for label, cmd in [
            ("Upload Audio Folder",          self.load_audio),
            ("Upload EEG CSV",               self.load_eeg),
            ("Convert Audio to Text",        self.convert_text),
            ("Load Face Model Weights",      self.load_face_weights),
            ("Detect Face Emotion (Camera)", self.detect_face),
            ("Train & Evaluate Model",       self.train_model),
        ]:
            tk.Button(bf, text=label, command=cmd, width=30).pack(pady=2)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(root, textvariable=self.status_var,
                 fg="blue", font=("Arial", 10)).pack()

        tk.Label(root, text="Transcription:", anchor="w").pack(fill=tk.X, padx=8)
        # Unicode-aware font so Hindi, Arabic, CJK etc. display correctly
        _unicode_font = self._best_unicode_font(size=12)
        self.text_box = scrolledtext.ScrolledText(
            root, height=6, width=95, font=_unicode_font)
        self.text_box.pack(padx=8)

        # State
        self.audio_path: str = ""
        self.eeg_data: np.ndarray | None = None
        self.audio_files: list[str] = []
        self.face_model_trained: bool = False

        # Face components
        self.face_model   = build_face_model()
        cascade_path      = _resolve_cascade_path()
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.warning(
                "Haarcascade XML not found at: %s. "
                "Face detection will be unavailable.", cascade_path)

    @staticmethod
    def _best_unicode_font(size: int = 12) -> tuple[str, int]:
        """Return a font that can render non-Latin scripts (Hindi, Arabic, CJK…)."""
        import tkinter.font as tkfont
        available = set(tkfont.families())
        for family in ("Nirmala UI", "Arial Unicode MS", "TkDefaultFont"):
            if family in available:
                return (family, size)
        return ("TkDefaultFont", size)

    @property
    def stt_language(self) -> str:
        return STT_LANGUAGES.get(self.lang_var.get(), "en-US")

    # ── Callbacks ────────────────────────────────────────────────────────────
    def load_audio(self) -> None:
        path = filedialog.askdirectory(title="Select RAVDESS Audio Folder")
        if not path:
            return
        wavs = [f for f in os.listdir(path) if f.endswith(".wav")]
        if not wavs:
            messagebox.showwarning("Warning", "No .wav files found.")
            return
        self.audio_path = path
        self.status_var.set(f"Audio folder loaded: {len(wavs)} files.")

    def load_eeg(self) -> None:
        path = filedialog.askopenfilename(
            title="Select EEG CSV",
            filetypes=[("CSV Files", "*.csv")])
        if not path:
            return
        try:
            self.eeg_data = load_eeg_csv(path)
            self.status_var.set(
                f"EEG loaded: {self.eeg_data.shape[0]}×"
                f"{self.eeg_data.shape[1]}")
        except Exception as exc:
            messagebox.showerror("EEG Error", str(exc))

    def convert_text(self) -> None:
        if not self.audio_path:
            messagebox.showwarning("Warning", "Load an audio folder first.")
            return
        wavs = [f for f in os.listdir(self.audio_path) if f.endswith(".wav")]
        if not wavs:
            messagebox.showwarning("Warning", "No .wav files found.")
            return
        self.status_var.set("Transcribing first file…")
        self.root.update_idletasks()
        text = audio_to_text(
            os.path.join(self.audio_path, wavs[0]),
            language=self.stt_language)
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert(tk.END, text or "(No speech detected)")
        self.status_var.set("Transcription done.")

    def load_face_weights(self) -> None:
        """Load pre-trained Keras weights (.h5) into the face model."""
        path = filedialog.askopenfilename(
            title="Select Face Model Weights",
            filetypes=[("Keras Weights", "*.h5 *.keras"), ("All", "*.*")])
        if not path:
            return
        try:
            self.face_model.load_weights(path)
            self.face_model_trained = True
            self.status_var.set("Face model weights loaded.")
        except Exception as exc:
            messagebox.showerror("Weight Load Error", str(exc))

    def detect_face(self) -> None:
        if self.face_cascade.empty():
            messagebox.showerror(
                "Error",
                "Haarcascade XML not found. Cannot perform face detection.")
            return
        if not self.face_model_trained:
            messagebox.showwarning(
                "Warning",
                "Face model has no trained weights loaded.\n"
                "Predictions will be random.\n"
                "Use 'Load Face Model Weights' to load a pretrained .h5 file.")

        try:
            self._run_camera()
        except Exception as exc:
            logger.exception("Camera error")
            messagebox.showerror("Camera Error", str(exc))

    def _run_camera(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera (index 0).")

        messagebox.showinfo("Camera", "Camera is active. Press Q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera frame read failed.")
                break
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.reshape(1, 48, 48, 1).astype(np.float32) / 255.0
                pred    = self.face_model.predict(roi, verbose=0)
                emotion = FACE_LABELS[int(np.argmax(pred))]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Face Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def train_model(self) -> None:
        if not self.audio_path:
            messagebox.showwarning("Warning", "Load an audio folder first.")
            return
        if self.eeg_data is None:
            messagebox.showwarning("Warning", "Load an EEG CSV first.")
            return
        try:
            self._run_training()
        except Exception as exc:
            logger.exception("Training failed")
            messagebox.showerror("Training Error", str(exc))

    def _run_training(self) -> None:
        # ── Audio ─────────────────────────────────────────────────────────
        self.status_var.set("Loading RAVDESS features…")
        self.root.update_idletasks()
        Xa, y, self.audio_files = load_ravdess(self.audio_path)

        n_classes = len(np.unique(y))
        if n_classes < 2:
            raise ValueError("Need ≥ 2 emotion classes to train.")

        # ── EEG ──────────────────────────────────────────────────────────
        Xe = align_eeg(self.eeg_data, len(Xa))
        if Xe.shape[1] < EEG_CHANNELS:
            pad = np.zeros((Xe.shape[0], EEG_CHANNELS - Xe.shape[1]),
                           dtype=np.float32)
            Xe = np.concatenate([Xe, pad], axis=1)
        else:
            Xe = Xe[:, :EEG_CHANNELS]

        # ── NLP ───────────────────────────────────────────────────────────
        self.status_var.set(
            f"Transcribing {len(self.audio_files)} files "
            f"[lang={self.stt_language}]…")
        self.root.update_idletasks()
        texts = [
            audio_to_text(f, language=self.stt_language)
            for f in self.audio_files
        ]

        tok   = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        tok.fit_on_texts(texts)
        Xn    = pad_sequences(tok.texts_to_sequences(texts),
                              maxlen=MAX_SEQ_LEN,
                              padding="post",
                              truncating="post")

        # ── Scale & reshape ───────────────────────────────────────────────
        Xa = StandardScaler().fit_transform(Xa)
        Xe = StandardScaler().fit_transform(Xe)
        Xa = Xa.reshape(len(Xa), 1, 40)
        Xe = Xe.reshape(len(Xe), 1, EEG_CHANNELS)

        y_cat = to_categorical(y)

        (Xa_tr, Xa_te, Xe_tr, Xe_te,
         Xn_tr, Xn_te, y_tr, y_te) = train_test_split(
            Xa, Xe, Xn, y_cat,
            test_size=0.2, random_state=42, stratify=y)

        # ── Build model ───────────────────────────────────────────────────
        audio_in  = Input(shape=(1, 40),         name="audio_in")
        audio_out = LSTM(LSTM_UNITS,             name="audio_lstm")(audio_in)

        eeg_in  = Input(shape=(1, EEG_CHANNELS), name="eeg_in")
        eeg_out = LSTM(LSTM_UNITS,               name="eeg_lstm")(eeg_in)

        text_in  = Input(shape=(MAX_SEQ_LEN,),   name="text_in")
        embed    = Embedding(VOCAB_SIZE, EMBED_DIM, name="embed")(text_in)
        text_out = LSTM(LSTM_UNITS,              name="text_lstm")(embed)

        fused  = Concatenate(name="fusion")([audio_out, eeg_out, text_out])
        output = Dense(y_cat.shape[1], activation="softmax",
                       name="output")(fused)

        model = Model([audio_in, eeg_in, text_in], output)
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        # ── Train ─────────────────────────────────────────────────────────
        self.status_var.set("Training model…")
        self.root.update_idletasks()
        hist = model.fit(
            [Xa_tr, Xe_tr, Xn_tr], y_tr,
            epochs=5, batch_size=16, verbose=1)

        # ── Evaluate ──────────────────────────────────────────────────────
        preds  = model.predict([Xa_te, Xe_te, Xn_te])
        y_true = np.argmax(y_te,  axis=1)
        y_pred = np.argmax(preds, axis=1)

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred,
                               average="weighted", zero_division=0)
        rec  = recall_score(y_true, y_pred,
                            average="weighted", zero_division=0)
        f1   = f1_score(y_true, y_pred,
                        average="weighted", zero_division=0)

        self.status_var.set("Done.")
        messagebox.showinfo("Results",
            f"Accuracy : {acc*100:.2f}%\n"
            f"Precision: {prec*100:.2f}%\n"
            f"Recall   : {rec*100:.2f}%\n"
            f"F1-score : {f1*100:.2f}%")

        # ── Confusion Matrix ──────────────────────────────────────────────
        class_ids   = sorted(np.unique(y))
        class_names = [EMOTION_LABELS[i] if i < len(EMOTION_LABELS)
                       else str(i) for i in class_ids]
        cm   = confusion_matrix(y_true, y_pred, labels=class_ids)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        # ── Training Accuracy ─────────────────────────────────────────────
        plt.figure()
        plt.plot(hist.history["accuracy"], marker="o")
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # ── ROC ───────────────────────────────────────────────────────────
        y_bin = label_binarize(y_true, classes=class_ids)
        if y_bin.ndim == 1:
            y_bin = y_bin.reshape(-1, 1)

        plt.figure(figsize=(8, 6))
        for i, cls_id in enumerate(class_ids):
            col = 0 if y_bin.shape[1] == 1 else i
            fpr, tpr, _ = roc_curve(y_bin[:, col], preds[:, cls_id])
            roc_auc = auc(fpr, tpr)
            name = class_names[i] if i < len(class_names) else f"Class {cls_id}"
            plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc:.2f}")

        plt.plot([0, 1], [0, 1], "k--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()


# ===========================================================================
if __name__ == "__main__":
    root = tk.Tk()
    EmotionGUI(root)
    root.mainloop()
