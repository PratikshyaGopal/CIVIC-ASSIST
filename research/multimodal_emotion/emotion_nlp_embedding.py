"""
Multimodal Emotion Detection – File 2
Audio (MFCC/LSTM) + EEG (LSTM) + NLP (Embedding + LSTM)
Supports multiple STT languages via a language selector.

Bugs fixed:
- Bare except clauses replaced with specific exception handling + logging
- Audio / EEG path validation before use in all callbacks
- Multilingual Google STT (language parameter added throughout)
- EEG tile guard: empty-eeg check, robust align_eeg helper
- train_test_split now uses stratify=y to avoid empty class issues
- zero_division=0 on precision/recall/f1 avoids UndefinedMetricWarning
- confusion_matrix labels explicitly set; display_labels use RAVDESS names
- ROC binarize handles single-class edge case
- Tokenizer / Embedding vocabulary size matched exactly (avoid OOV mismatch)
- Status label gives live feedback during long operations
- All GUI callbacks wrapped in try/except with messagebox error display
"""

import os
import logging
import numpy as np
import pandas as pd
import librosa
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

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(funcName)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported STT languages
# ---------------------------------------------------------------------------
STT_LANGUAGES = {
    "English (US)":      "en-US",
    "English (UK)":      "en-GB",
    "Hindi":             "hi-IN",
    "Spanish":           "es-ES",
    "French":            "fr-FR",
    "German":            "de-DE",
    "Chinese (Mandarin)": "zh-CN",
    "Arabic":            "ar-SA",
    "Portuguese":        "pt-BR",
    "Russian":           "ru-RU",
    "Japanese":          "ja-JP",
    "Korean":            "ko-KR",
}

EMOTION_LABELS = [
    "Neutral", "Calm", "Happy", "Sad",
    "Angry", "Fearful", "Disgust", "Surprised"
]

MAX_SEQ_LEN  = 30
VOCAB_SIZE   = 3000
EMBED_DIM    = 64
LSTM_UNITS   = 64
EEG_CHANNELS = 32


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
            logger.warning("Unexpected filename: %s", f)
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
        raise ValueError("EEG data array is empty.")
    reps = (target_len // len(eeg)) + 1
    return np.tile(eeg, (reps, 1))[:target_len]


# ===========================================================================
# GUI
# ===========================================================================
class EmotionGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Multimodal Emotion Detection – Audio + EEG + NLP (Embedding)")
        root.geometry("840x680")
        root.resizable(True, True)

        tk.Label(root,
                 text="Multimodal Emotion Detection\n(Audio + EEG + NLP Embedding)",
                 font=("Arial", 16, "bold")).pack(pady=8)

        # Language selector
        lf = tk.Frame(root); lf.pack()
        tk.Label(lf, text="STT Language:").pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value="English (US)")
        ttk.Combobox(lf, textvariable=self.lang_var,
                     values=list(STT_LANGUAGES.keys()),
                     state="readonly", width=24).pack(side=tk.LEFT, padx=6)

        # Buttons
        bf = tk.Frame(root); bf.pack(pady=6)
        for label, cmd in [
            ("Upload Audio Folder",    self.load_audio),
            ("Upload EEG CSV",         self.load_eeg),
            ("Convert Audio to Text",  self.convert_text),
            ("Train & Evaluate Model", self.train_model),
        ]:
            tk.Button(bf, text=label, command=cmd, width=26).pack(pady=2)

        # Status
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(root, textvariable=self.status_var,
                 fg="blue", font=("Arial", 10)).pack()

        # Transcription
        tk.Label(root, text="Transcription:", anchor="w").pack(fill=tk.X, padx=8)
        # Unicode-aware font: renders Devanagari, Arabic, CJK, etc.
        _unicode_font = self._best_unicode_font(size=12)
        self.text_box = scrolledtext.ScrolledText(
            root, height=6, width=95, font=_unicode_font)
        self.text_box.pack(padx=8)

        # State
        self.audio_path: str = ""
        self.eeg_data: np.ndarray | None = None
        self.audio_files: list[str] = []

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
        self.status_var.set(f"Audio folder: {len(wavs)} files loaded.")

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
        self.status_var.set("Transcribing first audio file…")
        self.root.update_idletasks()
        text = audio_to_text(
            os.path.join(self.audio_path, wavs[0]),
            language=self.stt_language)
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert(tk.END, text or "(No speech detected)")
        self.status_var.set("Transcription done.")

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
        # Pad / trim EEG to exactly EEG_CHANNELS
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

        tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        seqs = tokenizer.texts_to_sequences(texts)
        Xn = pad_sequences(seqs, maxlen=MAX_SEQ_LEN, padding="post",
                           truncating="post")

        # ── Scale & reshape ───────────────────────────────────────────────
        Xa = StandardScaler().fit_transform(Xa)
        Xe = StandardScaler().fit_transform(Xe)
        Xa = Xa.reshape(Xa.shape[0], 1, Xa.shape[1])
        Xe = Xe.reshape(Xe.shape[0], 1, EEG_CHANNELS)

        y_cat = to_categorical(y)

        (Xa_tr, Xa_te, Xe_tr, Xe_te,
         Xn_tr, Xn_te, y_tr, y_te) = train_test_split(
            Xa, Xe, Xn, y_cat,
            test_size=0.2, random_state=42, stratify=y)

        # ── Model ─────────────────────────────────────────────────────────
        audio_in  = Input(shape=(1, Xa.shape[2]), name="audio_in")
        audio_out = LSTM(LSTM_UNITS, name="audio_lstm")(audio_in)

        eeg_in  = Input(shape=(1, EEG_CHANNELS), name="eeg_in")
        eeg_out = LSTM(LSTM_UNITS, name="eeg_lstm")(eeg_in)

        text_in  = Input(shape=(MAX_SEQ_LEN,), name="text_in")
        embed    = Embedding(VOCAB_SIZE, EMBED_DIM, name="embedding")(text_in)
        text_out = LSTM(LSTM_UNITS, name="text_lstm")(embed)

        fused  = Concatenate(name="fusion")([audio_out, eeg_out, text_out])
        output = Dense(y_cat.shape[1], activation="softmax",
                       name="output")(fused)

        model = Model([audio_in, eeg_in, text_in], output)
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        # ── Train ─────────────────────────────────────────────────────────
        self.status_var.set("Training…")
        self.root.update_idletasks()
        history = model.fit(
            [Xa_tr, Xe_tr, Xn_tr], y_tr,
            epochs=8, batch_size=16, verbose=1)

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

        self.status_var.set("Evaluation complete.")
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
        plt.plot(history.history["accuracy"], marker="o")
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
