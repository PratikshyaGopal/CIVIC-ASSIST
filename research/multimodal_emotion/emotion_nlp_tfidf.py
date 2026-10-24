"""
Multimodal Emotion Detection – File 1
Audio (MFCC) + EEG + NLP (TF-IDF)
Supports multiple STT languages via a language selector.

Bugs fixed:
- Bare except clauses replaced with specific exception handling
- Audio folder / EEG path validation before use
- Multilingual Google STT (language parameter)
- EEG row-count guard to prevent zero-size tile
- train_test_split may yield empty test set without stratify; added stratify
- label_binarize applied to y_true (was correct) but classes= now dynamically set
- All messagebox errors shown instead of silent failures
- confusion_matrix classes set explicitly to avoid display mismatch
- zero_division parameter added to precision/recall/f1 to suppress warnings
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
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.utils import to_categorical

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(funcName)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported STT languages (Google Speech Recognition language codes)
# ---------------------------------------------------------------------------
STT_LANGUAGES = {
    "English (US)": "en-US",
    "English (UK)": "en-GB",
    "Hindi": "hi-IN",
    "Spanish": "es-ES",
    "French": "fr-FR",
    "German": "de-DE",
    "Chinese (Mandarin)": "zh-CN",
    "Arabic": "ar-SA",
    "Portuguese": "pt-BR",
    "Russian": "ru-RU",
    "Japanese": "ja-JP",
    "Korean": "ko-KR",
}

# RAVDESS emotion labels (1-indexed in filename, 0-indexed here)
EMOTION_LABELS = [
    "Neutral", "Calm", "Happy", "Sad",
    "Angry", "Fearful", "Disgust", "Surprised"
]


# ===========================================================================
# AUDIO FEATURE EXTRACTION
# ===========================================================================
def extract_audio_features(file_path: str) -> np.ndarray | None:
    """Return 40-dim MFCC mean vector or None on failure."""
    try:
        y, sr_ = librosa.load(file_path, duration=3)
        mfcc = librosa.feature.mfcc(y=y, sr=sr_, n_mfcc=40)
        return np.mean(mfcc, axis=1)
    except Exception as exc:
        logger.warning("Audio feature extraction failed for %s: %s", file_path, exc)
        return None


def load_ravdess(folder: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load RAVDESS .wav files from *folder*.
    Filename convention: XX-XX-<emotion>-XX-XX-XX-XX.wav  (emotion 1-8)
    Returns (features, labels, file_paths).
    """
    X, y, files = [], [], []
    for file in sorted(os.listdir(folder)):
        if not file.endswith(".wav"):
            continue
        feat = extract_audio_features(os.path.join(folder, file))
        if feat is None:
            continue
        parts = file.split("-")
        if len(parts) < 3:
            logger.warning("Unexpected filename format: %s", file)
            continue
        try:
            emotion = int(parts[2]) - 1          # 0-indexed
        except ValueError:
            logger.warning("Cannot parse emotion from filename: %s", file)
            continue
        X.append(feat)
        y.append(emotion)
        files.append(os.path.join(folder, file))

    if not X:
        raise ValueError(f"No valid .wav files found in: {folder}")
    return np.array(X), np.array(y), files


# ===========================================================================
# SPEECH → TEXT  (multilingual)
# ===========================================================================
def audio_to_text(audio_path: str, language: str = "en-US") -> str:
    """
    Convert an audio file to text using Google Web Speech API.
    *language* is a BCP-47 code, e.g. 'hi-IN', 'fr-FR'.
    Returns an empty string on failure.
    """
    recog = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recog.record(source)
        return recog.recognize_google(audio, language=language)
    except sr.UnknownValueError:
        logger.debug("STT: could not understand audio – %s", audio_path)
        return ""
    except sr.RequestError as exc:
        logger.warning("STT API error for %s: %s", audio_path, exc)
        return ""
    except Exception as exc:
        logger.warning("STT unexpected error for %s: %s", audio_path, exc)
        return ""


# ===========================================================================
# EEG CSV
# ===========================================================================
def load_eeg_csv(path: str, n_channels: int = 32) -> np.ndarray:
    """
    Load numeric EEG data from a CSV file.
    Uses up to *n_channels* columns.
    """
    df = pd.read_csv(path)
    numeric = df.select_dtypes(include=[np.number])
    cols = min(n_channels, numeric.shape[1])
    if cols == 0:
        raise ValueError("EEG CSV contains no numeric columns.")
    return numeric.iloc[:, :cols].values.astype(np.float32)


def align_eeg(eeg_data: np.ndarray, target_len: int) -> np.ndarray:
    """Tile EEG rows to match *target_len* samples."""
    if len(eeg_data) == 0:
        raise ValueError("EEG data is empty.")
    repeats = (target_len // len(eeg_data)) + 1
    return np.tile(eeg_data, (repeats, 1))[:target_len]


# ===========================================================================
# GUI
# ===========================================================================
class EmotionGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Multimodal Emotion Detection – Audio + EEG + NLP (TF-IDF)")
        root.geometry("820x680")
        root.resizable(True, True)

        # ── Header ──────────────────────────────────────────────────────────
        tk.Label(root,
                 text="Multimodal Emotion Detection\n(Audio + EEG + NLP)",
                 font=("Arial", 16, "bold")).pack(pady=8)

        # ── Language selector ────────────────────────────────────────────────
        lang_frame = tk.Frame(root)
        lang_frame.pack()
        tk.Label(lang_frame, text="STT Language:").pack(side=tk.LEFT)
        self.lang_var = tk.StringVar(value="English (US)")
        lang_menu = ttk.Combobox(lang_frame, textvariable=self.lang_var,
                                 values=list(STT_LANGUAGES.keys()),
                                 state="readonly", width=22)
        lang_menu.pack(side=tk.LEFT, padx=6)

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=6)
        for (label, cmd) in [
            ("Upload Audio Folder",     self.load_audio),
            ("Upload EEG CSV",          self.load_eeg),
            ("Convert Audio to Text",   self.convert_text),
            ("Train & Evaluate Model",  self.train_model),
        ]:
            tk.Button(btn_frame, text=label, command=cmd,
                      width=25).pack(pady=2)

        # ── Status label ─────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(root, textvariable=self.status_var,
                 fg="blue", font=("Arial", 10)).pack()

        # ── Transcription box ─────────────────────────────────────────────────
        tk.Label(root, text="Transcription:", anchor="w").pack(fill=tk.X, padx=8)
        # Use a Unicode-aware font so non-Latin scripts (Hindi, Arabic,
        # Chinese, etc.) render correctly instead of showing boxes/English.
        _unicode_font = self._best_unicode_font(size=12)
        self.text_box = scrolledtext.ScrolledText(
            root, width=95, height=7, font=_unicode_font)
        self.text_box.pack(padx=8)

        # ── State ────────────────────────────────────────────────────────────
        self.audio_path: str = ""
        self.eeg_data: np.ndarray | None = None
        self.audio_files: list[str] = []

    # ── Unicode font helper ──────────────────────────────────────────────────
    @staticmethod
    def _best_unicode_font(size: int = 12) -> tuple[str, int]:
        """
        Return a (family, size) font tuple that can render non-Latin scripts
        such as Devanagari (Hindi), Arabic, CJK, Cyrillic, etc.
        Preference order:
          1. Nirmala UI   – built into Windows 8+, excellent Devanagari support
          2. Arial Unicode MS – broad Unicode coverage
          3. TkDefaultFont    – safe fallback (may not render all scripts)
        """
        import tkinter.font as tkfont
        available = set(tkfont.families())
        for family in ("Nirmala UI", "Arial Unicode MS", "TkDefaultFont"):
            if family in available:
                return (family, size)
        return ("TkDefaultFont", size)

    # ── Properties ──────────────────────────────────────────────────────────
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
            messagebox.showwarning("Warning", "No .wav files found in selected folder.")
            return
        self.audio_path = path
        self.status_var.set(f"Audio folder loaded: {len(wavs)} files")

    def load_eeg(self) -> None:
        path = filedialog.askopenfilename(
            title="Select EEG CSV",
            filetypes=[("CSV Files", "*.csv")])
        if not path:
            return
        try:
            self.eeg_data = load_eeg_csv(path)
            self.status_var.set(
                f"EEG loaded: {self.eeg_data.shape[0]} rows × "
                f"{self.eeg_data.shape[1]} channels")
        except Exception as exc:
            messagebox.showerror("EEG Load Error", str(exc))

    def convert_text(self) -> None:
        if not self.audio_path:
            messagebox.showwarning("Warning", "Please load an audio folder first.")
            return
        wavs = [f for f in os.listdir(self.audio_path) if f.endswith(".wav")]
        if not wavs:
            messagebox.showwarning("Warning", "No .wav files found.")
            return
        self.status_var.set("Converting first audio file to text…")
        self.root.update_idletasks()
        text = audio_to_text(
            os.path.join(self.audio_path, wavs[0]),
            language=self.stt_language)
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert(tk.END, text if text else "(No speech detected)")
        self.status_var.set("Conversion complete.")

    def train_model(self) -> None:
        # ── Validation ───────────────────────────────────────────────────────
        if not self.audio_path:
            messagebox.showwarning("Warning", "Please load an audio folder first.")
            return
        if self.eeg_data is None:
            messagebox.showwarning("Warning", "Please load an EEG CSV first.")
            return

        try:
            self._run_training()
        except Exception as exc:
            logger.exception("Training failed")
            messagebox.showerror("Training Error", str(exc))

    def _run_training(self) -> None:
        self.status_var.set("Loading audio features…")
        self.root.update_idletasks()

        # ── Audio ─────────────────────────────────────────────────────────
        Xa, y, self.audio_files = load_ravdess(self.audio_path)
        n_classes = len(np.unique(y))
        if n_classes < 2:
            raise ValueError("Need at least 2 distinct emotion classes to train.")

        # ── EEG ──────────────────────────────────────────────────────────
        Xe = align_eeg(self.eeg_data, len(Xa))

        # ── NLP (TF-IDF) ─────────────────────────────────────────────────
        self.status_var.set(
            f"Transcribing {len(self.audio_files)} files "
            f"[lang={self.stt_language}]…")
        self.root.update_idletasks()
        texts = [
            audio_to_text(f, language=self.stt_language)
            for f in self.audio_files
        ]
        tfidf = TfidfVectorizer(max_features=100)
        Xn = tfidf.fit_transform(texts).toarray()

        # ── Scaling & reshape ─────────────────────────────────────────────
        Xa = StandardScaler().fit_transform(Xa)
        Xe = StandardScaler().fit_transform(Xe)
        Xa = Xa.reshape(Xa.shape[0], 1, Xa.shape[1])
        Xe = Xe.reshape(Xe.shape[0], 1, Xe.shape[1])

        y_cat = to_categorical(y)

        # ── Split (stratified) ────────────────────────────────────────────
        Xa_tr, Xa_te, Xe_tr, Xe_te, Xn_tr, Xn_te, y_tr, y_te = \
            train_test_split(Xa, Xe, Xn, y_cat,
                             test_size=0.2, random_state=42,
                             stratify=y)

        # ── Build model ───────────────────────────────────────────────────
        audio_in  = Input(shape=(1, Xa.shape[2]), name="audio_input")
        audio_out = LSTM(64, name="audio_lstm")(audio_in)

        eeg_in  = Input(shape=(1, Xe.shape[2]), name="eeg_input")
        eeg_out = LSTM(64, name="eeg_lstm")(eeg_in)

        nlp_in  = Input(shape=(Xn.shape[1],), name="nlp_input")
        nlp_out = Dense(64, activation="relu", name="nlp_dense")(nlp_in)

        fusion = Concatenate(name="fusion")([audio_out, eeg_out, nlp_out])
        output = Dense(y_cat.shape[1],
                       activation="softmax", name="output")(fusion)

        model = Model([audio_in, eeg_in, nlp_in], output)
        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        # ── Train ─────────────────────────────────────────────────────────
        self.status_var.set("Training model…")
        self.root.update_idletasks()
        hist = model.fit(
            [Xa_tr, Xe_tr, Xn_tr], y_tr,
            epochs=8, batch_size=16, verbose=1)

        # ── Evaluate ──────────────────────────────────────────────────────
        preds  = model.predict([Xa_te, Xe_te, Xn_te])
        y_true = np.argmax(y_te,   axis=1)
        y_pred = np.argmax(preds,  axis=1)

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred,
                               average="weighted", zero_division=0)
        rec  = recall_score(y_true, y_pred,
                            average="weighted", zero_division=0)
        f1   = f1_score(y_true, y_pred,
                        average="weighted", zero_division=0)

        self.status_var.set("Done.")
        messagebox.showinfo("Performance",
            f"Accuracy : {acc*100:.2f}%\n"
            f"Precision: {prec*100:.2f}%\n"
            f"Recall   : {rec*100:.2f}%\n"
            f"F1-score : {f1*100:.2f}%")

        # ── Confusion Matrix ──────────────────────────────────────────────
        class_ids = sorted(np.unique(y))
        class_names = [EMOTION_LABELS[i] if i < len(EMOTION_LABELS)
                       else str(i) for i in class_ids]
        cm = confusion_matrix(y_true, y_pred, labels=class_ids)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        # ── Accuracy Curve ────────────────────────────────────────────────
        plt.figure()
        plt.plot(hist.history["accuracy"], marker="o")
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # ── ROC Curve (multiclass) ────────────────────────────────────────
        y_bin = label_binarize(y_true, classes=class_ids)
        plt.figure(figsize=(8, 6))
        for i, cls_id in enumerate(class_ids):
            if y_bin.shape[1] == 1:
                # binary after binarize gives shape (n,1) — handle gracefully
                fpr, tpr, _ = roc_curve(y_bin[:, 0], preds[:, cls_id])
            else:
                fpr, tpr, _ = roc_curve(y_bin[:, i], preds[:, cls_id])
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
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionGUI(root)
    root.mainloop()
