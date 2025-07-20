import librosa
import torch
import numpy as np

from faster_whisper import WhisperModel

# Load once
model = WhisperModel("base")

# Usage
segments, info = model.transcribe("audio_test.wav")
text = " ".join([segment.text for segment in segments])
# Install libraries if not present (run this cell first)
from speechbrain.pretrained import EncoderClassifier
from pyannote.audio import Pipeline

import whisper
from speechbrain.pretrained import EncoderClassifier
from speechbrain.inference.interfaces import foreign_class
import subprocess
import sys
import torch
from pyannote.audio import Pipeline

def install_libraries():
    required = {'speechbrain', 'pyannote.audio', 'openai-whisper', 'librosa', 'torch', 'numpy'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed

    if missing:
        print("Installing missing libraries...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    else:
        print("All required libraries are installed.")

try:
    import pkg_resources
    install_libraries()
except ImportError:
    print("Please install pkg_resources or run pip install manually.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "speechbrain", "pyannote.audio", "openai-whisper", "librosa", "torch", "numpy"])

# Load pre-trained models
try:
    whisper_transcription_model = whisper.load_model("base")  # Fallback; will be replaced with Tunisian ASR if available
    stress_model = EncoderClassifier.from_hparams(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", savedir="pretrained_models/emotion")
    stress_model2 = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    tone_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token="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")
    rhythm_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
except Exception as e:
    print(f"Error loading models: {e}. Please ensure libraries are installed and token is set.")
    raise



# Function to analyze vocal audio
def analyze_vocal_audio(audio_file, sample_rate=16000):
    """
    Analyzes audio for transcription, stress, tone, rhythm, for personal security alerts.

    Args:
        audio_file (str): Path to the audio file (WAV, 16kHz, mono).
        sample_rate (int): Audio sample rate (default: 16000 Hz).

    Returns:
        dict: Results including transcription, stress, tone, rhythm, and alert level.
    """
    #load models
    install_libraries()

    # Load audio
    try:
        audio, sr = librosa.load(audio_file, sr=sample_rate)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    except FileNotFoundError:
        print(f"Error: Audio file '{audio_file}' not found. Please provide a WAV file or record one.")
        return None
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

    # 1. Transcription
    #we will use whisper.
    whisper_transcription_model = whisper.load_model("base")
    result = whisper_transcription_model.transcribe(audio_file, language="ar")
    transcription = result["text"]
    print(f"Transcription from Whisper: {transcription}")

    # 2. Stress Detection
    try:

        stress_input = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        if torch.cuda.is_available():
            stress_input = stress_input.cuda()
            stress_model = stress_model.cuda()
        #stress_out = stress_model.encode_batch(stress_input, wav_lens=[1.0])

        #stress_out = stress_model.classify_batch(stress_input)
        #tress_prob = stress_out[0].softmax(dim=-1)
        #stress_label = stress_model.hparams.label_encoder.decode(stress_out[0].argmax().item())
        #stress_score = stress_prob.max().item()

        out_prob, score, index, text_lab = stress_model2.classify_file(audio_file)
        stress_label = text_lab[0]
        stress_score = score


        """# Boost stress score with text keywords
        #will be replaced by a model that takes the transcription and detects the stress/danger from it
        stress_words = ["danger", "help", "run", "saha","awnouni","ohrob", "mousiba", "aidez-moi"]
        if any(word in transcription.lower() for word in stress_words):
            stress_score = min(1.0, stress_score + 0.2)
        print(f"Stress Label: {stress_label}, Score: {float(stress_score):.2f}")""" #model created and will be called from testtunbert
    except Exception as e:
        print(f"Stress detection failed: {e}. Using default 'neutral' and 0.0 score.")
        stress_label, stress_score = "neutral", 0.0

    # 3. Tone Analysis
    try:
        #pitch, _ = librosa.pitches.melodia(audio, sr=sample_rate)
        pitch = librosa.yin(audio, fmin=50, fmax=300, sr=sample_rate)
        energy = np.sum(librosa.feature.rms(y=audio)**2)
        tone_threshold = 0.4  # Adjust based on testing
        tone = "fearful" if np.mean(pitch[~np.isnan(pitch)]) > tone_threshold or energy > tone_threshold else "calm"
        print(f"Tone: {tone}, Pitch Mean: {np.mean(pitch[~np.isnan(pitch)]):.2f}, Energy: {energy:.2f}")
    except Exception as e:
        print(f"Tone analysis failed: {e}. Using default 'calm'.")
        tone = "calm"

    # 4. Rhythm Analysis
    try:
        rhythm_out = rhythm_pipeline({"waveform": torch.tensor(audio).unsqueeze(0), "sample_rate": sample_rate})
        speech_segments = rhythm_out.get_timeline().support()
        total_duration = librosa.get_duration(y=audio, sr=sample_rate)
        speech_duration = sum(seg.duration for seg in speech_segments)
        speech_rate = len(speech_segments) / total_duration if total_duration > 0 else 0
        pause_ratio = 1 - (speech_duration / total_duration) if total_duration > 0 else 0
        rhythm = "fast" if speech_rate > 2.0 else "slow"
        print(f"Rhythm: {rhythm}, Speech Rate: {speech_rate:.2f} segments/s, Pause Ratio: {pause_ratio:.2f}")
    except Exception as e:
        print(f"Rhythm analysis failed: {e}. Using default 'slow'.")
        rhythm = "slow"

    # 5. Alert Level
    alert_level = "High" if stress_score > 0.7 or tone == "fearful" or rhythm == "fast" else "Low"
    print(f"Security Alert Level: {alert_level}")

    return {
        "transcription": transcription,
        "stress_label": stress_label,
        "stress_score": float(stress_score),
        "tone": tone,
        "rhythm": rhythm,
        "alert_level": alert_level
    }

from transformers import BertTokenizer, BertForSequenceClassification, pipeline
def classify_text(text: str, model_path: str = "../models/fine_tuned_model_tunBERT") -> tuple:
    model = BertForSequenceClassification.from_pretrained(model_path, ...)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
    result = clf(text)[0]
    return result['label'], round(result['score'], 2)

def audio_hate_classification(audio_file: str, model_path: str = "../models/fine_tuned_model_tunBERT") -> tuple:
    """
    Classifies the audio file using a fine-tuned BERT model.
    
    Args:
        audio_file (str): Path to the audio file.
        model_path (str): Path to the fine-tuned BERT model.
        
    Returns:
        tuple: Predicted label and confidence score


    """
    transcription, stress_label, stress_score, tone, rhythm, alert_level = analyze_vocal_audio(audio_file)
    if transcription:
        label, confidence = classify_text(transcription, model_path)
        print(f"Text Classification: {label}, Confidence: {confidence}")
    else :
        label, confidence = "unknown", 0.0
        print("No transcription available for classification.")
    return transcription, label, confidence, stress_label, stress_score, tone, rhythm, alert_level


"""from transformers import pipeline

classifier = pipeline("text-classification", model="/content/drive/My Drive/Colab Notebooks/fine_tuned_model", tokenizer="/content/drive/My Drive/Colab Notebooks/fine_tuned_model")
test_text = "اسغي ياشعب تونس تدعوا بالاسلام كفار"
result = classifier(test_text)
label_reverse_mapping = {0: 'normal', 1: 'abusive', 2: 'hate'}
predicted_label = label_reverse_mapping[int(result[0]['label'].split('_')[-1])]
print(f"Prediction: {predicted_label}, Confidence: {result[0]['score']:.4f}")"""




#####################################################################
def load_audio(audio_path, sample_rate=16000):

    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        return audio, sample_rate
    except FileNotFoundError:
        print(f"Audio file {audio_path} not found.")
        return None, None
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None
    

def transcribe_audio(audio_file):
    audio, sample_rate = load_audio(audio_file)
    if audio is None:
        return None

    try:
        whisper_transcription_model = whisper.load_model("base")
        result = whisper_transcription_model.transcribe(audio_file, language="ar")
        transcription = result["text"]
        print(f"Transcription from Whisper: {transcription}")
        return transcription
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None
    

def stress_detection(audio_file):
    audio, sample_rate = load_audio(audio_file)
    if audio is None:
        return None

    try:
        stress_model2 = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
        out_prob, score, index, text_lab = stress_model.classify_file(audio_file)
        stress_label = text_lab[0]
        stress_score = score
        print(f"Stress Label: {stress_label}, Score: {float(stress_score):.2f}")
        return {"label": stress_label, "confidence": float(stress_score)}
    except Exception as e:
        print(f"Error detecting stress: {e}")
        return None


def tone_analysis(audio_file):
    audio, sample_rate = load_audio(audio_file)
    if audio is None:
        return None

    try:
        pitch = librosa.yin(audio, fmin=50, fmax=300, sr=sample_rate)
        energy = np.sum(librosa.feature.rms(y=audio)**2)
        tone_threshold = 0.4  # Adjust based on testing
        tone = "fearful" if np.mean(pitch[~np.isnan(pitch)]) > tone_threshold or energy > tone_threshold else "calm"
        print(f"Tone: {tone}, Pitch Mean: {np.mean(pitch[~np.isnan(pitch)]):.2f}, Energy: {energy:.2f}")
        return {"label": tone, "confidence": 1.0}  # Confidence can be adjusted based on testing
    except Exception as e:
        print(f"Error analyzing tone: {e}")
        return None
    

def rhythm_analysis(audio_file):
    audio, sample_rate = load_audio(audio_file)
    if audio is None:
        return None

    try:
        rhythm_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
        rhythm_out = rhythm_pipeline({"waveform": torch.tensor(audio).unsqueeze(0), "sample_rate": sample_rate})
        speech_segments = rhythm_out.get_timeline().support()
        total_duration = librosa.get_duration(y=audio, sr=sample_rate)
        speech_duration = sum(seg.duration for seg in speech_segments)
        speech_rate = len(speech_segments) / total_duration if total_duration > 0 else 0
        pause_ratio = 1 - (speech_duration / total_duration) if total_duration > 0 else 0
        rhythm = "fast" if speech_rate > 2.0 else "slow"
        print(f"Rhythm: {rhythm}, Speech Rate: {speech_rate:.2f} segments/s, Pause Ratio: {pause_ratio:.2f}")
        return {"label": rhythm, "confidence": 1.0}  # Confidence can be adjusted based on testing
    except Exception as e:
        print(f"Error analyzing rhythm: {e}")
        return None
    except Exception as e:
        print(f"Error analyzing rhythm: {e}")
        return None
    
def classify_hate_text(text: str, model_path: str = "../models/fine_tuned_model_tunBERT") -> tuple:
    model = BertForSequenceClassification.from_pretrained(model_path, ...)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
    result = clf(text)[0]
    return result['label'], round(result['score'], 2)
    

def analyze_audio(audio_file):
    """
    Analyzes audio for transcription, stress, tone, rhythm, and alert level.
    """
    transcription = transcribe_audio(audio_file)
    if transcription is None:
        return None

    stress_result = stress_detection(audio_file)
    tone_result = tone_analysis(audio_file)
    rhythm_result = rhythm_analysis(audio_file)

    alert_level = "High" if (stress_result and stress_result['confidence'] > 0.7) or \
                        (tone_result and tone_result['label'] == "fearful") or \
                        (rhythm_result and rhythm_result['label'] == "fast") else "Low"
    return {
        "transcription": transcription, 
        "stress": stress_result,
        "tone": tone_result,
        "rhythm": rhythm_result,
        "alert_level": alert_level
    }
