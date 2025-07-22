from transformers import BertForSequenceClassification, BertTokenizer, pipeline
import whisper
import librosa
import numpy as np
import torch
from speechbrain.inference.interfaces import foreign_class
from pyannote.audio import Pipeline
rhythm_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")


stress_model2 = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

def load_audio_file(audio_file, sample_rate=16000): #ok
    # Load audio
    try:
        audio, sr = librosa.load(audio_file, sr=sample_rate)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            return audio, sr
        return audio, sample_rate
    except FileNotFoundError:
        print(f"Error: Audio file '{audio_file}' not found. Please provide a WAV file or record one.")
        return None
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

def transcribe_audio(audio_path, model_name="base"): #ok
    """
    Transcribe the audio file to text using OpenAI's Whisper model.

    Args:
        audio_path (str): Path to the audio file.
        model_name (str): Name of the Whisper model to use (default: "base").

    Returns:
        str: Transcribed text.
    """
    #load file:
    audio_file,sr = load_audio_file(audio_path)
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_file, language="ar")
    return result["text"]

def classify_text(text, model_path= '../models/fine_tuned_model_tunBERT'): #ok
    """
    Classify the given text using the fine-tuned TunBERT model.

    Args:
        text (str): The text to classify.
        model_path (str): Path to the fine-tuned TunBERT model (default: "../../models/fine_tuned_model_tunBERT").

    Returns:
        dict: Classification result with label ("normal", "danger", "hate") and score.
    """
    model = BertForSequenceClassification.from_pretrained(model_path, use_safetensors=True)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # Ensure custom labels are set
    model.config.id2label = {0: "normal", 1: "danger", 2: "hate"}
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
    result = clf(text)[0]
    return result



def audio_features(audio_path): #ok A PART ALAHOU AALAM PROBLEME FI STRESS DETECTION
    audio_file,sample_rate=load_audio_file(audio_path)
    #stress
    try:
        out_prob, score, index, text_lab = stress_model2.classify_file(audio_file)
        stress_label = text_lab[0]
        stress_score = score
    except Exception as e:
        print(f"Stress detection failed: {e}. Using default 'neutral' and 0.0 score.")
        stress_label, stress_score = "neutral", 0.0
    print(f"Stress: {stress_label}, Score: {stress_score:.2f}")

    #rhythm

    try:
        rhythm_out = rhythm_pipeline({"waveform": torch.tensor(audio_file).unsqueeze(0), "sample_rate": sample_rate})
        speech_segments = rhythm_out.get_timeline().support()
        total_duration = librosa.get_duration(y=audio_file, sr=sample_rate)
        speech_duration = sum(seg.duration for seg in speech_segments)
        speech_rate = len(speech_segments) / total_duration if total_duration > 0 else 0
        pause_ratio = 1 - (speech_duration / total_duration) if total_duration > 0 else 0
        rhythm = "fast" if speech_rate > 2.0 else "slow"
        print(f"Rhythm: {rhythm}, Speech Rate: {speech_rate:.2f} segments/s, Pause Ratio: {pause_ratio:.2f}")
    except Exception as e:
        print(f"Rhythm analysis failed: {e}. Using default 'slow'.")
        rhythm = "slow"


    #tone
    try:
        #pitch, _ = librosa.pitches.melodia(audio, sr=sample_rate)
        pitch = librosa.yin(audio_file, fmin=50, fmax=300, sr=sample_rate)
        energy = np.sum(librosa.feature.rms(y=audio_file)**2)
        tone_threshold = 0.5  # Adjust based on testing
        tone = "fearful" if np.mean(pitch[~np.isnan(pitch)]) > tone_threshold or energy > tone_threshold else "calm"
        print(f"Tone: {tone}, Pitch Mean: {np.mean(pitch[~np.isnan(pitch)]):.2f}, Energy: {energy:.2f}")
    except Exception as e:
        print(f"Tone analysis failed: {e}. Using default 'calm'.")
        tone = "calm"

    return {
        "stress": {
            "label": stress_label,
            "score": stress_score
        },
        "rhythm": {
            "label": rhythm,
            "speech_rate": speech_rate,
            "pause_ratio": pause_ratio
        }, 
        "tone": {
            "label": tone,
            "pitch_mean": float(np.mean(pitch[~np.isnan(pitch)])) if np.any(~np.isnan(pitch)) else 0.0,
            "energy": float(energy)
        }
    }


def analyze_vocal(audio_path):
    """
    Comprehensive analysis of vocal input: transcription, classification, and audio features.

    Returns:
        dict: Combined results including transcription, classification, and audio features.
    """
    
    transcription = transcribe_audio(audio_path)
    
    features = audio_features(audio_path)
    
    classification = classify_text(transcription)
    return {
        "transcription": transcription,
        "classification": classification,
        "audio_features": features
    }
