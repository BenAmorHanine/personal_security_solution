from transformers import BertForSequenceClassification, BertTokenizer, pipeline
import whisper
from speechbrain.pretrained import EncoderClassifier

# Globals for caching
_whisper_model = None
_stress_model = None
_tone_pipeline = None
_rhythm_pipeline = None
_bert_pipeline = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def get_stress_model():
    global _stress_model
    if _stress_model is None:
        _stress_model = EncoderClassifier.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            savedir="tmp/stress_model"
        )
    return _stress_model

def get_tone_pipeline():
    global _tone_pipeline
    if _tone_pipeline is None:
        # Replace this with your actual tone detection logic or model
        _tone_pipeline = pipeline("text-classification", model="path/to/tone_model")
    return _tone_pipeline

def get_rhythm_pipeline():
    global _rhythm_pipeline
    if _rhythm_pipeline is None:
        # Replace this with your actual rhythm detection logic or model
        _rhythm_pipeline = pipeline("text-classification", model="path/to/rhythm_model")
    return _rhythm_pipeline

def get_bert_pipeline():
    global _bert_pipeline
    if _bert_pipeline is None:
        model = BertForSequenceClassification.from_pretrained("path/to/bert_model")
        tokenizer = BertTokenizer.from_pretrained("path/to/bert_model")
        _bert_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return _bert_pipeline

def detect_stress(audio_path):
    stress_model = get_stress_model()
    prediction = stress_model.classify_file(audio_path)
    label = prediction[3]  # string label like 'angry', 'happy', etc.
    score = prediction[1].item()  # confidence score
    return {"label": label, "confidence": score}

def detect_tone(text):
    tone = get_tone_pipeline()
    result = tone(text)[0]
    return {"label": result["label"], "confidence": result["score"]}

def detect_rhythm(text):
    rhythm = get_rhythm_pipeline()
    result = rhythm(text)[0]
    return {"label": result["label"], "confidence": result["score"]}


def analyze_audio(audio_path, transcript_text):
    return {
        "stress": detect_stress(audio_path),
        "tone": detect_tone(transcript_text),
        "rhythm": detect_rhythm(transcript_text),
        "content": classify_text(transcript_text)
    }
