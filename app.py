import soundfile as sf
import librosa
from transformers import pipeline

#https://huggingface.co/mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme
#https://huggingface.co/spaces/lgtitony/doan
#https://huggingface.co/spaces/lgtitony/doan/tree/main

def load_audio_16k(path: str):
    """
    Load WAV file and resample to 16kHz if needed (pure Python).
    Returns (audio_array, sample_rate=16000)
    """
    audio, sr = sf.read(path, dtype="float32")

    # Convert stereo -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if necessary
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    return audio, sr


def transcribe_phoneme(audio_path: str) -> str:
    audio, sr = load_audio_16k(audio_path)

    result = pipe({
        "array": audio,
        "sampling_rate": sr
    })

    return result["text"]


pipe = pipeline(model="mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme")
audio_file = "audio_samples/parola.wav"  # <-- your path here
phonemes = transcribe_phoneme(audio_file)
print("Phoneme output:")
print(phonemes)
