import torch
from nemo.collections.asr.models import ASRModel
from huggingface_hub import hf_hub_download

# Download model files
checkpoint_path = hf_hub_download(repo_id="boldvoice/nemotron-phoneme-ipa-v1", filename="checkpoint.ckpt")
tokenizer_model = hf_hub_download(repo_id="boldvoice/nemotron-phoneme-ipa-v1", filename="tokenizer/tokenizer.model")
vocab_txt = hf_hub_download(repo_id="boldvoice/nemotron-phoneme-ipa-v1", filename="tokenizer/vocab.txt")

# Setup tokenizer directory
import shutil
from pathlib import Path
tokenizer_dir = Path("tokenizer")
tokenizer_dir.mkdir(exist_ok=True)
shutil.copy(tokenizer_model, tokenizer_dir / "tokenizer.model")
shutil.copy(vocab_txt, tokenizer_dir / "vocab.txt")

# Load base model and change vocabulary
model = ASRModel.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b")
model.change_vocabulary(new_tokenizer_dir=str(tokenizer_dir), new_tokenizer_type="bpe")

# Load fine-tuned weights
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)
cleaned_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("model.", "", 1) if k.startswith("model.") else k
    cleaned_state_dict[new_key] = v
model.load_state_dict(cleaned_state_dict, strict=False)

# Disable CUDA graphs for inference
if hasattr(model, 'decoding') and model.decoding is not None:
    decoding_cfg = model.cfg.decoding
    decoding_cfg.greedy.loop_labels = False
    decoding_cfg.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(decoding_cfg)

# Configure streaming latency (choose one)
# Option 1: Lowest latency (80ms)
model.encoder.set_default_att_context_size([70, 0])

# Option 2: Highest quality (1.12s)
# model.encoder.set_default_att_context_size([70, 13])

# Move to GPU and set eval mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Transcribe audio
audio_files = ["audio_samples/parola.wav"]
predictions = model.transcribe(audio_files)

# Output: list of IPA strings like "ˈhɛˌloʊ̆"
for pred in predictions:
    if hasattr(pred, 'text'):
        text = pred.text
    else:
        text = str(pred)
    # Remove SentencePiece artifacts
    text = text.replace("▁", "").strip()
    print(text)