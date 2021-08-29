import numpy as np
from omegaconf import OmegaConf
import torch
from scipy.io.wavfile import write






def split_text_naive(text: str, max_size: int = 128) -> list:  # ?? TODO
    """ split text to batches of 128 chars """
    n = len(text)
    if n <= 128:
        return [text]

    parts = []
    for i in range(0, n, max_size):
        chank = text[i: i + 128]
        parts.append(chank)
    return parts






# def voicer(text: str):

language = 'ru'
speaker = 'kseniya_16khz'
device = torch.device('cpu')
# model, symbols, sample_rate, example_text, apply_tts = torch.hub.load(
#     repo_or_dir='snakers4/silero-models',
#     model='silero_tts',
#     language=language,
#     speaker=speaker
# )
model, symbols, sample_rate, example_text, apply_tts = torch.hub.load(
    source='local',
    repo_or_dir='/home/mr/.cache/torch/hub/snakers4_silero-models_master',
    model='silero_tts',
    language=language,
    speaker=speaker
)

example_text = 'Я уеду жить в Лондон'

audio = apply_tts(texts=[example_text],
                  model=model,
                  sample_rate=sample_rate,
                  symbols=symbols,
                  device=device)

print(audio)
write('aud.wav', sample_rate, audio[0].detach().numpy())