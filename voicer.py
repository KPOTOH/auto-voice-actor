import torch

language = 'ru'
speaker = 'kseniya_v2'
sample_rate = 16000
device = torch.device('cpu')

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=speaker)
model.to(device)  # gpu or cpu

audio = model.apply_tts(texts=[example_text],
                        sample_rate=sample_rate)