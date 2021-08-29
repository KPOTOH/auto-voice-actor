import re
from itertools import chain
from typing import Tuple, List

import numpy as np
import torch
from scipy.io.wavfile import write
from iteration_utilities import deepflatten


def split_text(text: str, max_size=150, punctuation=None) -> list:
    """ split text to batches of <[max_size] chars (rnn/cnn limit) """
    n = len(text)
    if n <= max_size:
        return [text]

    punctuation = punctuation or ['\n', '...', '.', '!', '?', '?!', ';', ',']

    for mark in punctuation.copy():
        punctuation.remove(mark)
        if mark in text.rstrip('.?;!,'):
            chanks = text.split(mark)
            if mark != '\n':
                chanks = [x + mark for x in chanks[:-1]] + [chanks[-1]]
            parts = [
                split_text(
                    chank.lstrip(),
                    punctuation=punctuation) for chank in chanks]
            return parts

    print('WoW', len(text), text)  # TODO
    return [text]


def expand_text_parts(parts: list) -> list:
    normal_parts = list(deepflatten(parts, types=list))
    normal_parts = [x for x in normal_parts if len(x) > 1]
    return normal_parts


def make_audio(text_parts: List[str], language='ru', speaker='kseniya_16khz'):
    device = torch.device('cpu')
    # model, symbols, sample_rate, example_text, apply_tts = torch.hub.load(
    #     repo_or_dir='snakers4/silero-models',
    #     model='silero_tts',
    #     language=language,
    #     speaker=speaker
    # )
    model, symbols, sample_rate, _example_text, apply_tts = torch.hub.load(
        source='local',
        repo_or_dir='/home/mr/.cache/torch/hub/snakers4_silero-models_master',
        model='silero_tts',
        language=language,
        speaker=speaker
    )
    audio_collection = []
    for i in range(0, len(text_parts), 4):
        batch = text_parts[i: i+4]
        print('Processing:', batch)
        audio_sample = apply_tts(
            texts=batch,
            model=model,
            sample_rate=sample_rate,
            symbols=symbols,
            device=device
        )
        for aud in audio_sample:
            audio_collection.append(aud.detach().numpy())
    
    audio = np.concatenate(audio_collection)
    return audio, sample_rate


def voicing(text: str, filename='data/audio/aud.wav'):
    text_parts = split_text(text)
    text_parts = expand_text_parts(text_parts)
    audio, sample_rate = make_audio(text_parts)
    write(filename, sample_rate, audio)


if __name__ == "__main__":
    example_text = 'Я уеду жить в Лондон! Безпритязательное произношение?'
    voicing(example_text)
