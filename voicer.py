import re
from itertools import chain
from typing import Tuple, List

import numpy as np
import torch
from scipy.io.wavfile import write
from iteration_utilities import deepflatten


# def split_sentence(sentence: str, max_size: int):
#     assert len(sentence) < max_size


def split_text(text: str, max_size=150, punctuation=None) -> list:
    """ split text to batches of <[max_size] chars (rnn/cnn limit) """
    n = len(text)
    if n <= max_size:
        return [text]

    punctuation = punctuation or ['...', '.', '!', '?', '?!', ';', ',']

    for mark in punctuation.copy():
        punctuation.remove(mark)
        if mark in text.rstrip('.?;!,'):
            # print(mark)
            chanks = text.split(mark)
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


# example_text = 'Я уеду жить в Лондон! А может быть не в Лондон. Буду там по Биг Бену часы сверять; он, наверняка, работает как атомные часы. Безпритязательное произношение?'
example_text = 'Сейчас в ВК происходит активное обсуждение студентами и их родителями вопроса об "уплотнении" , чтобы дать возможность заселиться тем студентам, которые не смогут заселиться в общежития, ввиду отсутствия свободных мест. Мы также хотели бы узнать мнение об этом, как у студентов, проживающих в общежитиях, так и у студентов, нуждающихся в общежитиях. Цель нашего опроса узнать, как вы относитесь к установке двухъярусной кровати в комнате/квартире общежития вместо ОДНОЙ одноярусной, чтобы заселить студентов, нуждающихся в местах в общежитии? Таким образом, в общежитиях БФУ им. И. Канта увеличится на одно свободное место в каждой комнате/квартире, что даст возможность заселить всех нуждающихся в общежитиях студентов.'


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
    audio = apply_tts(
        texts=text_parts,
        model=model,
        sample_rate=sample_rate,
        symbols=symbols,
        device=device
    )
    audio_collection = [x.detach().numpy() for x in audio]
    audio = np.concatenate(audio_collection)
    return audio, sample_rate


def voicer(text: str, filename: str = 'data/audio/aud.wav'):
    text_parts = split_text(text)
    text_parts = expand_text_parts(text_parts)
    audio, sample_rate = make_audio(text_parts)
    write(filename, sample_rate, audio)


if __name__ == "__main__":
    voicer(example_text)
