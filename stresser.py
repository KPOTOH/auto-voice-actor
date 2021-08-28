import re

import click
from stressrnn import StressRNN

STRESS_SYMBOL = "+"
stress_rnn = StressRNN()


def default_stressing(text: str, accuracy_threshold=0.75) -> str:
    assert isinstance(text, str)
    stressed_text = stress_rnn.put_stress(
        text, 
        stress_symbol=STRESS_SYMBOL, 
        accuracy_threshold=accuracy_threshold,
    )
    return stressed_text


def individual_stressing(word: str) -> str:
    """
    stress individual word. If rnn cannot stress the word
    `accuracy_threshold` decreased by 0.1 until <=0.
    If rnn didn't cope return unstressed word
    """
    assert STRESS_SYMBOL not in word, "word already stressed"
    accuracy_threshold=0.75

    while accuracy_threshold > 0:
        stressed_word = default_stressing(word, accuracy_threshold)
        if STRESS_SYMBOL in stressed_word:
            return stressed_word
        accuracy_threshold -= 0.1

    return word


def splitter(text: str) -> list:
    splitted_text = text.split()
    return splitted_text


def stress_finisher(stressed_text: str):
    """ if not all words have been stressed after rnn processing
    they will be stressed individually
    """
    tokens = splitter(stressed_text)
    for word in tokens:
        if STRESS_SYMBOL not in word:
            stressed_word = individual_stressing(word)
            #  replace word in text (only 1st occurence)
            stressed_text = stressed_text.replace(word, stressed_word, 1)

    return stressed_text


def move_stress_symbol(stressed_text: str):
    """rnn return text with stress symbol after character (i+1).
    Move stress symbol to previous position (i-1)
    
    replace o+ with +o and so on for all vowels - fastest way"""

    svowels = ["а+", "е+", "ё+", "и+", "о+", "у+", "ы+", "э+", "ю+", "я+",]
    for sv in svowels:
        usv = sv.upper()
        stressed_text = stressed_text.replace(sv, '~' + sv[0])
        stressed_text = stressed_text.replace(usv, '~' + usv[0])
    stressed_text = stressed_text.replace('~', '+')
    return stressed_text


def _move_stress_symbol_naive(stressed_text: str):
    """ rnn return text with stress symbol after character (i+1).
    Move stress symbol to previous position (i-1)
    """
    n = len(stressed_text)
    modified_text = ''
    i = 0
    while i < n - 1:
        if stressed_text[i + 1] == STRESS_SYMBOL:
            modified_text += stressed_text[i + 1] + stressed_text[i]
            i += 1
        else:
            modified_text += stressed_text[i]
        i += 1

    if i + 1 == n:
        modified_text += stressed_text[-1]

    assert len(stressed_text) == len(modified_text)
    return modified_text


def _move_stress_symbol(stressed_text: str):
    """rnn return text with stress symbol after character (i+1).
    Move stress symbol to previous position (i-1)

    regexp version, 4-5 times slower """
    def _find_stressed_vowel(match_obj: re.Match) -> str:
        return match_obj.group()[::-1]

    pattern = '([аеёиоуыэюяАЕЁИОУЫЭЮЯ]\+)'
    new_text = re.sub(pattern, _find_stressed_vowel, stressed_text)
    return new_text


def stressing(text: str, accuracy_threshold: float):
    stressed_text = default_stressing(text, accuracy_threshold)
    stressed_text = stress_finisher(stressed_text)
    stressed_text = move_stress_symbol(stressed_text)
    return stressed_text


@click.command(name="stresser")
@click.argument("text")
@click.option("--accuracy_threshold", default=0.75, show_default=True,
              help="accuracy of rnn output")
def put_stress_to_text(text: str, accuracy_threshold: float):
    stressed_text = stressing(text, accuracy_threshold)
    print(stressed_text)


if __name__ == "__main__":
    put_stress_to_text()
