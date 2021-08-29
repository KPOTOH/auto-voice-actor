import click

from stresser import stressing
from voicer import voicing

DEFAULT_TEXT = 'Нам бы, нам бы, нам бы, нам бы всем на дно'

def read_from_file(filename: str):
    with open(filename, 'r') as fin:
        text = fin.read()
    return text


@click.command(name="seiyu")
@click.option("--text", default=DEFAULT_TEXT, show_default=True, help="text to process from command line")
@click.option("-i", "--inpath", default=None, help="load text from file")
@click.option("-o", "--outpath", default="speech.wav", show_default=True, help="path for output")
def perfect_voice(text: str, inpath: str, outpath: str):
    assert text != inpath, "Need pass one of text or from-file argument"
    if inpath is not None:
        text = read_from_file(inpath)
    assert isinstance(text, str), "Text are not string"

    stress_accuracy_threshold = 0.75
    stressed_text = stressing(text, stress_accuracy_threshold)
    voicing(stressed_text, outpath)


if __name__ == "__main__":
    perfect_voice()
