import click

from stresser import stressing
from voicer import voicing


@click.command(name="seiyu")
@click.argument("text")
@click.option("-f", "--outpath", default='speech.wav', show_default=True,
              help="path for output")
@click.option("--stress_accuracy_threshold", default=0.75, show_default=True,
              help="accuracy of stressing rnn output")
def perfect_voice(text: str, outpath: str, stress_accuracy_threshold: float):
    stressed_text = stressing(text, stress_accuracy_threshold)
    voicing(stressed_text, outpath)


if __name__ == "__main__":
    perfect_voice()
