"""Command-line interface for Arabic diacritization.

Usage:
    python -m diacritize "بسم الله الرحمن الرحيم"
    echo "بسم الله الرحمن الرحيم" | python -m diacritize
    python -m diacritize --file input.txt --output output.txt
    python -m diacritize --no-cache "بسم الله الرحمن الرحيم"
"""

from __future__ import annotations

import sys

import click


@click.command()
@click.argument("text", required=False)
@click.option("--file", "-f", "input_file", type=click.Path(exists=True), help="Input file path.")
@click.option("--output", "-o", "output_file", type=click.Path(), help="Output file path.")
@click.option("--no-cache", is_flag=True, help="Use model only (skip sentence cache).")
def main(
    text: str | None,
    input_file: str | None,
    output_file: str | None,
    no_cache: bool,
) -> None:
    """Diacritize Arabic text using BiLSTM + sentence cache."""
    from diacritize.pipeline import Diacritizer

    diacritizer = Diacritizer.from_pretrained(no_cache=no_cache)

    # Determine input text
    if input_file:
        with open(input_file, encoding="utf-8") as f:
            lines = f.read().splitlines()
    elif text:
        lines = [text]
    elif not sys.stdin.isatty():
        lines = sys.stdin.read().splitlines()
    else:
        click.echo("Error: provide text as argument, --file, or pipe via stdin.", err=True)
        sys.exit(1)

    # Diacritize
    results = [diacritizer.diacritize(line) for line in lines if line.strip()]

    # Output
    output = "\n".join(results)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output + "\n")
        click.echo(f"Written to {output_file}")
    else:
        click.echo(output)
