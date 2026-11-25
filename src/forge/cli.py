"""Command-line interface for ForgeAI."""

import typer

from forge import __version__

app = typer.Typer(
    name="forge",
    help="ForgeAI: autonomous factory for building, testing, and refining agents.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """ForgeAI CLI root command."""


@app.command()
def version() -> None:
    """Show the installed ForgeAI version."""
    typer.echo(f"forge {__version__}")


if __name__ == "__main__":
    app()
