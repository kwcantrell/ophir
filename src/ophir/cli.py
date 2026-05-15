import typer

from ophir import register

app = typer.Typer(help="Ophir CLI")
app.add_typer(register.app, name="register")


@app.command()
def serve(
    port: int = typer.Option(7860, help="Gradio server port"),
    share: bool = typer.Option(False, help="Expose a public share link"),
    debug: bool = typer.Option(True, help="Launch Gradio in debug mode"),
) -> None:
    """Launch the Ophir Gradio UI."""
    from ophir import ui

    ui.serve(port=port, share=share, debug=debug)
