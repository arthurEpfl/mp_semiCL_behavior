"""The thesis figures: one script per figure, writing to figures/out/ (``twin.paths.FIGURES``)."""


def show_png(path, width=900):
    """Display a PNG in a notebook (no-op outside IPython)."""
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
    except ImportError:
        return
    if get_ipython() is not None:
        display(Image(filename=str(path), width=width))
