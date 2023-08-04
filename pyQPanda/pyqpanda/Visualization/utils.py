"""Common visualization utilities."""

MATPLOTLIB_INLINE_BACKENDS = {
    "module://ipykernel.pylab.backend_inline",
    "module://matplotlib_inline.backend_inline",
    "nbAgg",
}


def matplotlib_close_if_inline(figure):
    """
    Ensures that a matplotlib figure is closed if the backend being used renders figures inline.
    
    This function prevents the duplication of figures that can occur when an inline backend captures
    and displays the figure simultaneously with other rendering processes. It is designed to be used
    within the pyQPanda package, which facilitates programming quantum computers and operates on a
    quantum circuit simulator or quantum cloud service.
    
        Args:
            figure (matplotlib.figure.Figure): The matplotlib figure instance to be potentially closed.
    
        Notes:
            The function will only close the figure if the backend in use is recognized as an inline backend.
            The function assumes that matplotlib is already imported and that the figure has been created.
    """
    # This can only called if figure has already been created, so matplotlib must exist.
    import matplotlib.pyplot

    if matplotlib.get_backend() in MATPLOTLIB_INLINE_BACKENDS:
        matplotlib.pyplot.close(figure)
