__version__ = "1.0.0"


def main():
    """Lazy wrapper so desktop-only deps aren't imported at package level."""
    from .main import main as _main
    return _main()

