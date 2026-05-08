import sys


def _route():
    """Dispatch to the CLI client when `--cli` is in argv, else launch the GUI."""
    if "--cli" in sys.argv[1:]:
        from nativelab.cli import run_cli
        sys.exit(run_cli(sys.argv[1:]))
    from nativelab.main import main
    main()


if __name__ == "__main__":
    _route()
