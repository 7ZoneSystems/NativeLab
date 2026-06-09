import os
import sys


def main(argv=None) -> int:
    """Dispatch to the CLI client when `--cli` is in argv, else launch the GUI."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--cli" in argv:
        return cli_main(argv)
    os.environ["NATIVELAB_GUI"] = "1"
    from nativelab.main import main as gui_main
    gui_main()
    return 0


def cli_main(argv=None) -> int:
    """Run NativeLab's terminal client without importing the GUI entrypoint."""
    argv = list(sys.argv[1:] if argv is None else argv)
    os.environ["NATIVELAB_CLI"] = "1"
    os.environ["NATIVELAB_NO_GUI"] = "1"
    from nativelab.cli import run_cli
    return int(run_cli(argv) or 0)


def _route():
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    _route()
