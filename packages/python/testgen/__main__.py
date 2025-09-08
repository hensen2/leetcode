"""Entry point for running testgen as a module"""

import sys

from .cli.main import main

if __name__ == "__main__":
    main(sys.argv[1:])
