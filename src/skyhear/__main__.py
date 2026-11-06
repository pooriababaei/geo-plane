"""Allow running as `python -m skyhear`."""

import sys

from .cli import main

sys.exit(main())
