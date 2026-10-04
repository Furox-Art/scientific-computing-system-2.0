"""Enable ``python -m cds2`` to run the command line interface.

Without this module, ``python -m cds2`` fails with "No module named
cds2.__main__", so the documented invocations were the ``cds2`` console script
and ``python -m cds2.cli``. Running the package directly is the conventional
form and costs three lines, so it is supported rather than documented as a
papercut.
"""

from __future__ import annotations

from .cli import main

raise SystemExit(main())
