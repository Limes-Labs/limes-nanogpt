"""Poor man's configurator — override globals from a config file."""

import sys

for arg in sys.argv[1:]:
    if arg.endswith(".py"):
        print(f"Overriding config with {arg}")
        with open(arg) as f:
            exec(f.read())  # noqa: S102
        break