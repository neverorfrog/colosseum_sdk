"""Robot subpackage with auto-discovery of robot configurations.

Mirrors colosseum's tasks/__init__.py pattern: iterates direct subpackages and
imports each one, triggering their @register_robot decorators at import time.
"""

import importlib
import pkgutil

for _info in pkgutil.iter_modules(__path__, __name__ + "."):
    importlib.import_module(_info.name)
