from __future__ import annotations

import importlib


def get_module_version(module_list: list[str]) -> dict[str, str]:
    version = {}
    for module_name in module_list:
        module = importlib.import_module(module_name)
        version[module_name] = module.__version__
    return version
