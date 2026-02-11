import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import parse_size, resolve_path


def test_parse_size_returns_width_height_tuple():
    assert parse_size("1280x960") == (1280, 960)


def test_resolve_path_resolves_relative_to_project_root():
    resolved = resolve_path("python/config.py")
    assert resolved.name == "config.py"
    assert resolved.exists()
