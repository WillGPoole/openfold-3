"""Tests for template structure path resolution."""

from pathlib import Path

import pytest

from openfold3.core.data.primitives.structure.template_paths import (
    resolve_template_structure_path,
)


def test_literal_directory_resolution(tmp_path: Path):
    structure = tmp_path / "1abc.cif"
    structure.touch()

    assert resolve_template_structure_path(tmp_path, "1abc", "1abc.cif") == structure


def test_missing_template_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No template structure found"):
        resolve_template_structure_path(tmp_path, "missing", "missing.cif")
