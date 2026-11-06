"""Tests for template entry IDs and structure paths."""

from pathlib import Path

import pytest

from openfold3.core.data.primitives.structure.template_paths import (
    TemplateEntry,
    iter_template_structure_entries,
)


@pytest.mark.parametrize(
    ("filename", "entry_id"),
    [("3juh.cif", "3juh"), ("3mb6.cif.gz", "3mb6")],
)
def test_from_cif_path_keeps_path_and_removes_full_suffix(filename, entry_id):
    path = Path("templates") / filename

    entry = TemplateEntry.from_cif_path(path)

    assert entry == TemplateEntry(entry_id=entry_id, path=path)
    assert entry.filename == filename


def test_from_cif_path_rejects_unsupported_suffix():
    with pytest.raises(ValueError, match="must end in '.cif' or '.cif.gz'"):
        TemplateEntry.from_cif_path(Path("template.pdb"))


def test_template_entry_is_immutable():
    entry = TemplateEntry(entry_id="3mb6", path=Path("3mb6.cif.gz"))

    with pytest.raises(AttributeError):
        entry.entry_id = "other"


def test_discovery_uses_normalized_ids_and_preserves_paths(tmp_path):
    plain_path = tmp_path / "model.cif"
    compressed_path = tmp_path / "model_rank_2.cif.gz"
    (tmp_path / "ignored.pdb").touch()
    plain_path.touch()
    compressed_path.touch()

    plain_entries = list(iter_template_structure_entries(tmp_path, "cif"))
    compressed_entries = list(iter_template_structure_entries(tmp_path, "cif.gz"))

    assert plain_entries == [TemplateEntry(entry_id="model", path=plain_path)]
    assert compressed_entries == [
        TemplateEntry(entry_id="model_rank_2", path=compressed_path)
    ]


def test_discovery_of_missing_directory_is_empty(tmp_path):
    assert list(iter_template_structure_entries(tmp_path / "missing", "cif")) == []
