"""Utilities for resolving template structure paths."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


@dataclass(frozen=True)
class TemplateEntry:
    """A template entry ID paired with the structure file that contains it."""

    entry_id: str
    path: Path

    @property
    def filename(self) -> str:
        return self.path.name

    @classmethod
    def from_path(cls, path: Path, suffix: str) -> "TemplateEntry":
        """Create an entry by removing the complete configured filename suffix."""
        return cls(path.name.removesuffix(suffix), path)

    @classmethod
    def from_cif_path(cls, path: Path) -> "TemplateEntry":
        """Create an entry from an uncompressed or gzip-compressed CIF path."""
        for suffix in (".cif.gz", ".cif"):
            if path.name.endswith(suffix):
                return cls.from_path(path, suffix)
        raise ValueError(f"CIF input must end in '.cif' or '.cif.gz': {path}")


def iter_template_structure_paths(
    template_structures_directory: Path,
    template_file_format: str,
) -> Iterator[Path]:
    """Yield all matching template files in deterministic filename order."""
    if not template_structures_directory.is_dir():
        return

    suffix = f".{template_file_format}"
    paths_by_name = {
        path.name: path
        for path in tqdm(
            template_structures_directory.iterdir(),
            desc="Scanning template structures",
        )
        if path.is_file() and path.name.endswith(suffix)
    }
    yield from (paths_by_name[name] for name in sorted(paths_by_name))


def iter_template_structure_entries(
    template_structures_directory: Path,
    template_file_format: str,
) -> Iterator[TemplateEntry]:
    """Yield discovered template files paired with normalized entry IDs."""
    suffix = f".{template_file_format}"
    for path in iter_template_structure_paths(
        template_structures_directory, template_file_format
    ):
        yield TemplateEntry.from_path(path, suffix)
