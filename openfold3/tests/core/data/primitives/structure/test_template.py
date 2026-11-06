# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import openfold3
from openfold3.core.data.io.structure.atom_array import write_atomarray_to_npz
from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.primitives.structure.component import BiotiteCCDWrapper
from openfold3.core.data.primitives.structure.template import (
    TemplateCacheEntry,
    parse_template_structure,
    sample_templates,
)
from openfold3.tests.utils.template_helpers import (
    TEMPLATE_ID,
    make_cache_entry,
    template_structure_array_path,
    write_cache_npz,
)

MMCIFS_DIR = Path(openfold3.__file__).parent / "tests" / "test_data" / "mmcifs"


def _cache_entry():
    """The single template both written to the cache and expected back out."""
    return make_cache_entry([[1, 1], [2, 2]])


def _write_cache_npz(path: Path) -> Path:
    return write_cache_npz(path, {TEMPLATE_ID: _cache_entry()})


def _assembly_data(cache_npz: Path) -> dict:
    return {
        "A": {
            "template_ids": [TEMPLATE_ID],
            "cache_entry_file_path": cache_npz,
        }
    }


def _cache_none(tmp_path: Path) -> None:
    return None


def _cache_tmp_path(tmp_path: Path) -> Path:
    return tmp_path


def _structure_arrays_none(tmp_path: Path) -> None:
    """No preparsed structure arrays -> the existence filter is skipped."""
    return None


def _structure_arrays_dummy(tmp_path: Path) -> Path:
    """A dummy structure erray (empty inside)"""
    array_dir = tmp_path / "arrays"
    struct_path = template_structure_array_path(array_dir)
    struct_path.parent.mkdir(parents=True, exist_ok=True)
    struct_path.touch()
    return array_dir


@pytest.mark.parametrize(
    "make_cache_directory, make_structure_array_directory, expected",
    [
        pytest.param(
            _cache_none, _structure_arrays_none, {}, id="no_cache__no_arrays__drops"
        ),
        pytest.param(
            _cache_none, _structure_arrays_dummy, {}, id="no_cache__with_arrays__drops"
        ),
        pytest.param(
            _cache_tmp_path,
            _structure_arrays_none,
            {TEMPLATE_ID: _cache_entry()},
            id="cache__no_arrays__loads",
        ),
        pytest.param(
            _cache_tmp_path,
            _structure_arrays_dummy,
            {TEMPLATE_ID: _cache_entry()},
            id="cache__with_arrays__loads",
        ),
    ],
)
def test_sample_templates_cache_directory_gate(
    tmp_path, make_cache_directory, make_structure_array_directory, expected
):
    cache_npz = _write_cache_npz(tmp_path / "chainA.npz")

    actual = sample_templates(
        assembly_data=_assembly_data(cache_npz),
        template_cache_directory=make_cache_directory(tmp_path),
        n_templates=4,
        take_top_k=True,  # deterministic: k = min(len(ids), n_templates)
        chain_id="A",
        template_structure_array_directory=make_structure_array_directory(tmp_path),
        template_file_format="npz",
    )

    np.testing.assert_equal(
        {k: dataclasses.asdict(v) for k, v in actual.items()},
        {k: dataclasses.asdict(v) for k, v in expected.items()},
    )


def test_sample_templates_preserves_direct_cif_path(tmp_path):
    direct_cif_path = tmp_path / "3mb6.cif.gz"
    cache_npz = write_cache_npz(
        tmp_path / "chainA.npz",
        {
            TEMPLATE_ID: TemplateCacheEntry(
                index=0,
                release_date="2000-01-01",
                idx_map=np.asarray([[1, 1], [2, 2]]),
                cif_path=direct_cif_path,
            )
        },
    )

    actual = sample_templates(
        assembly_data=_assembly_data(cache_npz),
        template_cache_directory=tmp_path,
        n_templates=1,
        take_top_k=True,
        chain_id="A",
        template_structure_array_directory=None,
        template_file_format="cif",
    )

    assert actual[TEMPLATE_ID].cif_path == direct_cif_path


def test_sample_templates_array_filter_spares_pinned_templates(tmp_path):
    """Pinned CIF templates do not require a preparsed structure array."""
    array_dir = tmp_path / "arrays"
    array_dir.mkdir()
    direct_cif_path = tmp_path / "custom_template.cif"
    cache_npz = write_cache_npz(
        tmp_path / "chainA.npz",
        {
            TEMPLATE_ID: TemplateCacheEntry(
                index=0,
                release_date="2000-01-01",
                idx_map=np.asarray([[1, 1], [2, 2]]),
                cif_path=direct_cif_path,
            )
        },
    )

    actual = sample_templates(
        assembly_data=_assembly_data(cache_npz),
        template_cache_directory=tmp_path,
        n_templates=1,
        take_top_k=True,
        chain_id="A",
        template_structure_array_directory=array_dir,
        template_file_format="npz",
    )

    assert list(actual) == [TEMPLATE_ID]


def test_cif_direct_path_takes_precedence_over_structure_array(
    tmp_path, biotite_ccd_wrapper
):
    """Pinned CIF coordinates are used even when a structure array exists."""
    pinned_path = (
        Path(openfold3.__file__).parent / "tests" / "test_data" / "mmcifs" / "1ubq.cif"
    )
    decoy_path = (
        Path(openfold3.__file__).parent / "tests" / "test_data" / "mmcifs" / "1a8q.cif"
    )
    parsed_decoy = parse_mmcif(decoy_path)
    assert not hasattr(parsed_decoy, "reason")
    _, decoy_array = parsed_decoy
    array_path = tmp_path / "1ubq" / "1ubq_A.npz"
    array_path.parent.mkdir()
    write_atomarray_to_npz(decoy_array, array_path)

    actual = parse_template_structure(
        template_structures_directory=None,
        template_structure_array_directory=tmp_path,
        template_pdb_chain_id="1ubq_A",
        template_file_format="npz",
        ccd=biotite_ccd_wrapper,
        cif_path=pinned_path,
    )
    expected = parse_template_structure(
        template_structures_directory=None,
        template_structure_array_directory=None,
        template_pdb_chain_id="1ubq_A",
        template_file_format="npz",
        ccd=biotite_ccd_wrapper,
        cif_path=pinned_path,
    )

    assert actual is not None and expected is not None
    assert actual.array_length() == expected.array_length()
    np.testing.assert_array_equal(actual.coord, expected.coord)


def test_cif_direct_template_id_with_underscores_is_parsed(tmp_path):
    """A pinned filename containing underscores must still resolve to a chain.

    Template IDs are `f"{entry_id}_{chain_id}"`, and in CIF-direct mode the entry ID is
    the pinned file's stem -- which users routinely give names like `6TEL_relaxed` or
    `model_1_rank_2`. Splitting on every underscore rather than the last one makes the
    whole query die with "too many values to unpack", after preprocessing has already
    accepted the template.

    Regression test for https://github.com/aqlaboratory/openfold-3/issues/406
    """
    pinned = MMCIFS_DIR / "1ubq.cif"
    template_id = "1ubq_not_a_pdb_id_A"

    parsed = parse_template_structure(
        template_structures_directory=None,
        template_structure_array_directory=None,
        template_pdb_chain_id=template_id,
        template_file_format="cif",
        ccd=BiotiteCCDWrapper(),
        cif_path=pinned,
    )

    assert parsed is not None and parsed.array_length() > 0
