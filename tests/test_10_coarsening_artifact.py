# SPDX-FileCopyrightText: 2026 NORCE Research AS
# SPDX-License-Identifier: GPL-3.0
# pylint: disable=R0913,R0914,R0915,R0917

"""Test the optional coarsening artifact (--coarsening-artifact).

The Norne golden test reproduces the reference coarse properties in
examples/norne (C12X26_*.INC) from the fine .prop files and the artifact only,
without re-running the partition."""

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from opm.io.ecl import EclFile as OpmFile

from pycopm.core.pycopm import main
from pycopm.utils.coarsening import redistribute_removed_pore_volume
from pycopm.utils.coarsening_artifact import (
    load_coarsening_artifact,
    validate_coarsening_artifact,
)
from pycopm.utils.files_writer import round_like_e

REPO_ROOT = Path(__file__).parents[1]
HELLO_WORLD = REPO_ROOT / "examples" / "decks" / "HELLO_WORLD.DATA"
NORNE = REPO_ROOT / "examples" / "norne"
NORNE_DECK = NORNE / "NORNE_ATW2013_PYCOPM_STATIC.DATA"

# Options that reproduce examples/norne/C12X26_*.INC byte for byte
NORNE_OPTIONS = [
    "-x",
    "1:2,3:5,6:7,8:11,12:14,15:19,20:24,25:29,30:33,34:38,39:41,42:46",
    "-y",
    "1:2,3:5,6:7,8:10,11:12,13:17,18:22,23:27,28:32,33:37,38:42,43:47,48:52,"
    "53:57,58:62,63:66,67:71,72:76,77:81,82:86,87:90,91:94,95:98,99:102,"
    "103:107,108:112",
    "-s",
    "pvmean",
    "-a",
    "min",
    "-p",
    "1",
    "-q",
    "1",
    # The reference used NORNE_12x26x22.DATA; with -q 1, pycopm opens the
    # case-sensitive name of the OPM output files (upper case), so an upper-case
    # name is required on case-sensitive file systems
    "-w",
    "NORNE_12X26X22.DATA",
    "-l",
    "C12X26_",
]

# PERMZ = kv/kh * PERMX per layer (MULTIPLY in NORNE_ATW2013_PYCOPM_STATIC.DATA)
NORNE_KVKH = [
    0.2, 0.04, 0.25, 0.0, 0.13, 0.13, 0.13, 0.13, 0.09, 0.07, 0.19,
    0.13, 0.64, 0.64, 0.64, 0.64, 0.64, 0.016, 0.004, 0.004, 1.0, 1.0,
]  # fmt: skip

HELLO_WORLD_CASES = {
    "UNIFORM_MIN_P1": ["-c", "5,5,1", "-a", "min", "-p", "1"],
    "RANGES_MODE_P1": [
        "-x",
        "6:10",
        "-y",
        "0,2,2,2,0,0,0,0,0,0,0,0,2,2,2,2,2,2,0,0,0",
        "-a",
        "mode",
        "-p",
        "1",
    ],
    "UNIFORM_MAX_P0": ["-c", "2,3,1", "-a", "max", "-p", "0", "-s", "mean"],
}


def read_property(path, keyword=None):
    """Read one keyword of an OPM include/.prop file with n*value expansion."""
    values = []
    reading = keyword is None
    for raw_line in Path(path).read_text(encoding="ISO-8859-1").splitlines():
        line = raw_line.split("--", maxsplit=1)[0].strip()
        if not line:
            continue
        if not reading:
            reading = line.split()[0].upper() == keyword.upper()
            continue
        if keyword is None and line[0].isalpha():
            continue
        for token in line.split():
            if token.startswith("/"):
                return np.asarray(values, dtype=float)
            if "*" in token:
                count, value = token.split("*")
                values.extend([float(value)] * int(count))
            else:
                values.append(float(token.rstrip("/")))
            if token.endswith("/"):
                return np.asarray(values, dtype=float)
    return np.asarray(values, dtype=float)


def upscale_from_artifact(arrays, rule, fine_values, fine_pore_volume=None):
    """Aggregate one fine property with the rule recorded in the artifact.

    Only the artifact arrays are used, not pycopm. ``fine_values`` holds one
    value per logical fine cell; values of inactive fine cells are ignored."""
    coarse_count = int(np.prod(arrays["coarse_dims"]))
    mapping = arrays["fine_to_coarse_global"]
    mapped = arrays["fine_mapping_valid"]
    used = mapped & arrays["fine_actnum"].astype(bool)
    pore_volume = (
        arrays["fine_pore_volume"] if fine_pore_volume is None else fine_pore_volume
    )
    thickness = arrays["fine_cell_dz"]
    values = np.asarray(fine_values, dtype=float)

    def group_sum(mask, weights):
        return np.bincount(mapping[mask], weights=weights[mask], minlength=coarse_count)

    def divide(numerator, denominator, valid):
        return np.divide(
            numerator, denominator, out=np.zeros(coarse_count), where=valid
        )

    if rule in ("pvmean", "pv_weighted_mean"):
        denominator = group_sum(mapped, pore_volume)
        return divide(
            group_sum(used, values * pore_volume), denominator, denominator > 0
        )
    if rule == "thickness_weighted_mean":
        numerator = group_sum(used, values * thickness)
        denominator = group_sum(mapped, thickness)
        return divide(numerator, denominator, numerator * denominator > 0)
    if rule == "thickness_weighted_harmonic_mean":
        positive = used & (values > 0)
        inverse = np.divide(
            thickness, values, out=np.zeros_like(thickness), where=positive
        )
        numerator = group_sum(used, thickness)
        denominator = group_sum(positive, inverse)
        return divide(numerator, denominator, numerator * denominator > 0)
    if rule == "mean":
        counts = np.bincount(mapping[used], minlength=coarse_count)
        return divide(group_sum(used, values), counts, counts > 0)
    if rule in ("min", "max"):
        result = np.full(coarse_count, np.inf if rule == "min" else -np.inf)
        reducer = np.minimum if rule == "min" else np.maximum
        reducer.at(result, mapping[used], values[used])
        result[np.isinf(result)] = 0.0
        return result
    raise NotImplementedError(f"rule {rule} is not reproduced by this helper")


def comparison_metrics(reconstructed, reference, relative_slack=0.0):
    """Compare values written with seven significant digits.

    The tolerance is half a unit in the seventh significant digit of the
    reference plus ``relative_slack`` times its magnitude."""
    error = np.abs(reconstructed - reference)
    magnitude = np.abs(reference)
    nonzero = magnitude > 0
    unit = np.zeros_like(reference)
    unit[nonzero] = 10.0 ** (np.floor(np.log10(magnitude[nonzero])) - 6)
    tolerance = 0.5 * unit * (1 + 1e-9) + relative_slack * magnitude
    relative = np.divide(error, magnitude, out=np.zeros_like(error), where=nonzero)
    return {
        "count": int(reference.size),
        "max_abs_error": float(error.max()),
        "max_rel_error": float(relative.max()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "outside_tolerance": int(np.count_nonzero(error > tolerance)),
    }


def audit_artifact(arrays, metadata):
    """Apply the independent checks requested for every artifact."""
    fine_count = int(np.prod(arrays["fine_dims"]))
    coarse_count = int(np.prod(arrays["coarse_dims"]))
    mapping = arrays["fine_to_coarse_global"]
    valid = arrays["fine_mapping_valid"]
    assert len(mapping) == fine_count
    assert np.all((mapping[valid] >= 0) & (mapping[valid] < coarse_count))
    assert np.all(mapping[~valid] == -1)

    coarse_active = arrays["coarse_active_global"]
    global_to_active = arrays["coarse_global_to_active"]
    assert np.array_equal(
        global_to_active[coarse_active], np.arange(coarse_active.size)
    )
    inactive = arrays["coarse_actnum_pycopm"] == 0
    assert np.all(global_to_active[inactive] == -1)

    # pycopm aggregates the pore volume of all fine cells of a cluster; the
    # inactive fine cells carry zero pore volume in the INIT file
    fine_pore_volume = arrays["fine_pore_volume"]
    assert np.all(fine_pore_volume[arrays["fine_actnum"] == 0] == 0)
    raw = np.bincount(
        mapping[valid], weights=fine_pore_volume[valid], minlength=coarse_count
    )
    np.testing.assert_allclose(raw, arrays["coarse_pore_volume_raw"], rtol=1e-12)

    # The recorded transfers explain the corrected pore volume. pycopm does
    # not subtract the pore volume of the (inactive) source cells.
    credited = arrays["coarse_pore_volume_raw"].copy()
    debited = arrays["coarse_pore_volume_raw"].copy()
    for source, destination, amount, fraction in zip(
        arrays["pv_transfer_src"],
        arrays["pv_transfer_dst"],
        arrays["pv_transfer_amount"],
        arrays["pv_transfer_fraction"],
    ):
        assert amount == pytest.approx(fraction * raw[source], rel=1e-12)
        credited[destination] += amount
        debited[source] -= amount
        debited[destination] += amount
    final = arrays["coarse_pore_volume_final"]
    np.testing.assert_allclose(credited, final, rtol=1e-12)
    active = arrays["coarse_actnum_pycopm"].astype(bool)
    np.testing.assert_allclose(debited[active], final[active], rtol=1e-12)
    if metadata["pore_volume_correction"]["conserves_total_pore_volume"]:
        assert np.sum(final[active]) == pytest.approx(
            np.sum(fine_pore_volume), rel=1e-12
        )
        np.testing.assert_allclose(debited[~active], 0.0, atol=1e-6 * raw.max())
    validate_coarsening_artifact(arrays)


def text_outputs(folder):
    """Return the generated deck and include files of a pycopm run."""
    return {
        path.name: path.read_bytes()
        for path in sorted(Path(folder).iterdir())
        if path.suffix in (".INC", ".DATA")
    }


@pytest.mark.parametrize("case", HELLO_WORLD_CASES)
def test_artifact_hello_world(flow, tmp_path, monkeypatch, case):
    """The artifact is consistent and does not change the generated deck."""
    monkeypatch.chdir(tmp_path)
    options = ["-f", flow, "-i", str(HELLO_WORLD), "-m", "all", "-w", case]
    options += HELLO_WORLD_CASES[case]
    main([*options, "-o", "without"])
    main([*options, "-o", "with", "--coarsening-artifact", "art/artifact.npz"])

    # Regression: identical deck and include files with and without artifact
    assert text_outputs("with") == text_outputs("without")

    arrays, metadata = load_coarsening_artifact(tmp_path / "art" / "artifact.npz")
    audit_artifact(arrays, metadata)
    assert arrays["metadata_json"].dtype.kind == "U"
    if case == "UNIFORM_MIN_P1":
        assert arrays["pv_transfer_src"].size > 0

    # Mapping and activity agree with the files written by pycopm
    opernum = read_property("with/HELLO_WORLD_OPERNUM_PYCOPM_REFTOCOA.INC", "OPERNUM")
    assert np.array_equal(arrays["fine_to_coarse_global"] + 1, opernum)
    grid = Path(f"with/{metadata['coarsening_options']['include_prefix']}GRID.INC")
    actnum = read_property(grid, "ACTNUM")
    assert np.array_equal(arrays["coarse_actnum_pycopm"], actnum)
    porv = read_property(
        f"with/{metadata['coarsening_options']['include_prefix']}PORV.INC"
    )
    written = arrays["coarse_pore_volume_written"]
    assert np.array_equal(written, arrays["coarse_pore_volume_final"])
    np.testing.assert_array_equal(round_like_e(written, 7), porv)
    init = OpmFile(str(tmp_path / "with" / f"{case}.INIT"))
    dims = np.asarray(init["INTEHEAD"])[8:11]
    assert np.array_equal(dims, arrays["coarse_dims"])

    # Nominal mapping of the active fine cells to packed active coarse cells
    nominal = arrays["fine_active_to_coarse_active_nominal"]
    fine_active = arrays["fine_active_global"]
    coarse = arrays["fine_to_coarse_global"][fine_active]
    assert np.array_equal(nominal >= 0, arrays["coarse_actnum_pycopm"][coarse] == 1)
    assert np.array_equal(
        arrays["coarse_active_global"][nominal[nominal >= 0]], coarse[nominal >= 0]
    )

    # Properties reproduced from the artifact (-precision 7 by default)
    upscaling = metadata["property_upscaling"]["properties"]
    fine_init = OpmFile(str(tmp_path / "with" / "HELLO_WORLD_PREP_PYCOPM_DRYRUN.INIT"))
    compared = 0
    for name in ("poro", "permx"):
        fine_values = np.zeros(arrays["fine_actnum"].size)
        fine_values[fine_active] = np.asarray(fine_init[name.upper()])
        rule = upscaling[name]["rule"]
        reconstructed = upscale_from_artifact(arrays, rule, fine_values)
        written_path = Path(
            f"with/{metadata['coarsening_options']['include_prefix']}{name.upper()}.INC"
        )
        if written_path.is_file():
            reference = read_property(written_path)
            metrics = comparison_metrics(reconstructed, reference)
            assert metrics["outside_tolerance"] == 0, (name, rule, metrics)
            compared += 1
    assert compared > 0


def test_artifact_options(flow, tmp_path, monkeypatch):
    """The option is validated and restricted to deck coarsening."""
    monkeypatch.chdir(tmp_path)
    invalid_calls = [
        ["-i", "input.toml", "--coarsening-artifact", "a.npz"],
        ["-f", flow, "-i", str(HELLO_WORLD), "--coarsening-artifact", "a.npz"],
        ["-f", flow, "-i", str(HELLO_WORLD), "-c", "2,2,1", "--coarsening-artifact", "a.txt"],
        [
            "-f", flow, "-i", str(HELLO_WORLD), "-c", "2,2,1", "-dual", "poro <= 0.1",
            "--coarsening-artifact", "a.npz",
        ],
        [
            "-f", flow, "-i", str(HELLO_WORLD), "-c", "2,2,1", "-m", "prep",
            "--coarsening_artifact", "a.npz",
        ],
    ]  # fmt: skip
    for arguments in invalid_calls:
        with pytest.raises(SystemExit):
            main(arguments)
    assert not list(tmp_path.glob("*.npz"))


def test_redistribution_records():
    """The returned transfers reproduce the in-place pore-volume update."""
    nx, ny, nz = 4, 3, 2
    rng = np.random.default_rng(3)
    cell_groups = np.arange(nx * ny * nz) + 1
    actnum = np.ones(nx * ny * nz, dtype=int)
    actnum[[0, 5, 6, 13, 23]] = 0
    raw = rng.uniform(1.0, 2.0, nx * ny * nz)
    dck = SimpleNamespace(
        output_nx=nx,
        output_ny=ny,
        output_nz=nz,
        original_porv=raw.copy(),
        output_porv=raw.copy(),
        output_actnum=actnum,
    )
    cluster_maximum = np.ones(nx * ny * nz)
    transfers = redistribute_removed_pore_volume(
        dck, cell_groups, actnum.astype(float), cluster_maximum, np.ones(nx * ny * nz)
    )
    assert set(transfers.source.tolist()) == {0, 5, 6, 13, 23}
    assert np.all(actnum[transfers.destination] == 1)
    np.testing.assert_allclose(
        transfers.amount, transfers.fraction * raw[transfers.source]
    )
    credit = np.bincount(transfers.destination, transfers.amount, minlength=raw.size)
    np.testing.assert_allclose(raw + credit, dck.output_porv, rtol=1e-14)
    assert np.sum(dck.output_porv[actnum == 1]) == pytest.approx(np.sum(raw))


def link_or_copy(source, destination):
    """Symlink a file or folder, copying it if symlinks are unavailable."""
    try:
        os.symlink(source, destination)
    except OSError:
        copy = shutil.copytree if Path(source).is_dir() else shutil.copy2
        copy(source, destination)


def run_norne(flow, folder):
    """Coarsen the Norne deck in ``folder`` with the reference options."""
    cwd = Path.cwd()
    os.chdir(folder)
    try:
        main(
            [
                "-f", flow, "-i", NORNE_DECK.name, "-o", ".", "-m", "all",
                *NORNE_OPTIONS, "--coarsening-artifact", "coarsening_artifact.npz",
            ]
        )  # fmt: skip
    finally:
        os.chdir(cwd)
    return load_coarsening_artifact(Path(folder) / "coarsening_artifact.npz")


def write_prop(path, keyword, values):
    """Write a property with full double precision."""
    text = "\n".join(f"{value:.17g}" for value in values)
    Path(path).write_text(f"{keyword}\n{text}\n/\n", encoding="utf8")


@pytest.fixture(scope="module", name="norne_reference")
def fixture_norne_reference(flow, tmp_path_factory):
    """Run the Norne reference coarsening once for all Norne tests."""
    if not NORNE_DECK.is_file():
        pytest.skip("Norne reference case not found")
    folder = tmp_path_factory.mktemp("norne_reference")
    for name in (NORNE_DECK.name, "INCLUDE"):
        link_or_copy(NORNE / name, folder / name)
    arrays, metadata = run_norne(flow, folder)
    return folder, arrays, metadata


def test_artifact_norne_golden(norne_reference):
    """Reproduce the Norne reference coarse properties from the artifact."""
    folder, arrays, metadata = norne_reference

    # Regression: the generated files equal the stored reference run
    reference_files = {
        reference.name: reference for reference in NORNE.glob("C12X26_*.INC")
    }
    reference_files["NORNE_12X26X22.DATA.DATA"] = NORNE / "NORNE_12x26x22.DATA.DATA"
    name = "NORNE_ATW2013_PYCOPM_STATIC_OPERNUM_PYCOPM_REFTOCOA.INC"
    reference_files[name] = NORNE / name
    assert len(reference_files) == 19
    for name, reference in reference_files.items():
        assert (folder / name).read_bytes() == reference.read_bytes(), name

    audit_artifact(arrays, metadata)
    assert metadata["schema_version"] == 2
    assert arrays["fine_dims"].tolist() == [46, 112, 22]
    assert arrays["coarse_dims"].tolist() == [12, 26, 22]
    assert int(arrays["coarse_actnum_pycopm"].sum()) == 2230
    actnum = read_property(folder / "C12X26_GRID.INC", "ACTNUM")
    assert np.array_equal(arrays["coarse_actnum_pycopm"], actnum)
    written = read_property(folder / "C12X26_PORV.INC")
    np.testing.assert_array_equal(
        round_like_e(arrays["coarse_pore_volume_written"], 7), written
    )

    # Fine properties: reference .prop files, PERMZ from the deck MULTIPLY
    fine_count = int(np.prod(arrays["fine_dims"]))
    layer = np.arange(fine_count) // (46 * 112)
    poro = read_property(NORNE / "INCLUDE" / "PETRO" / "PORO_0704.prop", "PORO")
    permx = read_property(NORNE / "INCLUDE" / "PETRO" / "PERM_0704.prop", "PERMX")
    fine_from_prop = {
        "PORO": poro,
        "PERMX": permx,
        "PERMZ": permx * np.asarray(NORNE_KVKH)[layer],
    }

    # Ensemble-reuse coefficients against the reference .prop values
    pv_valid = arrays["fine_pv_per_porosity_valid"]
    assert np.array_equal(pv_valid, arrays["fine_actnum"] == 1)
    pv_from_prop = arrays["fine_pv_per_porosity"] * poro
    np.testing.assert_allclose(
        pv_from_prop[pv_valid], arrays["fine_pore_volume"][pv_valid], rtol=2.0**-22
    )
    multiplier = arrays["fine_permz_multiplier"]
    valid = arrays["fine_permz_multiplier_valid"]
    assert np.array_equal(valid, arrays["fine_actnum"] == 1)
    assert np.all(np.isnan(multiplier[~valid]))
    np.testing.assert_allclose(
        multiplier[valid], np.asarray(NORNE_KVKH)[layer[valid]], rtol=2.0**-23
    )
    reuse = metadata["ensemble_reuse"]["fine_permz_multiplier"]
    assert reuse["constant_per_layer_within_2**-22"]
    assert not metadata["fluid_in_place_correction"]["static_operator"]

    # Fine properties as read by pycopm (float32 INIT of the dry run)
    fine_init = OpmFile(
        str(folder / "NORNE_ATW2013_PYCOPM_STATIC_PREP_PYCOPM_DRYRUN.INIT")
    )
    fine_active = arrays["fine_active_global"]
    upscaling = metadata["property_upscaling"]["properties"]
    report = {"counts": metadata["counts"], "properties": {}}
    for name, from_prop in fine_from_prop.items():
        assert from_prop.size == fine_count
        from_init = np.zeros(fine_count)
        from_init[fine_active] = np.asarray(fine_init[name])
        reference = read_property(NORNE / f"C12X26_{name}.INC", name)
        assert reference.size == 6864
        rule = upscaling[name.lower()]["rule"]
        init_metrics = comparison_metrics(
            upscale_from_artifact(arrays, rule, from_init), reference
        )
        # The .prop values differ from the float32 INIT values by up to
        # 2**-24 relative, which is added to the tolerance
        prop_metrics = comparison_metrics(
            upscale_from_artifact(arrays, rule, from_prop), reference, 2.0**-23
        )
        report["properties"][name] = {
            "rule": rule,
            "from_init": init_metrics,
            "from_prop": prop_metrics,
            "nonzero_reference_values": int(np.count_nonzero(reference)),
        }
        assert init_metrics["outside_tolerance"] == 0, (name, init_metrics)
        assert prop_metrics["outside_tolerance"] == 0, (name, prop_metrics)
    report["pv"] = {
        "transfers": int(arrays["pv_transfer_src"].size),
        "sources": int(np.unique(arrays["pv_transfer_src"]).size),
        "fine_total": float(arrays["fine_pore_volume"].sum()),
        "final_active_total": float(
            arrays["coarse_pore_volume_final"][
                arrays["coarse_actnum_pycopm"] == 1
            ].sum()
        ),
        "written_total": float(arrays["coarse_pore_volume_written"].sum()),
        "written_positive_cells": int(
            np.count_nonzero(arrays["coarse_pore_volume_written"])
        ),
        "max_rel_pv_from_prop_poro": float(
            np.max(
                np.abs(
                    pv_from_prop[pv_valid] / arrays["fine_pore_volume"][pv_valid] - 1
                )
            )
        ),
    }
    print(json.dumps(report, indent=1))


def test_artifact_norne_realization(flow, norne_reference, tmp_path):
    """Reuse the reference artifact for a new PORO/PERMX realization.

    PORO and PERMX are perturbed cell by cell; cells with a reference pore
    volume below 2000 (including those removed by MINPV 500) keep their
    porosity so that the fine ACTNUM is unchanged. A new pycopm run provides
    the truth for the new coarse properties and pore volumes."""
    _, reference, _ = norne_reference
    fine_count = int(np.prod(reference["fine_dims"]))
    rng = np.random.default_rng(2026)
    poro = read_property(NORNE / "INCLUDE" / "PETRO" / "PORO_0704.prop", "PORO")
    permx = read_property(NORNE / "INCLUDE" / "PETRO" / "PERM_0704.prop", "PERMX")
    poro_factor = rng.uniform(0.85, 1.15, fine_count)
    poro_factor[reference["fine_pore_volume"] < 2000] = 1.0
    poro_new = poro * poro_factor
    permx_new = permx * rng.uniform(0.5, 2.0, fine_count)

    link_or_copy(NORNE_DECK, tmp_path / NORNE_DECK.name)
    include = tmp_path / "INCLUDE"
    include.mkdir()
    for entry in (NORNE / "INCLUDE").iterdir():
        if entry.name != "PETRO":
            link_or_copy(entry, include / entry.name)
    (include / "PETRO").mkdir()
    for entry in (NORNE / "INCLUDE" / "PETRO").iterdir():
        if entry.name not in ("PORO_0704.prop", "PERM_0704.prop"):
            link_or_copy(entry, include / "PETRO" / entry.name)
    write_prop(include / "PETRO" / "PORO_0704.prop", "PORO", poro_new)
    write_prop(include / "PETRO" / "PERM_0704.prop", "PERMX", permx_new)
    new, new_metadata = run_norne(flow, tmp_path)
    audit_artifact(new, new_metadata)

    # Static parts of the artifact are unchanged
    for name in (
        "fine_to_coarse_global",
        "fine_actnum",
        "coarse_actnum_pycopm",
        "pv_transfer_src",
        "pv_transfer_dst",
        "pv_transfer_fraction",
        "fine_cell_volume",
        "fine_cell_dz",
        "coarse_cluster_volume",
    ):
        assert np.array_equal(new[name], reference[name]), name

    # Fine pore volume and PERMZ from the reference coefficients
    active = reference["fine_actnum"] == 1
    pv_new = reference["fine_pv_per_porosity"] * poro_new
    np.testing.assert_allclose(
        pv_new[active], new["fine_pore_volume"][active], rtol=2.0**-22
    )
    fine_init = OpmFile(
        str(tmp_path / "NORNE_ATW2013_PYCOPM_STATIC_PREP_PYCOPM_DRYRUN.INIT")
    )
    permz_new = permx_new * reference["fine_permz_multiplier"]
    np.testing.assert_allclose(
        permz_new[active], np.asarray(fine_init["PERMZ"]), rtol=2.0**-22
    )

    # Coarse properties of the new pycopm run from the reference artifact
    rules = new_metadata["property_upscaling"]["properties"]
    report = {}
    for name, values in (
        ("PORO", poro_new),
        ("PERMX", permx_new),
        ("PERMZ", permz_new),
    ):
        reconstructed = upscale_from_artifact(
            reference, rules[name.lower()]["rule"], values, pv_new
        )
        truth = read_property(tmp_path / f"C12X26_{name}.INC", name)
        metrics = comparison_metrics(reconstructed, truth, 2.0**-21)
        report[name] = metrics
        assert metrics["outside_tolerance"] == 0, (name, metrics)

    # Raw and corrected coarse pore volume with the static routing
    coarse_count = int(np.prod(reference["coarse_dims"]))
    raw = np.bincount(
        reference["fine_to_coarse_global"], weights=pv_new, minlength=coarse_count
    )
    np.testing.assert_allclose(raw, new["coarse_pore_volume_raw"], rtol=2.0**-21)
    amount = reference["pv_transfer_fraction"] * raw[reference["pv_transfer_src"]]
    final = raw + np.bincount(
        reference["pv_transfer_dst"], weights=amount, minlength=coarse_count
    )
    np.testing.assert_allclose(final, new["coarse_pore_volume_final"], rtol=2.0**-21)

    # -q 1 is not a static operator: written/final differs between realizations
    coarse_active = reference["coarse_actnum_pycopm"] == 1
    ratio_reference = (
        reference["coarse_pore_volume_written"][coarse_active]
        / reference["coarse_pore_volume_final"][coarse_active]
    )
    ratio_new = (
        new["coarse_pore_volume_written"][coarse_active]
        / new["coarse_pore_volume_final"][coarse_active]
    )
    report["written_over_final"] = {
        "reference_min_max": [
            float(ratio_reference.min()),
            float(ratio_reference.max()),
        ],
        "new_min_max": [float(ratio_new.min()), float(ratio_new.max())],
        "max_abs_difference": float(np.max(np.abs(ratio_new - ratio_reference))),
    }
    assert report["written_over_final"]["max_abs_difference"] > 1e-6
    print(json.dumps(report, indent=1))
