# SPDX-FileCopyrightText: 2026 NORCE Research AS
# SPDX-License-Identifier: GPL-3.0
# pylint: disable=R0914

"""Export and validate the fine-to-coarse bookkeeping of a deck coarsening run.

The artifact is a compressed NumPy archive (``.npz``) containing only numeric
arrays and one JSON metadata string. All arrays are captured from the objects
used by :mod:`pycopm.utils.coarsening` while the coarse deck is generated; no
mapping is reconstructed from the written deck files.

Index conventions
-----------------
All indices are zero based. Logical (global) cell indices follow the Eclipse
natural ordering used by pycopm, with ``i`` varying fastest, then ``j``, then
``k``::

    global = i + nx * (j + ny * k)

which is the C-order flattening of an array shaped ``(nz, ny, nx)``. Packed
active indices enumerate active cells in increasing global order.
"""

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

import pycopm
from pycopm.config.config import ConfigViaDeck
from pycopm.utils.coarsening import CoarseningMaps, PoreVolumeTransfers

SCHEMA_NAME = "pycopm-coarsening-artifact"
SCHEMA_VERSION = 2
FLATTENING_ORDER = (
    "Eclipse natural order, zero based: global = i + nx*(j + ny*k), i fastest, "
    "then j, then k (C-order flattening of an array shaped (nz, ny, nx))"
)
TRANSMISSIBILITY_KEYWORDS = ("tranx", "trany", "tranz")
PV_WEIGHTED_KEYWORDS = ("poro", "swatinit", "disperc", "thconr")


@dataclass(slots=True)
class CoarseningArtifactRecord:
    """Snapshots taken while :func:`pycopm.utils.generate_decks.create_deck` runs.

    The arrays are copied at the moment pycopm holds them, so later in-place
    updates of ``dck.output_porv`` or ``dck.output_actnum`` do not alter them.
    """

    #: Coarse pore volume directly after aggregating the fine pore volume.
    coarse_pore_volume_raw: NDArray

    #: Coarse ACTNUM decided by pycopm (active-cell method and jump removal).
    coarse_actnum: NDArray

    #: Mask of coarse cells retained by the depth-jump criterion (``-j``).
    jump_removal_mask: NDArray

    #: Transfers recorded by the pore-volume correction (empty if not applied).
    transfers: PoreVolumeTransfers = field(default_factory=PoreVolumeTransfers)

    #: Coarse pore volume after the pore-volume correction (``-p``).
    coarse_pore_volume_final: NDArray = field(default_factory=lambda: np.array([]))


def write_coarsening_artifact(
    dck: ConfigViaDeck,
    coarsening: CoarseningMaps,
    record: CoarseningArtifactRecord,
    coarsening_options: dict[str, Any],
) -> Path:
    """Build, validate, and write the coarsening artifact.

    Parameters
    ----------
    dck
        Deck configuration after the coarsened deck has been written.
    coarsening
        Coarsening maps populated by :func:`create_coarsening_maps` and
        :func:`coarsen_properties`.
    record
        Snapshots taken during the coarsening workflow.
    coarsening_options
        Command-line options relevant to the coarsening.

    Returns
    -------
    Path
        Path of the written ``.npz`` file."""
    arrays = build_coarsening_artifact(dck, coarsening, record, coarsening_options)
    validate_coarsening_artifact(arrays)
    output_path = Path(dck.coarsening_artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return output_path


def build_coarsening_artifact(
    dck: ConfigViaDeck,
    coarsening: CoarseningMaps,
    record: CoarseningArtifactRecord,
    coarsening_options: dict[str, Any],
) -> dict[str, NDArray]:
    """Collect the artifact arrays from the pycopm coarsening objects.

    Parameters
    ----------
    dck, coarsening, record, coarsening_options
        See :func:`write_coarsening_artifact`.

    Returns
    -------
    dict[str, NDArray]
        Artifact arrays keyed by their NPZ names, including ``metadata_json``."""
    fine_dims = np.array(
        [dck.original_nx, dck.original_ny, dck.original_nz], dtype=np.int64
    )
    coarse_dims = np.array(
        [dck.output_nx, dck.output_ny, dck.output_nz], dtype=np.int64
    )
    fine_count = int(np.prod(fine_dims))
    coarse_count = int(np.prod(coarse_dims))

    # CoarseningMaps.cell_groups is the one-based coarse id of every logical
    # fine cell; it is the partition used by all coarse aggregations.
    fine_to_coarse = np.asarray(coarsening.cell_groups, dtype=np.int64) - 1
    reference_to_coarse = np.asarray(coarsening.reference_to_coarse, dtype=np.int64) - 1
    if not np.array_equal(fine_to_coarse, reference_to_coarse):
        raise ValueError(
            "pycopm cell groups and the per-axis reference-to-coarse map disagree"
        )
    fine_mapping_valid = np.ones(fine_count, dtype=bool)

    fine_actnum = np.asarray(dck.original_active_cell_mask, dtype=np.int8)
    coarse_actnum = np.asarray(record.coarse_actnum, dtype=np.int8)
    fine_active_global = np.flatnonzero(fine_actnum).astype(np.int64)
    coarse_active_global = np.flatnonzero(coarse_actnum).astype(np.int64)
    fine_global_to_active = _global_to_active(fine_active_global, fine_count)
    coarse_global_to_active = _global_to_active(coarse_active_global, coarse_count)
    fine_active_to_coarse_active_nominal = coarse_global_to_active[
        fine_to_coarse[fine_active_global]
    ]

    transfers = record.transfers
    coarse_pore_volume_raw = np.asarray(record.coarse_pore_volume_raw, dtype=np.float64)
    coarse_pore_volume_final = np.asarray(
        record.coarse_pore_volume_final, dtype=np.float64
    )
    coarse_pore_volume_written = np.asarray(dck.output_porv, dtype=np.float64)
    reuse_arrays, reuse_metadata = _ensemble_reuse_arrays(dck, fine_active_global)

    metadata = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "index_base": 0,
        "fine_flattening_order": FLATTENING_ORDER,
        "coarse_flattening_order": FLATTENING_ORDER,
        "pycopm_version_or_commit": _pycopm_version(),
        "input_deck": dck.original_deck_name,
        "output_deck": dck.output_deck_name,
        "coarsening_options": coarsening_options,
        "counts": {
            "fine_logical": fine_count,
            "fine_active": int(fine_active_global.size),
            "coarse_logical": coarse_count,
            "coarse_active_pycopm": int(coarse_active_global.size),
            "fine_mapped": int(np.count_nonzero(fine_mapping_valid)),
            "fine_unmapped": int(np.count_nonzero(~fine_mapping_valid)),
            "pv_transfers": int(transfers.source.size),
        },
        "definitions": _array_definitions(),
        "pore_volume_correction": _describe_pore_volume_correction(dck),
        "fluid_in_place_correction": _describe_fluid_in_place_correction(dck),
        "property_upscaling": _describe_property_upscaling(dck, coarsening),
        "ensemble_reuse": reuse_metadata,
    }
    return {
        "fine_dims": fine_dims,
        "coarse_dims": coarse_dims,
        "fine_to_coarse_global": fine_to_coarse,
        "fine_mapping_valid": fine_mapping_valid,
        "fine_actnum": fine_actnum,
        "coarse_actnum_pycopm": coarse_actnum,
        "fine_active_global": fine_active_global,
        "coarse_active_global": coarse_active_global,
        "fine_global_to_active": fine_global_to_active,
        "coarse_global_to_active": coarse_global_to_active,
        "fine_active_to_coarse_active_nominal": fine_active_to_coarse_active_nominal,
        "fine_cell_volume": np.asarray(coarsening.fine_cell_volume, dtype=np.float64),
        "coarse_cluster_volume": np.asarray(
            coarsening.coarse_cluster_volume, dtype=np.float64
        ),
        "fine_cell_dx": np.asarray(coarsening.fine_cell_dx, dtype=np.float64),
        "fine_cell_dy": np.asarray(coarsening.fine_cell_dy, dtype=np.float64),
        "fine_cell_dz": np.asarray(coarsening.fine_cell_dz, dtype=np.float64),
        "coarse_jump_retained": np.asarray(record.jump_removal_mask, dtype=np.int8),
        "coarse_well_cells": np.asarray(coarsening.well_cells, dtype=np.int64),
        "fine_pore_volume": np.asarray(dck.original_porv, dtype=np.float64),
        "coarse_pore_volume_raw": coarse_pore_volume_raw,
        "coarse_pore_volume_final": coarse_pore_volume_final,
        "coarse_pore_volume_written": coarse_pore_volume_written,
        "pv_transfer_src": np.asarray(transfers.source, dtype=np.int64),
        "pv_transfer_dst": np.asarray(transfers.destination, dtype=np.int64),
        "pv_transfer_amount": np.asarray(transfers.amount, dtype=np.float64),
        "pv_transfer_fraction": np.asarray(transfers.fraction, dtype=np.float64),
        **reuse_arrays,
        "metadata_json": np.array(json.dumps(metadata, indent=1, sort_keys=True)),
    }


def _ensemble_reuse_arrays(
    dck: ConfigViaDeck, fine_active_global: NDArray
) -> tuple[dict[str, NDArray], dict[str, Any]]:
    """Derive per-cell coefficients for new porosity and permeability fields.

    Both coefficients are ratios of the fine INIT arrays read by pycopm
    (float32 values of the OPM dry run), so they reproduce the reference
    inputs of pycopm to rounding. They are undefined for inactive fine cells,
    for which the INIT file holds no values."""
    fine_count = dck.original_cell_count
    init_file = dck.init_file

    def scatter(keyword: str) -> NDArray:
        values = np.zeros(fine_count, dtype=np.float64)
        if init_file.count(keyword):
            values[fine_active_global] = np.asarray(
                init_file[keyword], dtype=np.float64
            )
        return values

    pore_volume = np.asarray(dck.original_porv, dtype=np.float64)
    poro = scatter("PORO")
    pv_valid = (pore_volume > 0) & (poro > 0)
    pv_per_porosity = np.divide(
        pore_volume, poro, out=np.zeros(fine_count), where=pv_valid
    )

    permx = scatter("PERMX")
    permz = scatter("PERMZ")
    permz_valid = np.zeros(fine_count, dtype=bool)
    if init_file.count("PERMX") and init_file.count("PERMZ"):
        permz_valid[fine_active_global] = permx[fine_active_global] > 0
    permz_multiplier = np.divide(
        permz, permx, out=np.full(fine_count, np.nan), where=permz_valid
    )
    if not np.allclose(
        permx[permz_valid] * permz_multiplier[permz_valid],
        permz[permz_valid],
        rtol=1e-14,
        atol=0,
    ):
        raise ValueError(
            "invalid coarsening artifact: fine_permz_multiplier does not "
            "reproduce the reference PERMZ"
        )

    # Diagnostic: is the multiplier one constant per layer (deck COPY/MULTIPLY)?
    layer = np.arange(fine_count) // (dck.original_nx * dck.original_ny)
    layer_values: list[float | None] = []
    layer_constant = True
    for k in range(dck.original_nz):
        values = permz_multiplier[permz_valid & (layer == k)]
        if values.size == 0:
            layer_values.append(None)
            continue
        median = float(np.median(values))
        layer_values.append(median)
        layer_constant &= bool(
            np.all(np.abs(values - median) <= 2.0**-22 * abs(median))
        )
    active = pore_volume > 0
    metadata = {
        "fine_pv_per_porosity": {
            "definition": "PORV / PORO of the fine INIT file for active cells with "
            "PORO > 0, zero elsewhere; fine_pore_volume = fine_pv_per_porosity * "
            "fine_reference_poro holds to rounding",
            "valid_cells": int(np.count_nonzero(pv_valid)),
            "active_cells_not_representable": int(np.count_nonzero(active & ~pv_valid)),
            "assumes": "OPM pore volume is proportional to the cell's own "
            "porosity (fixed NTG, geometry, MULTPV and fine ACTNUM, including "
            "cells removed by MINPV)",
            "reference_precision": "PORV and PORO are float32 in the INIT file; "
            "a new float64 porosity field gives pore volumes within about "
            "2**-23 relative of the float32 PORV that pycopm would read",
        },
        "fine_permz_multiplier": {
            "definition": "PERMZ / PERMX of the fine INIT file for active cells "
            "with PERMX > 0, NaN elsewhere (pycopm ignores inactive cells)",
            "valid_cells": int(np.count_nonzero(permz_valid)),
            "active_cells_undefined": int(np.count_nonzero(active & ~permz_valid)),
            "constant_per_layer_within_2**-22": layer_constant,
            "layer_median_diagnostic": layer_values,
            "assumes": "the deck derives PERMZ from PERMX by multiplication "
            "(e.g., COPY + MULTIPLY); ratios carry float32 rounding",
        },
    }
    arrays = {
        "fine_reference_poro": poro,
        "fine_pv_per_porosity": pv_per_porosity,
        "fine_pv_per_porosity_valid": pv_valid,
        "fine_permz_multiplier": permz_multiplier,
        "fine_permz_multiplier_valid": permz_valid,
    }
    return arrays, metadata


def load_coarsening_artifact(path: str | Path) -> tuple[dict[str, NDArray], dict]:
    """Load an artifact without enabling pickle.

    Parameters
    ----------
    path
        Location of the ``.npz`` file.

    Returns
    -------
    arrays, metadata
        All arrays keyed by name and the decoded metadata dictionary."""
    with np.load(path, allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files}
    return arrays, json.loads(str(arrays["metadata_json"]))


def validate_coarsening_artifact(
    arrays: Mapping[str, NDArray],
    rtol: float = 1e-10,
) -> None:
    """Check the internal consistency of an artifact.

    Parameters
    ----------
    arrays
        Artifact arrays, e.g., from :func:`load_coarsening_artifact`.
    rtol
        Relative tolerance for floating-point sums.

    Raises
    ------
    ValueError
        If a check fails."""
    metadata = json.loads(str(arrays["metadata_json"]))
    fine_count = int(np.prod(arrays["fine_dims"]))
    coarse_count = int(np.prod(arrays["coarse_dims"]))
    fine_to_coarse = arrays["fine_to_coarse_global"]
    valid = arrays["fine_mapping_valid"]
    _check(metadata["schema"] == SCHEMA_NAME, "unexpected schema name")
    _check(metadata["index_base"] == 0, "index base must be zero")

    # Shapes of per-cell arrays
    for name in (
        "fine_to_coarse_global",
        "fine_mapping_valid",
        "fine_actnum",
        "fine_global_to_active",
        "fine_cell_volume",
        "fine_cell_dx",
        "fine_cell_dy",
        "fine_cell_dz",
        "fine_pore_volume",
        "fine_reference_poro",
        "fine_pv_per_porosity",
        "fine_pv_per_porosity_valid",
        "fine_permz_multiplier",
        "fine_permz_multiplier_valid",
    ):
        _check(
            arrays[name].shape == (fine_count,), f"{name} must have {fine_count} values"
        )
    for name in (
        "coarse_actnum_pycopm",
        "coarse_global_to_active",
        "coarse_cluster_volume",
        "coarse_jump_retained",
        "coarse_pore_volume_raw",
        "coarse_pore_volume_final",
        "coarse_pore_volume_written",
    ):
        _check(
            arrays[name].shape == (coarse_count,),
            f"{name} must have {coarse_count} values",
        )

    # Mapping range; unmapped fine cells must be flagged with -1
    _check(
        np.all((fine_to_coarse[valid] >= 0) & (fine_to_coarse[valid] < coarse_count)),
        "fine_to_coarse_global out of range",
    )
    _check(np.all(fine_to_coarse[~valid] == -1), "unmapped fine cells must be -1")

    # Active-index maps
    for prefix, count in (("fine", fine_count), ("coarse", coarse_count)):
        actnum = arrays["coarse_actnum_pycopm" if prefix == "coarse" else "fine_actnum"]
        active_global = arrays[f"{prefix}_active_global"]
        global_to_active = arrays[f"{prefix}_global_to_active"]
        _check(
            np.array_equal(active_global, np.flatnonzero(actnum)),
            f"{prefix}_active_global does not match ACTNUM",
        )
        _check(
            np.array_equal(
                global_to_active[active_global], np.arange(active_global.size)
            ),
            f"{prefix}_global_to_active is not the inverse of {prefix}_active_global",
        )
        inactive = np.ones(count, dtype=bool)
        inactive[active_global] = False
        _check(
            np.all(global_to_active[inactive] == -1),
            f"inactive {prefix} cells must map to -1",
        )
    nominal = arrays["coarse_global_to_active"][
        fine_to_coarse[arrays["fine_active_global"]]
    ]
    _check(
        np.array_equal(arrays["fine_active_to_coarse_active_nominal"], nominal),
        "fine_active_to_coarse_active_nominal is inconsistent",
    )

    # Volumes aggregate through the nominal mapping
    coarse_volume = np.bincount(
        fine_to_coarse[valid],
        weights=arrays["fine_cell_volume"][valid],
        minlength=coarse_count,
    )
    _check(
        np.allclose(coarse_volume, arrays["coarse_cluster_volume"], rtol=rtol, atol=0),
        "coarse_cluster_volume is not the sum of the fine cell volumes",
    )

    # Fine pore volume from the porosity coefficient (exact to rounding for the
    # reference; cells without a coefficient must have zero pore volume)
    pv_valid = arrays["fine_pv_per_porosity_valid"]
    reconstructed = arrays["fine_pv_per_porosity"] * arrays["fine_reference_poro"]
    fine_pore_volume = arrays["fine_pore_volume"]
    _check(
        np.allclose(
            reconstructed[pv_valid], fine_pore_volume[pv_valid], rtol=1e-14, atol=0
        )
        and np.all(arrays["fine_pv_per_porosity"][~pv_valid] == 0),
        "fine_pv_per_porosity * fine_reference_poro does not reproduce the "
        "fine pore volume",
    )
    _check(
        np.all(
            np.isfinite(
                arrays["fine_permz_multiplier"][arrays["fine_permz_multiplier_valid"]]
            )
        ),
        "fine_permz_multiplier must be finite where valid",
    )

    # Raw pore volume: pycopm sums the pore volume of all fine cells in a
    # cluster (inactive fine cells carry zero pore volume in the INIT file)
    raw = arrays["coarse_pore_volume_raw"]
    aggregated = np.bincount(
        fine_to_coarse[valid],
        weights=arrays["fine_pore_volume"][valid],
        minlength=coarse_count,
    )
    _check(
        np.allclose(aggregated, raw, rtol=rtol, atol=0),
        "coarse_pore_volume_raw is not reproduced by the fine-to-coarse mapping",
    )

    # Pore-volume transfers
    source = arrays["pv_transfer_src"]
    destination = arrays["pv_transfer_dst"]
    amount = arrays["pv_transfer_amount"]
    fraction = arrays["pv_transfer_fraction"]
    final = arrays["coarse_pore_volume_final"]
    coarse_actnum = arrays["coarse_actnum_pycopm"].astype(bool)
    _check(
        source.shape == destination.shape == amount.shape == fraction.shape,
        "pv_transfer arrays must have the same length",
    )
    if source.size:
        _check(
            np.all((source >= 0) & (source < coarse_count))
            and np.all((destination >= 0) & (destination < coarse_count)),
            "pv_transfer indices out of range",
        )
        _check(
            not np.any(coarse_actnum[source]), "pv_transfer sources must be inactive"
        )
        _check(
            np.all(coarse_actnum[destination]),
            "pv_transfer destinations must be active",
        )
        _check(
            np.allclose(amount, fraction * raw[source], rtol=rtol, atol=0),
            "pv_transfer_amount differs from pv_transfer_fraction * raw source PV",
        )
    credit = np.bincount(destination, weights=amount, minlength=coarse_count)
    debit = np.bincount(source, weights=amount, minlength=coarse_count)
    # pycopm only credits destinations; source cells keep their raw value in
    # the pore-volume array and are removed from the model through ACTNUM
    _check(
        np.allclose(raw + credit, final, rtol=rtol, atol=0),
        "pv transfers do not explain coarse_pore_volume_final",
    )
    effective = raw - debit + credit
    _check(
        np.allclose(effective[coarse_actnum], final[coarse_actnum], rtol=rtol, atol=0),
        "debit/credit transfers do not explain the active coarse pore volume",
    )
    if metadata["pore_volume_correction"]["conserves_total_pore_volume"]:
        _check(
            np.isclose(
                np.sum(final[coarse_actnum]),
                np.sum(arrays["fine_pore_volume"]),
                rtol=rtol,
                atol=0,
            ),
            "active coarse pore volume does not conserve the fine pore volume",
        )


def _check(condition: Any, message: str) -> None:
    """Raise a ``ValueError`` if a validation condition does not hold."""
    if not condition:
        raise ValueError(f"invalid coarsening artifact: {message}")


def _global_to_active(active_global: NDArray, count: int) -> NDArray:
    """Return the packed active index of every global cell, or -1."""
    global_to_active = np.full(count, -1, dtype=np.int64)
    global_to_active[active_global] = np.arange(active_global.size, dtype=np.int64)
    return global_to_active


def _pycopm_version() -> str:
    """Return the package version and, if available, the git commit."""
    version = f"pycopm {pycopm.__version__}"
    source_directory = Path(pycopm.__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_directory,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            cwd=source_directory,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return version
    return f"{version} (git {commit}{', modified' if status else ''})"


def _array_definitions() -> dict[str, str]:
    """Describe the semantics of each exported array."""
    return {
        "fine_to_coarse_global": "coarse logical index of every logical fine cell, "
        "taken from CoarseningMaps.cell_groups - 1 (the partition used by all "
        "pycopm aggregations); inactive fine cells are also mapped",
        "fine_mapping_valid": "True where fine_to_coarse_global holds a mapping; "
        "pycopm maps every logical fine cell, so all values are True",
        "fine_actnum": "pycopm fine activity: PORV > 0 in the INIT file of the "
        "OPM dry run of the input deck (after OPM MINPV/PINCH processing)",
        "coarse_actnum_pycopm": "coarse ACTNUM decided by pycopm (active-cell "
        "method -a and depth-jump removal -j), as written to the GRID include; "
        "it does not include cells later deactivated by OPM Flow",
        "fine_active_to_coarse_active_nominal": "packed active coarse index of "
        "the nominal cluster of each active fine cell, -1 if that cluster is "
        "inactive; independent of pore-volume transfers",
        "fine_cell_volume": "OPM EGrid.cellvolumes() of the fine grid, as used "
        "by pycopm",
        "coarse_cluster_volume": "sum of fine_cell_volume over the fine cells of "
        "each nominal cluster (pycopm total_volume, used as a denominator by "
        "some aggregations); it is NOT the geometric volume of the resulting "
        "coarse corner-point cell, which pycopm never computes and which can "
        "differ because the coarse cell faces are rebuilt from the retained "
        "pillars and ZCORN surfaces",
        "fine_cell_dx/dy/dz": "cell lengths computed by pycopm from the corner "
        "coordinates (mean edge lengths); dz is the thickness used by the "
        "physical permeability aggregation",
        "coarse_jump_retained": "1 where the depth-jump criterion (-j) keeps the "
        "coarse cell, 0 where it removes it (all 1 without -j)",
        "coarse_well_cells": "coarse cells with well completions; only used when "
        "-t > 0, where coarse permeabilities outside them are set to zero",
        "fine_pore_volume": "PORV of the fine INIT file (reference realization)",
        "coarse_pore_volume_raw": "bincount of fine_pore_volume over "
        "fine_to_coarse_global, before any correction",
        "coarse_pore_volume_final": "pycopm output pore volume after the -p "
        "correction; equals raw + bincount(pv_transfer_dst, pv_transfer_amount)",
        "coarse_pore_volume_written": "pycopm output pore volume when the "
        "artifact was written; differs from final only if the fluid-in-place "
        "correction (-q 1) rescaled it using OPM Flow results",
        "pv_transfer_*": "one row per (source, destination) pore-volume "
        "transfer performed by redistribute_removed_pore_volume; logical coarse "
        "indices; amount = fraction * coarse_pore_volume_raw[src]",
        "fine_reference_poro": "PORO of the fine INIT file (float32 values), "
        "zero for inactive fine cells",
        "fine_pv_per_porosity": "PORV / PORO of the fine INIT file; a new "
        "realization with the same fine ACTNUM has fine pore volume "
        "fine_pv_per_porosity * poro_new; zero where not valid",
        "fine_permz_multiplier": "PERMZ / PERMX of the fine INIT file; "
        "permz_fine = permx_fine * fine_permz_multiplier; NaN where not valid",
    }


def _describe_pore_volume_correction(dck: ConfigViaDeck) -> dict[str, Any]:
    """Describe the pore-volume correction applied in this run."""
    enabled = dck.pore_volume_correction == 1
    return {
        "mode": dck.pore_volume_correction,
        "applied": enabled,
        "algorithm": (
            "redistribute_removed_pore_volume: every inactive coarse cell whose "
            "cluster contains active fine cells (cluster max ACTNUM > selected "
            "ACTNUM) or that is removed by -j sends its raw pore volume in equal "
            "parts to the active cells found by the nearest-neighbour search"
            if enabled
            else "none: pore volume of inactive coarse cells is discarded"
        ),
        "source_debited_in_output_array": False,
        "transfer_convention": "final = raw + bincount(dst, amount); sources keep "
        "their raw value in coarse_pore_volume_final and are removed from the "
        "model by coarse ACTNUM = 0",
        "routing_depends_on": "coarse dimensions, coarse_actnum_pycopm and the "
        "fine ACTNUM; not on porosity values, provided the fine ACTNUM of a "
        "realization is unchanged",
        "conserves_total_pore_volume": enabled,
    }


def _describe_fluid_in_place_correction(dck: ConfigViaDeck) -> dict[str, Any]:
    """Describe the fluid-in-place correction (-q 1) of ``_correct_fluid_in_place``."""
    return {
        "applied": dck.correct_fluid_in_place == 1,
        "static_operator": False,
        "algorithm": [
            "run OPM Flow for 0.01 days on the fine deck (_1STEP) and the coarse "
            "deck (_CORR); read FIPOIL, FIPGAS (and SGAS) of report step 0",
            "P = PORV of the coarse _CORR INIT (float32, zero where OPM "
            "deactivated cells, e.g., by MINPV)",
            "oil step: c1 = sum(FIPOIL_fine) / sum(FIPOIL_coarse) - 1; "
            "H = FIPOIL_coarse > 0.1, L = FIPOIL_coarse <= 0.1; "
            "P[L] -= c1 * sum(P[H]) / |L|; P[H] *= 1 + c1; rerun _CORR",
            "gas step: G = SGAS_coarse > 0.1, L = FIPOIL_coarse <= 0.1 (new run); "
            "c2 = (sum(FIPGAS_fine) - sum(FIPGAS_coarse)) / sum(FIPGAS_coarse[G]); "
            "P[L] -= c2 * sum(P[G]) / |L|; P[G] *= 1 + c2",
            "coarse_pore_volume_written = P",
        ],
        "why_not_static": "the scale factors c1 and c2 are ratios of fine and "
        "coarse fluid-in-place totals, and the cell sets H, L, and G are "
        "thresholds on coarse FIPOIL and SGAS, all obtained from OPM Flow "
        "equilibrations of the specific realization; the update also mixes a "
        "multiplicative and a uniform subtractive part and starts from the "
        "OPM-processed coarse PORV, so no fixed per-cell factor or linear "
        "operator maps coarse_pore_volume_final to coarse_pore_volume_written",
        "realization_dependent_inputs": [
            "fine FIPOIL and FIPGAS per cell after equilibration (fine PORV, "
            "hence fine PORO)",
            "coarse FIPOIL, FIPGAS and SGAS per cell after equilibration of the "
            "coarse deck (coarse PORV and upscaled SWATINIT, hence PORO)",
            "coarse cells deactivated by OPM Flow (MINPV on the coarse PORV)",
        ],
    }


def _describe_property_upscaling(
    dck: ConfigViaDeck, coarsening: CoarseningMaps
) -> dict[str, Any]:
    """Describe the aggregation rule applied to each continuous property.

    The selection mirrors the branches in :func:`coarsen_properties`. In the
    formulas, sums run over the fine cells ``f`` of a coarse cluster, ``A``
    denotes the fine active cells (``fine_actnum``), ``PV`` the fine pore
    volume, ``dz`` ``fine_cell_dz``, and ``V`` ``fine_cell_volume``. Fine
    values are the float32 values of the fine INIT file. Coarse values without
    a valid denominator become zero."""
    methods = list(dck.continuous_aggregation_method)
    physical = any(not method for method in methods)
    formulas = {
        "pv_weighted_mean": "sum_A(v*PV) / sum_all(PV)",
        "thickness_weighted_mean": "sum_A(v*dz) / sum_all(dz)",
        "thickness_weighted_harmonic_mean": "sum_A(dz) / sum_{A, v>0}(dz/v)",
        "pv_weighted_sum_over_bulk_volume": "sum_A(v*PV) / sum_all(V)",
        "min": "min_A(v)",
        "max": "max_A(v)",
        "mean": "sum_A(v) / count_A",
        "pvmean": "sum_A(v*PV) / sum_all(PV)",
    }
    rules: dict[str, Any] = {}
    for name in dck.props_keywords + dck.solution_keywords:
        if name in TRANSMISSIBILITY_KEYWORDS:
            rule = "transmissibility_specific"
        elif physical:
            if name in ("permx", "permy"):
                rule = "thickness_weighted_mean"
            elif name == "permz":
                rule = "thickness_weighted_harmonic_mean"
            elif name in dck.multipliers_keywords:
                rule = "multiplier_specific"
            elif name in PV_WEIGHTED_KEYWORDS or name in dck.solution_keywords:
                rule = "pv_weighted_mean"
            else:
                rule = "pv_weighted_sum_over_bulk_volume"
        else:
            rule = methods[0] if len(methods) == 1 else "per_coarse_layer"
        entry: dict[str, Any] = {"rule": rule, "formula": formulas.get(rule, "")}
        if len(methods) > 1 and name not in TRANSMISSIBILITY_KEYWORDS:
            entry["layer_methods"] = methods
        if (
            name in ("permx", "permy", "permz")
            and dck.transmissibility_coarsening_method > 0
        ):
            entry["zeroed_outside_coarse_well_cells"] = True
        if name.upper() in coarsening.compacted_permeability:
            entry["written_as"] = (
                "COPY PERMX PERMY"
                if name == "permy"
                else "COPY PERMX PERMZ followed by MULTIPLY with a constant factor"
            )
        rules[name] = entry
    return {
        "continuous_aggregation_method": methods,
        "physical_aggregation": physical,
        "fine_values_source": "fine INIT file (float32), active cells only",
        "written_significant_digits": dck.significant_digits,
        "properties": rules,
    }
