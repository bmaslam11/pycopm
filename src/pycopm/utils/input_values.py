# SPDX-FileCopyrightText: 2024-2026 NORCE Research AS
# SPDX-License-Identifier: GPL-3.0
# pylint: disable=R0912,R0913,R0914,R0915,R0917,R1702
"""Create configuration objects from command-line arguments and TOML files."""

import argparse
import datetime as dt
import math
import tomllib
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np
from numpy.typing import NDArray
from opm.io.ecl import EGrid as OpmGrid

from pycopm.config.config import ConfigViaDeck, ConfigViaTOML
from pycopm.utils.terminal import (
    cli_correct_value,
    cli_error_value,
    cli_warning_value,
    pycopm_error,
    pycopm_warning,
)

TOML_KEYS = {
    "flow_command",
    "model_name",
    "execution_mode",
    "ensemble_size",
    "max_parallel_realizations",
    "max_realization_runtime_seconds",
    "min_successful_realizations",
    "random_seed",
    "saturation_function_method",
    "pore_volume_correction",
    "initialization_method",
    "observation_relative_errors",
    "observation_minimum_errors",
    "history_matching_end_date",
    "ert_arguments",
    "let_parameters",
    "rock_property_settings",
    "x_coarsening",
    "y_coarsening",
    "z_coarsening",
    "satnum_generation_method",
    "cleanup_file_suffixes",
}

INTERNAL_KEYS = {
    "output_directory",
    "resource_directory",
    "reference_case_name",
    "use_let_tables",
    "significant_digits",
    "original_nx",
    "original_ny",
    "original_nz",
    "output_nx",
    "output_ny",
    "output_nz",
    "original_cell_count",
    "original_to_output_i",
    "original_to_output_j",
    "original_to_output_k",
}

MODEL_AXIS_LENGTHS = {
    "norne": {"x_coarsening": 47, "y_coarsening": 113, "z_coarsening": 23},
    "drogon": {"x_coarsening": 47, "y_coarsening": 74, "z_coarsening": 32},
}

MODEL_LET_NAMES = {
    "norne": [
        "lw",
        "ew",
        "tw",
        "lo",
        "eo",
        "to",
        "lg",
        "eg",
        "tg",
        "log",
        "eog",
        "tog",
        "lmlto",
        "emlto",
        "tmlto",
        "lmltg",
        "emltg",
        "tmltg",
    ],
    "drogon": [
        "lw",
        "ew",
        "tw",
        "lo",
        "eo",
        "to",
        "lg",
        "eg",
        "tg",
        "log",
        "eog",
        "tog",
        "lmlto",
        "lmltg",
        "kwow",
        "kwoo",
        "kwgw",
        "kwgg",
    ],
}

ROCK_PROPERTY_NAMES = {"PERMX", "PERMY", "PERMZ"}


def create_deck_config(cmdargs: argparse.Namespace) -> ConfigViaDeck:
    """Create a deck configuration from parsed command arguments.

    Parameters
    ----------
    cmdargs : argparse.Namespace
        Command-line arguments for the deck-based workflow.

    Returns
    -------
    ConfigViaDeck
        Configuration populated from the command-line values."""
    return ConfigViaDeck(
        output_directory=str(Path(cmdargs.output_directory).expanduser().resolve()),
        flow_command=cmdargs.flow_command,
        input_deck_name=Path(cmdargs.input_deck_path).stem,
        input_deck_path=str(Path(cmdargs.input_deck_path).with_suffix("")),
        active_cell_methods=cmdargs.active_cell_methods.split(","),
        discrete_aggregation_method=cmdargs.discrete_aggregation_method.split(","),
        continuous_aggregation_method=cmdargs.continuous_aggregation_method.split(","),
        jump_thresholds=cmdargs.jump_thresholds.split(","),
        output_deck_name=cmdargs.output_deck_name,
        execution_mode=cmdargs.execution_mode,
        include_prefix=cmdargs.include_prefix,
        requested_ijk=[cmdargs.requested_ijk],
        completion_removal_level=int(cmdargs.completion_removal_level),
        deck_encoding=cmdargs.deck_encoding,
        pore_volume_correction=int(cmdargs.pore_volume_correction),
        correct_fluid_in_place=int(cmdargs.correct_fluid_in_place),
        transmissibility_coarsening_method=int(
            cmdargs.transmissibility_coarsening_method
        ),
        vicinity_specification=cmdargs.vicinity_specification,
        grid_transformation=cmdargs.grid_transformation,
        write_explicit_solution=int(cmdargs.write_explicit_solution) == 1,
        dual_porosity_criterion=cmdargs.dual_porosity_criterion,
        significant_digits=int(cmdargs.significant_digits),
        refinement_enabled=bool(
            cmdargs.x_refinement
            or cmdargs.y_refinement
            or cmdargs.z_refinement
            or cmdargs.refinement
        ),
        coarsening_enabled=bool(
            cmdargs.x_coarsening
            or cmdargs.y_coarsening
            or cmdargs.z_coarsening
            or cmdargs.coarsening
        ),
        coarsening_artifact=(
            str(Path(cmdargs.coarsening_artifact).expanduser().resolve())
            if getattr(cmdargs, "coarsening_artifact", "")
            else ""
        ),
    )


def parse_axis_modifications(uniform: str, localized: list) -> tuple[NDArray, list]:
    """Parse uniform or axis-specific grid modifications.

    Uniform input contains one value for each axis. Axis-specific coarsening
    also accepts one-based indices and inclusive ranges such as ``2:4,7``.

    Parameters
    ----------
    uniform : str
        Comma-separated x, y, and z modification values.
    localized : list
        Axis-specific specifications in x, y, and z order.

    Returns
    -------
    cijk, axis_values
        Uniform axis values and the three parsed axis-specific arrays. Only one
        representation is populated."""
    if uniform:
        cijk = np.fromstring(uniform, sep=",", dtype=int)
        refs: list = [[], [], []]
    else:
        cijk = np.array([])
        refs = []
        for i in range(3):
            argument = localized[i]
            if argument:
                if ":" in argument:
                    values = [0]
                    index = 1
                    for value in argument.split(","):
                        entry = value.split(":")
                        start_index = int(entry[0])
                        values.extend([0] * max(0, start_index - index))
                        if len(entry) == 2:
                            end_index = int(entry[1])
                            values.extend([2] * max(0, end_index - start_index))
                            index = end_index
                        else:
                            index = start_index
                    values.append(0)
                    refs.append(values)
                else:
                    refs.append(
                        list(np.fromstring(argument, sep=",", dtype=int).tolist())
                    )
            else:
                refs.append([])
    return cijk, refs


def _is_finite_number(value: Any) -> TypeGuard[int | float]:
    """Check whether a value is a finite non-Boolean number.

    Parameters
    ----------
    value : Any
        Value to inspect.

    Returns
    -------
    bool
        Whether the value is a finite integer or floating-point number."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_integer(value: Any) -> TypeGuard[int]:
    """Check whether a value is a non-Boolean integer.

    Parameters
    ----------
    value : Any
        Value to inspect.

    Returns
    -------
    bool
        Whether the value is an integer and not a Boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _add_validation_error(errors: list[str], message: str) -> None:
    """Add a TOML validation error.

    Parameters
    ----------
    errors : list[str]
        Validation messages collected during the current validation pass.
    message : str
        Human-readable validation message."""
    errors.append(message)


def _warn(message: str) -> None:
    """Display a TOML validation warning.

    Parameters
    ----------
    message : str
        Human-readable warning passed to the shared terminal helper."""
    pycopm_warning(message)


def _validate_string(cfg_file: dict[str, Any], key: str, errors: list[str]) -> bool:
    """Check that a TOML variable is a non-empty string.

    Parameters
    ----------
    cfg_file : dict[str, Any]
        TOML configuration values.
    key : str
        Configuration variable name.
    errors : list[str]
        Validation messages collected during the current validation pass.

    Returns
    -------
    bool
        Whether the variable is present and contains a non-empty string."""
    if key not in cfg_file:
        return False
    value = cfg_file[key]
    if not isinstance(value, str) or not value.strip():
        _add_validation_error(
            errors,
            f"variable {cli_error_value(key)} has invalid value "
            f"{cli_error_value(str(value))}, expected "
            f"{cli_correct_value('a non-empty string')}.",
        )
        return False
    return True


def _validate_integer(
    cfg_file: dict[str, Any], key: str, errors: list[str], minimum: int = 0
) -> bool:
    """Check that a TOML variable is an integer within its lower bound.

    Parameters
    ----------
    cfg_file : dict[str, Any]
        TOML configuration values.
    key : str
        Configuration variable name.
    errors : list[str]
        Validation messages collected during the current validation pass.
    minimum : int, optional
        Inclusive lower bound for accepted values.

    Returns
    -------
    bool
        Whether the variable is present and satisfies the integer constraint."""
    if key not in cfg_file:
        return False
    value = cfg_file[key]
    if not _is_integer(value) or value < minimum:
        _add_validation_error(
            errors,
            f"variable {cli_error_value(key)} has invalid value "
            f"{cli_error_value(str(value))}, expected an integer "
            f"greater than or equal to {minimum}.",
        )
        return False
    return True


def _validate_number_array(
    cfg_file: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    length: int | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> bool:
    """Check the shape and values of a numeric TOML array.

    Parameters
    ----------
    cfg_file : dict[str, Any]
        TOML configuration values.
    key : str
        Configuration variable name.
    errors : list[str]
        Validation messages collected during the current validation pass.
    length : int | None, optional
        Required number of entries.
    minimum : float | None, optional
        Inclusive lower bound for every entry.
    maximum : float | None, optional
        Inclusive upper bound for every entry.

    Returns
    -------
    bool
        Whether the variable is present and satisfies all array constraints."""
    if key not in cfg_file:
        return False
    value = cfg_file[key]
    if not isinstance(value, list):
        _add_validation_error(errors, f"variable '{key}' must be an array.")
        return False
    valid = True
    if length is not None and len(value) != length:
        _add_validation_error(
            errors, f"variable '{key}' has {len(value)} entries, expected {length}."
        )
        valid = False
    for index, entry in enumerate(value):
        if not _is_finite_number(entry):
            _add_validation_error(
                errors,
                f"variable {cli_error_value(f'{key}[{index}]')} has invalid value "
                f"{cli_error_value(str(entry))}, expected "
                f"{cli_correct_value('a finite number')}.",
            )
            valid = False
        elif minimum is not None and entry < minimum:
            _add_validation_error(
                errors,
                f"variable {cli_error_value(f'{key}[{index}]')} has invalid value "
                f"{cli_error_value(str(entry))}, expected a value greater than or "
                f"equal to {cli_correct_value(str(minimum))}.",
            )
            valid = False
        elif maximum is not None and entry > maximum:
            _add_validation_error(
                errors,
                f"variable {cli_error_value(f'{key}[{index}]')} has invalid value "
                f"{cli_error_value(str(entry))}, expected a value less than or "
                f"equal to {cli_correct_value(str(maximum))}.",
            )
            valid = False
    return valid


def _validate_coarsening(
    cfg_file: dict[str, Any], key: str, expected_length: int, errors: list[str]
) -> None:
    """Validate one model-specific axis coarsening array.

    The array must contain non-negative integers, match the number of grid
    boundaries for the selected model, and retain both outer boundaries.

    Parameters
    ----------
    cfg_file : dict[str, Any]
        TOML configuration values.
    key : str
        Name of the x, y, or z coarsening variable.
    expected_length : int
        Required number of entries for the selected reference model.
    errors : list[str]
        Validation messages collected during the current validation pass."""
    if key not in cfg_file:
        return
    values = cfg_file[key]
    if not isinstance(values, list):
        _add_validation_error(errors, f"variable '{key}' must be an array.")
        return
    if len(values) != expected_length:
        _add_validation_error(
            errors,
            f"variable '{key}' has {len(values)} entries, expected {expected_length}.",
        )
    for index, value in enumerate(values):
        if not _is_integer(value) or value < 0:
            _add_validation_error(
                errors,
                f"variable {cli_error_value(f'{key}[{index}]')} has invalid value "
                f"{cli_error_value(str(value))}, "
                "expected a non-negative integer.",
            )
    if values and values[0] != 0:
        _add_validation_error(
            errors, f"variable '{key}[0]' must be 0 to retain the first grid boundary."
        )
    if values and values[-1] != 0:
        _add_validation_error(
            errors, f"variable '{key}[-1]' must be 0 to retain the last grid boundary."
        )


def _validate_let_parameters(
    cfg_file: dict[str, Any], model: str | None, errors: list[str]
) -> None:
    """Validate the ordered LET-parameter definitions.

    Each row contains a coefficient name, initial value, estimation flag,
    distribution name, lower bound, and upper bound. The order is checked because
    downstream table generation addresses coefficients by position.

    Parameters
    ----------
    cfg_file : dict[str, Any]
        TOML configuration values.
    model : str | None
        Normalized reference-model name, when valid.
    errors : list[str]
        Validation messages collected during the current validation pass."""
    if "let_parameters" not in cfg_file:
        return
    rows = cfg_file["let_parameters"]
    if not isinstance(rows, list):
        _add_validation_error(
            errors, "variable 'let_parameters' must be an array of arrays."
        )
        return
    expected_names = MODEL_LET_NAMES.get(model) if model is not None else None
    if expected_names and len(rows) != len(expected_names):
        _add_validation_error(
            errors,
            f"variable 'let_parameters' has {len(rows)} rows, expected "
            f"{len(expected_names)} for model '{model}'.",
        )
    seen: set[str] = set()
    for index, row in enumerate(rows):
        name = f"let_parameters[{index}]"
        if not isinstance(row, list) or len(row) != 6:
            size = len(row) if isinstance(row, list) else type(row).__name__
            _add_validation_error(
                errors,
                f"variable {cli_error_value(name)} has invalid shape "
                f"{cli_error_value(str(size))}, expected "
                f"{cli_correct_value('6 entries')}.",
            )
            continue
        coefficient, initial, estimated, distribution, lower, upper = row
        if not isinstance(coefficient, str) or not coefficient.strip():
            _add_validation_error(
                errors, f"variable '{name}[0]' must be a non-empty coefficient name."
            )
        elif coefficient in seen:
            _add_validation_error(
                errors, f"duplicate LET-parameter name '{coefficient}'."
            )
        else:
            seen.add(coefficient)
        if (
            expected_names
            and index < len(expected_names)
            and coefficient != expected_names[index]
        ):
            _add_validation_error(
                errors,
                f"variable {cli_error_value(f'{name}[0]')} has invalid value "
                f"{cli_error_value(str(coefficient))}, expected "
                f"{cli_correct_value(expected_names[index])} at this position.",
            )
        if not _is_finite_number(initial):
            _add_validation_error(
                errors, f"variable '{name}[1]' must be a finite number."
            )
        if not _is_integer(estimated) or estimated not in {0, 1}:
            _add_validation_error(errors, f"variable '{name}[2]' must be 0 or 1.")
        if not isinstance(distribution, str) or not distribution.strip():
            _add_validation_error(
                errors, f"variable '{name}[3]' must be a non-empty distribution name."
            )
        if not _is_finite_number(lower) or not _is_finite_number(upper):
            _add_validation_error(
                errors, f"variables '{name}[4:6]' must be finite numbers."
            )
        elif lower >= upper:
            _add_validation_error(
                errors,
                f"variable {cli_error_value(name)} has invalid uniform bounds "
                f"{cli_error_value(str([lower, upper]))}, expected the lower "
                f"bound to be less than {cli_correct_value('the upper bound')}.",
            )
        elif (
            isinstance(distribution, str)
            and distribution.lower() == "uniform"
            and _is_finite_number(initial)
            and not lower <= initial <= upper
        ):
            _add_validation_error(
                errors,
                f"variable {cli_error_value(f'{name}[1]')} has invalid value "
                f"{cli_error_value(str(initial))}, expected a value between "
                f"{cli_correct_value(str(lower))} and "
                f"{cli_correct_value(str(upper))} for the uniform distribution.",
            )


def _validate_rock_properties(cfg_file: dict[str, Any], errors: list[str]) -> None:
    """Validate rock-property history matching settings.

    Each row contains a permeability name, estimation flag, and aggregation
    method. Property names are normalized to uppercase after validation.

    Parameters
    ----------
    cfg_file : dict[str, Any]
        TOML configuration values.
    errors : list[str]
        Validation messages collected during the current validation pass."""
    if "rock_property_settings" not in cfg_file:
        return
    rows = cfg_file["rock_property_settings"]
    if not isinstance(rows, list):
        _add_validation_error(
            errors, "variable 'rock_property_settings' must be an array of arrays."
        )
        return
    seen: set[str] = set()
    for index, row in enumerate(rows):
        name = f"rock_property_settings[{index}]"
        if not isinstance(row, list) or len(row) != 3:
            _add_validation_error(
                errors,
                f"variable '{name}' must contain name, estimation flag, and aggregation method.",
            )
            continue
        prop, estimated, method = row
        if not isinstance(prop, str) or prop.upper() not in ROCK_PROPERTY_NAMES:
            _add_validation_error(
                errors,
                f"variable {cli_error_value(f'{name}[0]')} has invalid value "
                f"{cli_error_value(str(prop))}, expected "
                "PERMX, PERMY, or PERMZ.",
            )
        elif prop.upper() in seen:
            _add_validation_error(errors, f"duplicate rock property '{prop.upper()}'.")
        else:
            seen.add(prop.upper())
            row[0] = prop.upper()
        if not _is_integer(estimated) or estimated not in {0, 1}:
            _add_validation_error(errors, f"variable '{name}[1]' must be 0 or 1.")
        if method not in {"max", "mean"}:
            _add_validation_error(
                errors,
                f"variable {cli_error_value(f'{name}[2]')} has invalid value "
                f"{cli_error_value(str(method))}, expected "
                f"{cli_correct_value('max')} or {cli_correct_value('mean')}.",
            )
    missing = ROCK_PROPERTY_NAMES - seen
    if missing:
        _add_validation_error(
            errors, f"missing rock property settings for {', '.join(sorted(missing))}."
        )


def _validate_toml(cfg_file: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize TOML configuration values.

    Unknown and internally managed variables are reported and removed. Remaining
    errors are collected so the user receives one complete validation report.

    Parameters
    ----------
    cfg_file : dict[str, Any]
        Raw values loaded from the TOML configuration.

    Returns
    -------
    dict[str, Any]
        Validated and normalized values suitable for ``ConfigViaTOML``.

    Raises
    ------
    SystemExit
        If one or more configuration values are invalid."""
    if not isinstance(cfg_file, dict):
        pycopm_error(
            f"invalid TOML content {cli_error_value(type(cfg_file).__name__)}, "
            f"expected {cli_correct_value('a dictionary of configuration variables')}."
        )
    cfg_file = cfg_file.copy()
    errors: list[str] = []

    internal = sorted(INTERNAL_KEYS & cfg_file.keys())
    if internal:
        formatted = ", ".join(cli_warning_value(key) for key in internal)
        plural = len(internal) != 1
        pycopm_warning(
            f"variable{'s' if plural else ''} {formatted} "
            f"{'are' if plural else 'is'} managed internally and will be ignored."
        )
        for key in internal:
            cfg_file.pop(key)
    unknown = sorted(cfg_file.keys() - TOML_KEYS)
    if unknown:
        formatted = ", ".join(cli_warning_value(key) for key in unknown)
        plural = len(unknown) != 1
        pycopm_warning(
            f"unknown TOML variable{'s' if plural else ''} {formatted} will be ignored."
        )
        for key in unknown:
            cfg_file.pop(key)

    required = set(TOML_KEYS) - {"satnum_generation_method"}
    for key in sorted(required - cfg_file.keys()):
        _add_validation_error(errors, f"missing required TOML variable '{key}'.")

    _validate_string(cfg_file, "flow_command", errors)
    if _validate_string(cfg_file, "model_name", errors):
        model = cfg_file["model_name"].lower()
        cfg_file["model_name"] = model
        if model not in MODEL_AXIS_LENGTHS:
            _add_validation_error(
                errors,
                f"variable {cli_error_value('model_name')} has invalid value "
                f"{cli_error_value(str(model))}, expected "
                f"{cli_correct_value('norne')} or {cli_correct_value('drogon')}.",
            )
    else:
        model = None

    if _validate_string(cfg_file, "execution_mode", errors):
        mode = cfg_file["execution_mode"].lower()
        cfg_file["execution_mode"] = mode
        if mode not in {"single-run", "files", "ert"}:
            _add_validation_error(
                errors,
                f"variable {cli_error_value('execution_mode')} has invalid value "
                f"{cli_error_value(str(mode))}, "
                "expected 'single-run', 'files', or 'ert'.",
            )
    else:
        mode = None

    for key, minimum in (
        ("ensemble_size", 1),
        ("max_parallel_realizations", 1),
        ("max_realization_runtime_seconds", 0),
        ("min_successful_realizations", 1),
        ("random_seed", 0),
    ):
        _validate_integer(cfg_file, key, errors, minimum)

    ensemble = cfg_file.get("ensemble_size")
    parallel = cfg_file.get("max_parallel_realizations")
    successful = cfg_file.get("min_successful_realizations")
    if _is_integer(ensemble):
        if _is_integer(parallel) and parallel > ensemble:
            _add_validation_error(
                errors,
                "variable 'max_parallel_realizations' cannot exceed 'ensemble_size'.",
            )
        if _is_integer(successful) and successful > ensemble:
            _add_validation_error(
                errors,
                "variable 'min_successful_realizations' cannot exceed 'ensemble_size'.",
            )

    for key, choices in (
        ("saturation_function_method", {0, 1}),
        ("pore_volume_correction", {0, 1}),
        ("initialization_method", {0, 1}),
    ):
        if _validate_integer(cfg_file, key, errors) and cfg_file[key] not in choices:
            _add_validation_error(
                errors,
                f"variable {cli_error_value(key)} has invalid value "
                f"{cli_error_value(str(cfg_file[key]))}, "
                f"expected one of {sorted(choices)}.",
            )

    if "satnum_generation_method" in cfg_file:
        if _validate_integer(cfg_file, "satnum_generation_method", errors) and cfg_file[
            "satnum_generation_method"
        ] not in {0, 1, 2}:
            _add_validation_error(
                errors, "variable 'satnum_generation_method' must be 0, 1, or 2."
            )
        if model == "drogon":
            pycopm_warning(
                f"variable {cli_warning_value('satnum_generation_method')} is only "
                f"effective for {cli_correct_value('model_name = norne')} and will be ignored."
            )
            cfg_file.pop("satnum_generation_method", None)
    elif model == "norne":
        cfg_file["satnum_generation_method"] = 0

    _validate_number_array(
        cfg_file, "observation_relative_errors", errors, length=3, minimum=0, maximum=1
    )
    _validate_number_array(
        cfg_file, "observation_minimum_errors", errors, length=3, minimum=0
    )

    if "history_matching_end_date" in cfg_file:
        value = cfg_file["history_matching_end_date"]
        if isinstance(value, dt.datetime):
            cfg_file["history_matching_end_date"] = value.date().isoformat()
        elif isinstance(value, dt.date):
            cfg_file["history_matching_end_date"] = value.isoformat()
        elif isinstance(value, str):
            try:
                cfg_file["history_matching_end_date"] = dt.date.fromisoformat(
                    value
                ).isoformat()
            except ValueError:
                _add_validation_error(
                    errors,
                    "variable 'history_matching_end_date' must be an ISO date in "
                    "YYYY-MM-DD format.",
                )
        else:
            _add_validation_error(
                errors,
                "variable 'history_matching_end_date' must be a TOML date or ISO date string.",
            )

    _validate_string(cfg_file, "ert_arguments", errors)
    if mode != "ert" and "ert_arguments" in cfg_file:
        pycopm_warning(
            f"variable {cli_warning_value('ert_arguments')} is not executed for "
            f"{cli_warning_value(f'execution_mode = {mode}')}, but is retained for "
            "generated ERT files."
        )

    if model in MODEL_AXIS_LENGTHS:
        for key, length in MODEL_AXIS_LENGTHS[model].items():
            _validate_coarsening(cfg_file, key, length, errors)

    if "cleanup_file_suffixes" in cfg_file:
        suffixes = cfg_file["cleanup_file_suffixes"]
        if not isinstance(suffixes, list):
            _add_validation_error(
                errors, "variable 'cleanup_file_suffixes' must be an array of strings."
            )
        else:
            for index, suffix in enumerate(suffixes):
                if not isinstance(suffix, str) or not suffix.strip():
                    _add_validation_error(
                        errors,
                        f"variable 'cleanup_file_suffixes[{index}]' must be a non-empty string.",
                    )
                elif any(character in suffix for character in "'\"/*?[]"):
                    _add_validation_error(
                        errors,
                        f"variable 'cleanup_file_suffixes[{index}]' contains unsafe "
                        "filename-pattern characters.",
                    )

    _validate_let_parameters(cfg_file, model, errors)
    _validate_rock_properties(cfg_file, errors)

    if cfg_file.get("saturation_function_method") == 1 and not cfg_file.get(
        "let_parameters"
    ):
        _add_validation_error(
            errors, "LET saturation functions require a non-empty 'let_parameters'."
        )

    estimated_let = any(
        isinstance(row, list) and len(row) >= 3 and row[2] == 1
        for row in cfg_file.get("let_parameters", [])
    )
    estimated_rock = any(
        isinstance(row, list) and len(row) >= 2 and row[1] == 1
        for row in cfg_file.get("rock_property_settings", [])
    )
    if mode == "ert" and not (estimated_let or estimated_rock):
        _add_validation_error(
            errors,
            "execution_mode 'ert' requires at least one estimated LET or rock-property parameter.",
        )

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        pycopm_error(f"invalid TOML configuration:\n{details}")
    return cfg_file


def load_toml_config(
    input_file: str,
    output_directory: str,
    resource_directory: str,
    significant_digits: int,
) -> ConfigViaTOML:
    """Load, validate, and initialize a TOML configuration.

    Validation and normalization occur before the reference EGRID is opened and
    before ``ConfigViaTOML`` is constructed.

    Parameters
    ----------
    input_file : str
        TOML configuration path.
    output_directory : str
        Generated-project directory.
    resource_directory : str
        Directory containing templates and reference simulations.
    significant_digits : int
        Precision used when writing floating-point values.

    Returns
    -------
    ConfigViaTOML
        Validated configuration populated with reference-grid metadata.

    Raises
    ------
    SystemExit
        If the TOML configuration is invalid."""
    with open(input_file, "rb") as file_handle:
        cfg_file = _validate_toml(tomllib.load(file_handle))

    suffixes = cfg_file["cleanup_file_suffixes"]
    cfg_file["cleanup_file_suffixes"] = ",".join(f"'{suffix}'" for suffix in suffixes)
    for key in ("x_coarsening", "y_coarsening", "z_coarsening"):
        cfg_file[key] = np.asarray(cfg_file[key], dtype=int)

    name = "NORNE_ATW2013" if cfg_file["model_name"] == "norne" else "DROGON"
    case_path = (
        Path(resource_directory)
        / "reference_simulation"
        / cfg_file["model_name"]
        / name
    )
    grid = OpmGrid(f"{case_path}.EGRID")
    cfg = ConfigViaTOML(
        output_directory=output_directory,
        resource_directory=resource_directory,
        significant_digits=significant_digits,
        reference_case_name=name,
        original_nx=grid.dimension[0],
        original_ny=grid.dimension[1],
        original_nz=grid.dimension[2],
        original_cell_count=int(np.prod(grid.dimension)),
        **cfg_file,
    )
    return cfg
