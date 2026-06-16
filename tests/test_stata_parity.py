"""Real-Stata parity tests for the paper replication do-file."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stacked_eventstudy import estimate_stacked_eventstudy

DEFAULT_STATA_EXE = "/mnt/c/Program Files/StataNow19/StataMP-64.exe"
STATA_EXE = Path(os.environ.get("STATA_EXE", DEFAULT_STATA_EXE))
ESTIMATION_DO = (
    Path(__file__).resolve().parents[2]
    / "MelentyevaRiedel_StackedDiD"
    / "estimation.do"
)
STRICT_ATOL = 1e-8
ESTIMATE_ATOL = 1e-7
SE_ATOL = 1e-6
POST_TAIL_EVENT_TIME = 4
RESIDUAL_CENTER = 11
RESIDUAL_MODULUS = 23
RESIDUAL_SCALE = 0.05
MEMBERSHIP_COLUMNS = [
    "unit_id",
    "age",
    "treatment_age",
    "subevent",
    "event_time",
    "treated_in_subevent",
]


@pytest.mark.stata
def test_windows_stata_matches_python_full_stack_and_estimates(tmp_path: Path) -> None:
    """Compare Python outputs to a real Stata run of the paper's estimator."""
    _require_stata_dependencies(tmp_path=tmp_path)
    panel = _make_stata_validation_panel()
    stata_data_path = tmp_path / "stata_validation_input.dta"
    panel.to_stata(stata_data_path, write_index=False, version=118)

    do_path = tmp_path / "stata_validation.do"
    do_path.write_text(
        _make_validation_do_file(
            working_dir=_to_stata_path(tmp_path),
            dataset=stata_data_path.name,
            original_do=_to_stata_path(ESTIMATION_DO),
        ),
        encoding="utf-8",
    )
    _run_stata(do_path=do_path, cwd=tmp_path)
    _assert_stata_outputs_exist(
        output_paths=(
            tmp_path / "stata_stacked_membership.csv",
            tmp_path / "stata_cohort_weights.csv",
            tmp_path / "stata_cohort_params.csv",
            tmp_path / "stata_average_params.csv",
        ),
        log_path=tmp_path / "stata_validation.log",
    )

    python_result = estimate_stacked_eventstudy(
        data=panel.rename(
            columns={
                "persnr": "id",
                "agefirst": "treatment_age",
                "yrearn": "outcome",
            },
        ),
        id_col="id",
        age_col="age",
        treatment_age_col="treatment_age",
        outcome_col="outcome",
        l_min=-3,
        l_max=4,
        control_window=5,
        reference_event_time=-1,
        min_treatment_age=25,
        max_treatment_age=33,
        observed_min_age=22,
        backend="pyfixest",
        covariance_policy="stata",
        return_stacked_data=True,
    )
    assert python_result.stacked_data is not None

    _assert_stacked_membership_matches(
        stata_stack=pd.read_csv(tmp_path / "stata_stacked_membership.csv"),
        python_stack=python_result.stacked_data,
    )
    _assert_cohort_weights_match(
        stata_weights=pd.read_csv(tmp_path / "stata_cohort_weights.csv"),
        python_weights=python_result.cohort_weights,
    )
    _assert_cohort_estimates_match(
        stata_params=pd.read_csv(tmp_path / "stata_cohort_params.csv"),
        python_params=python_result.cohort_params,
    )
    _assert_average_estimates_match(
        stata_averages=pd.read_csv(tmp_path / "stata_average_params.csv"),
        python_averages=python_result.average_params,
    )


def _require_stata_dependencies(tmp_path: Path) -> None:
    """Fail clearly when Stata or required ado packages are unavailable."""
    if not STATA_EXE.exists():
        pytest.fail(
            "Windows Stata executable not found. "
            "Set STATA_EXE to the Stata executable.",
        )
    check_do = tmp_path / "stata_dependency_check.do"
    check_do.write_text(
        "which reghdfe\nwhich ftools\nwhich require\nexit, clear\n",
        encoding="utf-8",
    )
    result = _run_stata(do_path=check_do, cwd=tmp_path, check=False)
    log_text = (tmp_path / "stata_dependency_check.log").read_text(encoding="utf-8")
    missing_dependency = result.returncode != 0 or any(
        message in log_text
        for message in (
            "command reghdfe not found",
            "command ftools not found",
            "command require not found",
        )
    )
    if missing_dependency:
        pytest.fail(
            "Windows Stata is available, but required Stata ado packages are not. "
            "Install `reghdfe`, `ftools`, and `require` in Windows Stata before "
            "running `pytest -m stata`.",
        )


def _run_stata(
    do_path: Path,
    cwd: Path,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a Stata do-file in batch mode through WSL interop."""
    result = subprocess.run(  # noqa: S603
        [str(STATA_EXE), "/e", "do", _to_stata_path(do_path)],
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    if check and result.returncode != 0:
        log_path = cwd / f"{do_path.stem}.log"
        log_tail = ""
        if log_path.exists():
            log_tail = "\n".join(
                log_path.read_text(encoding="utf-8").splitlines()[-80:],
            )
        pytest.fail(
            f"Stata failed with exit code {result.returncode} while running {do_path}."
            f"\n{log_tail}",
        )
    return result


def _assert_stata_outputs_exist(
    output_paths: tuple[Path, ...],
    log_path: Path,
) -> None:
    """Fail with Stata log context when expected outputs were not written."""
    missing_outputs = [path.name for path in output_paths if not path.exists()]
    if not missing_outputs:
        return
    log_tail = ""
    if log_path.exists():
        log_tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-100:])
    pytest.fail(
        "Stata completed without writing expected validation outputs: "
        + ", ".join(missing_outputs)
        + f"\n{log_tail}",
    )


def _to_stata_path(path: Path) -> str:
    """Convert a WSL path to a Stata-friendly Windows path."""
    return subprocess.run(  # noqa: S603
        ["/usr/bin/wslpath", "-m", str(path)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _make_stata_validation_panel() -> pd.DataFrame:
    """Create a balanced synthetic panel compatible with the original do-file."""
    rows: list[dict[str, float | int]] = []
    unit_id = 1
    dynamic_effects = {
        -3: 0.0,
        -2: 0.0,
        -1: 0.0,
        0: -4.0,
        1: -6.0,
        2: -7.5,
        3: -8.0,
        POST_TAIL_EVENT_TIME: -8.5,
    }
    for agefirst in range(25, 39):
        for replicate in range(3):
            unit_fe = 0.3 * replicate
            for age in range(22, 43):
                event_time = age - agefirst
                treatment_effect = dynamic_effects.get(
                    event_time,
                    dynamic_effects[POST_TAIL_EVENT_TIME]
                    if event_time > POST_TAIL_EVENT_TIME
                    else 0.0,
                )
                residual = (
                    ((unit_id * 17 + age * 13) % RESIDUAL_MODULUS) - RESIDUAL_CENTER
                ) * RESIDUAL_SCALE
                rows.append(
                    {
                        "persnr": unit_id,
                        "age": age,
                        "agefirst": agefirst,
                        "time": event_time,
                        "yrearn": (
                            20.0 + unit_fe + 0.8 * age + treatment_effect + residual
                        ),
                    },
                )
            unit_id += 1
    return pd.DataFrame(rows)


def _make_validation_do_file(
    working_dir: str,
    dataset: str,
    original_do: str,
) -> str:
    """Return an instrumented Stata do-file that mirrors `estimation.do`."""
    return f"""
capture log close
clear all
version 17
set more off
set dp period
set linesize 80

local working_dir "{working_dir}"
local dataset "{dataset}"
local original_do "{original_do}"

local Time      time
local Pid       persnr
local Earnings  yrearn
local Age       age
local Agefirst  agefirst
local cohorts_next = 5
local periods_pre = 3
local periods_post = `cohorts_next' - 1
local maxagefirst = 38

capture program drop get_stacked_data
program define get_stacked_data
    syntax anything ///
        , agebirth(varname numeric) ///
        age(varname numeric) ///
        time(varname numeric) ///
        individual(varname numeric) ///
        periods_pre(integer) ///
        cohorts_next(integer)

    tokenize `anything'
    local sample `1'
    local cohort = `2'

    quietly {{
        gen byte `sample' = `cohort' if `agebirth' == `cohort'
        replace `sample' = `cohort' ///
            if inrange(`agebirth', `cohort'+1, `cohort'+`cohorts_next') ///
            & `time' < 0
        keep if `sample' == `cohort'
        replace `time' = . ///
            if inrange(`agebirth', `cohort'+1, `cohort'+`cohorts_next')
        keep if inrange(`age', `cohort'-`periods_pre', `cohort'+`cohorts_next'-1)

        tempvar obs obs_test
        sort `individual'
        by `individual': egen `obs' = count(`time') if !missing(`time')
        quietly sum `obs'
        local maxobs = r(max)
        drop if `obs' < `maxobs' & `obs' != 0 & !missing(`time')
        sort `individual'
        by `individual': egen `obs_test' = count(`time')
        assert `obs_test' == `maxobs' if !missing(`time')
        assert `obs_test' <= `maxobs' if missing(`time')
        drop `obs' `obs_test'
    }}
end

tempfile stackeddata
quietly save `stackeddata', emptyok

cd "`working_dir'"
log using "`working_dir'/stata_validation_inner.log", text replace
di "Original Stata do-file: `original_do'"
which reghdfe

use `Time' `Pid' `Earnings' `Age' `Agefirst' ///
    using "`working_dir'/`dataset'", clear
foreach var of varlist _all {{
    assert !missing(`var')
}}

keep if inrange(`Age', 20, .)
quietly sum `Age'
local cohort_first = `r(min)' + `periods_pre'
local cohort_last = `maxagefirst' - `cohorts_next'

quietly forvalues c = `cohort_first'/`cohort_last' {{
    preserve
    get_stacked_data subevent `c' ///
        , agebirth(`Agefirst') ///
        age(`Age') ///
        time(`Time') ///
        individual(`Pid') ///
        periods_pre(`periods_pre') ///
        cohorts_next(`cohorts_next')
    append using `stackeddata'
    save `"`stackeddata'"', replace
    restore
}}

use `stackeddata', clear
isid `Pid' `Age' subevent
assert inrange(subevent, `cohort_first', `cohort_last')

gen int event_time = `Age' - subevent
gen byte treated_in_subevent = !missing(`Time')
preserve
keep `Pid' `Age' `Agefirst' subevent event_time treated_in_subevent
rename `Pid' unit_id
rename `Age' age
rename `Agefirst' treatment_age
sort subevent unit_id age
export delimited using "`working_dir'/stata_stacked_membership.csv", replace
restore

quietly forvalues c = `cohort_first'/`cohort_last' {{
    forvalues t = 2/`periods_pre' {{
        gen byte time_pre`t'_cohort`c' = (`Time' == -`t' & subevent == `c')
        replace time_pre`t'_cohort`c' = 0 if `Time' == . & subevent == `c'
    }}
    capture forvalues t = 0/`periods_post' {{
        gen byte time_post`t'_cohort`c' = (`Time' == `t' & subevent == `c')
        replace time_post`t'_cohort`c' = 0 if `Time' == . & subevent == `c'
    }}
}}

reghdfe `Earnings' time_pre*_cohort* time_post*_cohort* ///
    , nocons vce(cluster `Pid') ///
    absorb(i.`Pid'#i.subevent i.`Age'#i.subevent, savefe)

tempname weight_handle
postfile `weight_handle' int subevent double weight ///
    using "`working_dir'/stata_cohort_weights.dta", replace
quietly forvalues c = `cohort_first'/`cohort_last' {{
    tempvar n w
    gen byte `n' = (`Agefirst' == `c') if e(sample) & !missing(`Time')
    egen `w' = mean(`n') if e(sample) ///
        & inrange(`Agefirst', `cohort_first', `cohort_last') ///
        & !missing(`Time')
    sum `w' if !missing(`Time')
    scalar weight_`c' = r(mean)
    post `weight_handle' (`c') (r(mean))
}}
postclose `weight_handle'
preserve
use "`working_dir'/stata_cohort_weights.dta", clear
format weight %21.15g
export delimited using "`working_dir'/stata_cohort_weights.csv", replace
restore

tempname cohort_handle average_handle
postfile `cohort_handle' int subevent int event_time double estimate std_error ///
    using "`working_dir'/stata_cohort_params.dta", replace
postfile `average_handle' int event_time double estimate std_error ///
    using "`working_dir'/stata_average_params.dta", replace

quietly forvalues c = `cohort_first'/`cohort_last' {{
    forvalues j = -`periods_pre'/`periods_post' {{
        if (`j' != -1) {{
            local suffix pre
            if (`j' >= 0) local suffix post
            local t = abs(`j')
            local coefname time_`suffix'`t'_cohort`c'
            post `cohort_handle' (`c') (`j') (_b[`coefname']) (_se[`coefname'])
        }}
    }}
}}

forvalues j = -`periods_pre'/`periods_post' {{
    if (`j' == -1) continue
    local suffix pre
    if (`j' >= 0) local suffix post
    local t = abs(`j')
    local lincom_exp 0
    forvalues c = `cohort_first'/`cohort_last' {{
        local lincom_exp `lincom_exp' + time_`suffix'`t'_cohort`c' * weight_`c'
    }}
    quietly lincom `lincom_exp'
    post `average_handle' (`j') (r(estimate)) (r(se))
}}
postclose `cohort_handle'
postclose `average_handle'

preserve
use "`working_dir'/stata_cohort_params.dta", clear
format estimate std_error %21.15g
export delimited using "`working_dir'/stata_cohort_params.csv", replace
restore

preserve
use "`working_dir'/stata_average_params.dta", clear
format estimate std_error %21.15g
export delimited using "`working_dir'/stata_average_params.csv", replace
restore

clear
log close
exit, clear
"""


def _assert_stacked_membership_matches(
    stata_stack: pd.DataFrame,
    python_stack: pd.DataFrame,
) -> None:
    """Assert identical stacked row membership."""
    python_membership = (
        python_stack.loc[
            :,
            MEMBERSHIP_COLUMNS,
        ]
        .sort_values(["subevent", "unit_id", "age"])
        .reset_index(drop=True)
    )
    stata_membership = (
        stata_stack.loc[
            :,
            MEMBERSHIP_COLUMNS,
        ]
        .sort_values(["subevent", "unit_id", "age"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        python_membership.astype(stata_membership.dtypes.to_dict()),
        stata_membership,
        check_dtype=False,
    )


def _assert_cohort_weights_match(
    stata_weights: pd.DataFrame,
    python_weights: pd.DataFrame,
) -> None:
    """Assert cohort weights match Stata weights."""
    comparison = python_weights.merge(
        stata_weights,
        on="subevent",
        how="inner",
        validate="one_to_one",
        suffixes=("_python", "_stata"),
    )
    assert comparison.shape[0] == python_weights.shape[0]
    assert np.allclose(
        comparison["weight_python"],
        comparison["weight_stata"],
        atol=STRICT_ATOL,
        rtol=0.0,
    )


def _assert_cohort_estimates_match(
    stata_params: pd.DataFrame,
    python_params: pd.DataFrame,
) -> None:
    """Assert cohort-level estimates and standard errors match Stata."""
    comparison = python_params.merge(
        stata_params,
        on=["subevent", "event_time"],
        how="inner",
        validate="one_to_one",
        suffixes=("_python", "_stata"),
    )
    assert comparison.shape[0] == python_params.shape[0]
    assert np.allclose(
        comparison["estimate_python"],
        comparison["estimate_stata"],
        atol=ESTIMATE_ATOL,
        rtol=0.0,
    )
    assert np.allclose(
        comparison["std_error_python"],
        comparison["std_error_stata"],
        atol=SE_ATOL,
        rtol=0.0,
    )


def _assert_average_estimates_match(
    stata_averages: pd.DataFrame,
    python_averages: pd.DataFrame,
) -> None:
    """Assert aggregated estimates and standard errors match Stata."""
    comparison = python_averages.merge(
        stata_averages,
        on="event_time",
        how="inner",
        validate="one_to_one",
        suffixes=("_python", "_stata"),
    )
    assert comparison.shape[0] == python_averages.shape[0]
    assert np.allclose(
        comparison["estimate_python"],
        comparison["estimate_stata"],
        atol=ESTIMATE_ATOL,
        rtol=0.0,
    )
    assert np.allclose(
        comparison["std_error_python"],
        comparison["std_error_stata"],
        atol=SE_ATOL,
        rtol=0.0,
    )
