# ============================================================
# NICMOS BINARY SEARCH PIPELINE
#
# Dataset
#     ↓
# Target
#     ↓
# Filter
#     ↓
# GRID → BFGS → HMC → DIAGNOSTIC PLOT
# ============================================================


from pathlib import Path
from collections import defaultdict
import pickle

import jax
import jax.numpy as np

from astropy.io import fits


# ============================================================
# PROJECT IMPORTS
# ============================================================

from models import (
    exposure_from_file,
    SinglePointFit,
    BinaryFit,
)

from apertures import *
from detectors import *

# Change these import names to whatever you call
# the Python files containing the cleaned functions.
from pipeline.grid import run_grid_for_group
from pipeline.bfgs import run_bfgs_for_group
from pipeline.hmc import run_hmc_for_group


jax.config.update(
    "jax_enable_x64",
    True
)


# ============================================================
# 1. DATASET
# ============================================================

DATASET = "MAST_10143"


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)


DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / DATASET
    / "MAST"
    / "HST"
)


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / DATASET
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. OPTIONAL TARGET / FILTER SELECTION
# ============================================================
#
# None = run everything found in the dataset.
#
# Or, for example:
#
# FILTERS_TO_RUN = ["F170M"]
#
# TARGETS_TO_RUN = ["2MASS..."]
#
# ============================================================

FILTERS_TO_RUN =  ["F170M"]

TARGETS_TO_RUN = None


# ============================================================
# 3. MODEL SETTINGS
# ============================================================

MODEL_CONFIG = {

    "wid": 64,

    "oversample": 4,

    "psf_oversample": 4,

    "nwavels": 20,

    "npoly": 5,

    "n_zernikes": 20,
}


# ============================================================
# 4. GRID SETTINGS
# ============================================================

GRID_CONFIG = {

    **MODEL_CONFIG,

    "n_aberrations": 26,

    # Single source position grid
    "x_min": -5,
    "x_max": 5,

    "y_min": -5,
    "y_max": 5,

    "Nx": 50,
    "Ny": 50,

    # Binary grid
    "r_min": 0,
    "r_max": 5.0,

    "Nr": 20,

    "theta_min": -180,
    "theta_max": 180,

    "Ntheta": 50,

    "c_min": 1.0,
    "c_max": 10.0,

    "Nc": 20,
}


# ============================================================
# 5. BFGS SETTINGS
# ============================================================

BFGS_CONFIG = {

    "max_steps": 100,

    "n_zernikes":
        MODEL_CONFIG["n_zernikes"],

    "oversample":
        MODEL_CONFIG["oversample"],


    "single_bfgs_keys": [

        "positions",

        "spectrum",

        "aberrations",

        "cold_mask_shift",

        "bias",

        "despace",

        "mag",
    ],


    "binary_bfgs_keys": [

        "positions",

        "separation",

        "position_angle",

        "primary_spectrum",

        "secondary_spectrum",

        "aberrations",

        "cold_mask_shift",

        "bias",

        "despace",

        "mag",
    ],


    "single_scale_values": {

        "positions": 0.1,

        "spectrum": 50.0,

        "aberrations": 1.0,

        "cold_mask_shift": 0.1,

        "bias": 0.01,

        "despace": 1.0,

        "mag": 0.1,
    },


    "binary_scale_values": {

        "positions": 0.1,

        "separation": 0.1,

        "position_angle": 1.0,

        "primary_spectrum": 50.0,

        "secondary_spectrum": 50.0,

        "aberrations": 1.0,

        "cold_mask_shift": 0.1,

        "bias": 0.01,

        "despace": 1.0,

        "mag": 0.1,
    },
}


# ============================================================
# 6. HMC SETTINGS
# ============================================================

HMC_CONFIG = {

    # Screening HMC
    "num_warmup": 20,

    "num_samples": 20,

    "seed": 0,
}


# ============================================================
# 7. PIPELINE OPTIONS
# ============================================================

MAKE_GRID_PLOTS = False

MAKE_BFGS_PLOTS = False

MAKE_HMC_PLOT = True

SHOW_HMC_PLOT = False


# If True:
# existing stages are loaded instead of rerun.
#
# Very useful because HMC is expensive.
RESUME = True


# ============================================================
# 8. BUILD MODEL OBJECTS
# ============================================================

wid = MODEL_CONFIG["wid"]

oversample = (
    MODEL_CONFIG["oversample"]
)

psf_oversample = (
    MODEL_CONFIG["psf_oversample"]
)

nwavels = (
    MODEL_CONFIG["nwavels"]
)

npoly = (
    MODEL_CONFIG["npoly"]
)

n_zernikes = (
    MODEL_CONFIG["n_zernikes"]
)


# Spectrum basis used by SinglePointFit / BinaryFit

spectrum_basis = np.ones(
    (
        nwavels,
        npoly
    )
)


# ------------------------------------------------------------
# GRID optics
# ------------------------------------------------------------

grid_optics = NICMOSOptics(
    512,
    wid,
    oversample
)


grid_oversampled_optics = NICMOSOptics(

    512,

    wid,

    oversample
    *
    psf_oversample,

    psf_oversample=
        psf_oversample
)


# ------------------------------------------------------------
# BFGS / HMC optics
# ------------------------------------------------------------

science_optics = (
    NICMOSSecondaryFresnelOptics(

        512,

        wid,

        oversample,

        mag=3.3,

        defocus=0.0,

        despace=0.0,

        n_zernikes=n_zernikes
    )
)


# ------------------------------------------------------------
# Detector
# ------------------------------------------------------------

detector = NICMOSDetector(
    oversample,
    wid
)


# ============================================================
# 9. FIND FITS FILES
# ============================================================

files = sorted(
    DATA_DIR.glob(
        "*_cal.fits"
    )
)


if len(files) == 0:

    raise FileNotFoundError(

        f"No *_cal.fits files found in:\n"
        f"{DATA_DIR}"
    )


print()
print("=" * 70)
print("DATASET:", DATASET)
print("DATA DIRECTORY:", DATA_DIR)
print("FILES FOUND:", len(files))
print("=" * 70)


# ============================================================
# 10. CREATE TARGET + FILTER GROUPS
# ============================================================

groups = defaultdict(

    lambda: {

        "single": [],

        "binary": [],

        "files": [],
    }
)


for fname in files:

    # --------------------------------------------------------
    # Read metadata first
    # --------------------------------------------------------

    with fits.open(fname) as hdul:

        header = hdul[0].header

        filt = (
            header["FILTER"]
            .strip()
        )

        target = (
            header["TARGNAME"]
            .strip()
        )


    print(
        fname.name,
        "→",
        target,
        "→",
        filt
    )


    # --------------------------------------------------------
    # Optional target selection
    # --------------------------------------------------------

    if TARGETS_TO_RUN is not None:

        if target not in TARGETS_TO_RUN:

            continue


    # --------------------------------------------------------
    # Optional filter selection
    # --------------------------------------------------------

    if FILTERS_TO_RUN is not None:

        if filt not in FILTERS_TO_RUN:

            continue


    # --------------------------------------------------------
    # Create the SINGLE exposure
    # --------------------------------------------------------

    exp_single = exposure_from_file(

        str(fname),

        SinglePointFit(
            spectrum_basis,
            filt
        ),

        crop=wid
    )


    # --------------------------------------------------------
    # Create the BINARY exposure
    # --------------------------------------------------------

    exp_binary = exposure_from_file(

        str(fname),

        BinaryFit(
            spectrum_basis,
            filt
        ),

        crop=wid
    )


    # --------------------------------------------------------
    # Group by TARGET + FILTER
    # --------------------------------------------------------

    key = (
        target,
        filt
    )


    groups[key][
        "single"
    ].append(
        exp_single
    )


    groups[key][
        "binary"
    ].append(
        exp_binary
    )


    groups[key][
        "files"
    ].append(
        str(fname)
    )


# ============================================================
# 11. PRINT GROUP SUMMARY
# ============================================================

print()
print("=" * 70)
print("GROUPS FOUND")
print("=" * 70)


for (
    target,
    filt
), group in groups.items():

    print(

        target,
        "|",
        filt,
        "|",
        len(
            group["single"]
        ),
        "exposure(s)"
    )


if len(groups) == 0:

    raise RuntimeError(
        "No target/filter groups selected."
    )


# ============================================================
# 12. RUN PIPELINE
# ============================================================

pipeline_results = {}


for (
    target,
    filt
), group in sorted(
    groups.items()
):

    print()
    print()
    print("#" * 70)
    print("# TARGET:", target)
    print("# FILTER:", filt)
    print("# EXPOSURES:", len(group["single"]))
    print("#" * 70)


    exposures_single = (
        group["single"]
    )

    exposures_binary = (
        group["binary"]
    )


    # --------------------------------------------------------
    # Result folder for this target + filter
    # --------------------------------------------------------

    safe_target = (
        str(target)
        .replace(" ", "_")
        .replace("/", "_")
    )


    group_dir = (

        RESULTS_DIR

        /

        safe_target

        /

        filt
    )


    group_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    grid_path = (
        group_dir
        /
        "grid.pkl"
    )


    bfgs_path = (
        group_dir
        /
        "bfgs.pkl"
    )


    hmc_path = (
        group_dir
        /
        "hmc.pkl"
    )


    # ========================================================
    # GRID
    # ========================================================

    if (
        RESUME
        and
        grid_path.exists()
    ):

        print()
        print(
            "GRID already exists."
        )

        print(
            "Loading:",
            grid_path
        )


        with open(
            grid_path,
            "rb"
        ) as f:

            grid_output = (
                pickle.load(f)
            )


    else:

        grid_output = (
            run_grid_for_group(

                target=target,

                filt=filt,

                exposures_single=
                    exposures_single,

                exposures_binary=
                    exposures_binary,

                optics=
                    grid_optics,

                oversampled_optics=
                    grid_oversampled_optics,

                detector=
                    detector,

                spectrum_basis=
                    spectrum_basis,

                config=
                    GRID_CONFIG,

                results_dir=
                    RESULTS_DIR,

                make_plots=
                    MAKE_GRID_PLOTS,

                save=True,
            )
        )


    # ========================================================
    # BFGS
    # ========================================================

    if (
        RESUME
        and
        bfgs_path.exists()
    ):

        print()
        print(
            "BFGS already exists."
        )

        print(
            "Loading:",
            bfgs_path
        )


        with open(
            bfgs_path,
            "rb"
        ) as f:

            bfgs_output = (
                pickle.load(f)
            )


    else:

        bfgs_output = (
            run_bfgs_for_group(

                target=target,

                filt=filt,

                exposures_single=
                    exposures_single,

                exposures_binary=
                    exposures_binary,

                grid_output=
                    grid_output,

                optics=
                    science_optics,

                detector=
                    detector,

                config=
                    BFGS_CONFIG,

                results_dir=
                    RESULTS_DIR,

                make_plots=
                    MAKE_BFGS_PLOTS,

                save=True,
            )
        )


    # ========================================================
    # HMC
    # ========================================================

    if (
        RESUME
        and
        hmc_path.exists()
    ):

        print()
        print(
            "HMC already exists."
        )

        print(
            "Skipping expensive HMC run:"
        )

        print(
            hmc_path
        )


        with open(
            hmc_path,
            "rb"
        ) as f:

            hmc_output = (
                pickle.load(f)
            )


    else:

        (
            hmc_output,
            single_mcmc,
            binary_mcmc,
            model_single,
            model_binary,

        ) = run_hmc_for_group(

            target=target,

            filt=filt,

            exposures_single=
                exposures_single,

            exposures_binary=
                exposures_binary,

            bfgs_output=
                bfgs_output,

            optics=
                science_optics,

            detector=
                detector,

            config=
                HMC_CONFIG,

            results_dir=
                RESULTS_DIR,

            save=True,

            make_plot=
                MAKE_HMC_PLOT,

            show_plot=
                SHOW_HMC_PLOT,
        )


    # ========================================================
    # STORE GROUP RESULT
    # ========================================================

    pipeline_results[
        (
            target,
            filt
        )
    ] = {

        "grid":
            grid_output,

        "bfgs":
            bfgs_output,

        "hmc":
            hmc_output,
    }


# ============================================================
# 13. FINISHED
# ============================================================

print()
print()
print("=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)


for (
    target,
    filt
) in pipeline_results:

    print(
        "✓",
        target,
        "|",
        filt
    )


print()
print(
    "Results saved in:"
)

print(
    RESULTS_DIR
)