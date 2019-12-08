"""Utilities for ARPA language models."""
import logging
import shlex
import subprocess
import typing
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


def fst_to_arpa(
    fst_text_path: typing.Union[str, Path],
    isymbols_path: typing.Union[str, Path],
    osymbols_path: typing.Union[str, Path],
    arpa_path: Path,
    **kwargs,
):
    """Convert text FST to ARPA language model using opengrm."""
    for task in fst_to_arpa_tasks(
        fst_text_path, isymbols_path, osymbols_path, arpa_path, **kwargs
    ):
        run_task(task)


# -----------------------------------------------------------------------------


def arpa_to_fst(arpa_path: typing.Union[str, Path], fst_path: typing.Union[str, Path]):
    """Convert ARPA language model to FST. Typically for language model mixing."""
    run_task(arpa_to_fst_task(arpa_path, fst_path))


def arpa_to_fst_task(
    arpa_path: typing.Union[str, Path], fst_path: typing.Union[str, Path]
) -> typing.Dict[str, typing.Any]:
    """Generate doit compatible task for ARPA to FST conversion."""
    return {
        "name": "base_lm_to_fst",
        "file_dep": [arpa_path],
        "targets": [fst_path],
        "actions": ["ngramread --ARPA %(dependencies)s %(targets)s"],
    }


# -----------------------------------------------------------------------------


def fst_to_arpa_tasks(
    fst_text_path: typing.Union[str, Path],
    isymbols_path: typing.Union[str, Path],
    osymbols_path: typing.Union[str, Path],
    arpa_path: typing.Union[str, Path],
    fst_path: typing.Optional[typing.Union[str, Path]] = None,
    counts_path: typing.Optional[typing.Union[str, Path]] = None,
    model_path: typing.Optional[typing.Union[str, Path]] = None,
    base_fst_weight: typing.Optional[
        typing.Tuple[typing.Union[str, Path], float]
    ] = None,
    merge_path: typing.Optional[typing.Union[str, Path]] = None,
) -> typing.Iterable[typing.Dict[str, typing.Any]]:
    """Generate doit compatible tasks for FST to ARPA conversion."""
    # Text -> FST
    fst_text_path = Path(fst_text_path)
    fst_path = Path(fst_path or (fst_text_path.parent / (fst_text_path.stem + ".fst")))

    yield {
        "name": "compile_fst",
        "file_dep": [fst_text_path, isymbols_path, osymbols_path],
        "targets": [fst_path],
        "actions": [
            "fstcompile "
            "--keep_isymbols --keep_osymbols "
            f"--isymbols={isymbols_path} --osymbols={osymbols_path} "
            f"{fst_text_path} %(targets)s"
        ],
    }

    # FST -> n-gram counts
    counts_path = counts_path or Path(str(fst_path) + ".counts")
    yield {
        "name": "intent_counts",
        "file_dep": [fst_path],
        "targets": [counts_path],
        "actions": ["ngramcount %(dependencies)s %(targets)s"],
    }

    # n-gram counts -> model
    model_path = model_path or Path(str(fst_path) + ".model")
    yield {
        "name": "intent_model",
        "file_dep": [counts_path],
        "targets": [model_path],
        "actions": ["ngrammake %(dependencies)s %(targets)s"],
    }

    if base_fst_weight:
        # Mixed language modeling
        base_path, base_weight = base_fst_weight
        if base_weight > 0:
            merge_path = merge_path or Path(str(fst_path) + ".merge")

            # merge
            yield {
                "name": "lm_merge",
                "file_dep": [base_path, model_path],
                "targets": [merge_path],
                "actions": [
                    "ngrammerge "
                    f"--alpha={base_weight} "
                    "%(dependencies)s %(targets)s"
                ],
            }

            # Use merged model instead
            model_path = merge_path

    # model -> ARPA
    yield {
        "name": "intent_arpa",
        "file_dep": [model_path],
        "targets": [arpa_path],
        "actions": ["ngramprint --ARPA %(dependencies)s > %(targets)s"],
    }


# -----------------------------------------------------------------------------


def run_task(task: typing.Dict[str, typing.Any]):
    """Execute a doit compatible task."""
    for action in task["actions"]:
        file_dep = " ".join(f'"{d}"' for d in task.get("file_dep", []))
        targets = " ".join(f'"{t}"' for t in task.get("targets", []))
        command = action % {"dependencies": file_dep, "targets": targets}
        _LOGGER.debug(command)
        subprocess.check_call(command, shell=True)
