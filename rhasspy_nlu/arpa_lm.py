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
    for task in fst_to_arpa_tasks(
        fst_text_path, isymbols_path, osymbols_path, arpa_path, **kwargs
    ):
        for action in task["actions"]:
            file_dep = " ".join(f'"{d}"' for d in task.get("file_dep", []))
            targets = " ".join(f'"{t}"' for t in task.get("targets", []))
            command = action % {"dependencies": file_dep, "targets": targets}
            _LOGGER.debug(command)
            subprocess.check_call(command, shell=True)


def fst_to_arpa_tasks(
    fst_text_path: typing.Union[str, Path],
    isymbols_path: typing.Union[str, Path],
    osymbols_path: typing.Union[str, Path],
    arpa_path: typing.Union[str, Path],
    fst_path: typing.Optional[typing.Union[str, Path]] = None,
    counts_path: typing.Optional[typing.Union[str, Path]] = None,
    model_path: typing.Optional[typing.Union[str, Path]] = None,
) -> typing.Iterable[typing.Dict[str, typing.Any]]:
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

    # model -> ARPA
    yield {
        "name": "intent_arpa",
        "file_dep": [model_path],
        "targets": [arpa_path],
        "actions": ["ngramprint --ARPA %(dependencies)s > %(targets)s"],
    }
