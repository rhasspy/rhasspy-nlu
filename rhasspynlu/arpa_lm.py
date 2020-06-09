"""Utilities for ARPA language models."""
import logging
import re
import shutil
import subprocess
import tempfile
import typing
from pathlib import Path

import networkx as nx

from .jsgf_graph import graph_to_fst
from .ngram import get_intent_ngram_counts

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


def graph_to_arpa(
    graph: nx.DiGraph,
    arpa_path: typing.Union[str, Path],
    vocab_path: typing.Optional[typing.Union[str, Path]] = None,
    intent_filter: typing.Optional[typing.Callable[[str], bool]] = None,
    **fst_to_arpa_args,
):
    """Convert intent graph to ARPA language model using opengrm."""
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        fst_text_path = temp_dir / "graph.fst.txt"
        isymbols_path = temp_dir / "isymbols.txt"
        osymbols_path = temp_dir / "osymbols.txt"

        # Graph -> binary FST
        graph_to_fst(graph, intent_filter=intent_filter).write_fst(
            fst_text_path, isymbols_path, osymbols_path
        )

        if vocab_path:
            # Extract vocabulary
            with open(vocab_path, "w") as vocab_file:
                with open(isymbols_path, "r") as isymbols_file:
                    for line in isymbols_file:
                        line = line.strip()
                        if line:
                            # symbol N
                            isymbol = line[: line.rfind(" ")]
                            if isymbol and (isymbol[0] not in ["_", "<"]):
                                print(isymbol, file=vocab_file)

                _LOGGER.debug("Wrote vocabulary to %s", vocab_path)

        # Convert to ARPA
        fst_to_arpa(
            fst_text_path, isymbols_path, osymbols_path, arpa_path, **fst_to_arpa_args
        )


def fst_to_arpa(
    fst_text_path: typing.Union[str, Path],
    isymbols_path: typing.Union[str, Path],
    osymbols_path: typing.Union[str, Path],
    arpa_path: typing.Union[str, Path],
    **kwargs,
):
    """Convert text FST to ARPA language model using opengrm."""
    for task in fst_to_arpa_tasks(
        fst_text_path, isymbols_path, osymbols_path, arpa_path, **kwargs
    ):
        run_task(task)


def graph_to_arpa_small(
    graph: nx.DiGraph,
    arpa_path: typing.Union[str, Path],
    vocab_path: typing.Optional[typing.Union[str, Path]] = None,
    dictionary_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
    balance_counts: bool = True,
    estimate_ngram: typing.Optional[typing.Union[str, Path]] = None,
):
    """Convert intent graph to ARPA language model using MITLM. Works better for small graphs."""
    estimate_ngram = estimate_ngram or shutil.which("estimate-ngram")
    assert estimate_ngram, "Missing estimate-ngram in PATH"

    # Generate counts
    _LOGGER.debug("Generating ngram counts")
    intent_counts = get_intent_ngram_counts(graph, balance_counts=balance_counts)

    # Create ngram counts file
    with tempfile.NamedTemporaryFile(mode="w+") as count_file:
        for intent_name in intent_counts:
            for ngram, count in intent_counts[intent_name].items():
                if dictionary_word_transform:
                    ngram = [dictionary_word_transform(w) for w in ngram]

                # word [word] ... <TAB> count
                print(*ngram, file=count_file, end="")
                print("\t", count, file=count_file)

        count_file.seek(0)
        with tempfile.NamedTemporaryFile(mode="w+") as vocab_file:
            ngram_command = [
                str(estimate_ngram),
                "-order",
                "3",
                "-counts",
                count_file.name,
                "-write-lm",
                str(arpa_path),
                "-write-vocab",
                vocab_file.name,
            ]

            _LOGGER.debug(ngram_command)
            subprocess.check_call(ngram_command)

            if vocab_path:
                # Copy over real file
                vocab_file.seek(0)
                with open(vocab_path, "w") as real_vocab_file:
                    for line in vocab_file:
                        line = line.strip()
                        if line and (line[0] not in ["_", "<"]):
                            print(line, file=real_vocab_file)


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
    fst_path = Path(fst_path or fst_text_path.with_suffix(".fst"))

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
        "actions": ["ngrammake --method=witten_bell %(dependencies)s %(targets)s"],
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


def mix_language_models(
    small_arpa_path: typing.Union[str, Path],
    large_fst_path: typing.Union[str, Path],
    mix_weight: float,
    mixed_arpa_path: typing.Union[str, Path],
    small_fst_path: typing.Optional[typing.Union[str, Path]] = None,
    mixed_fst_path: typing.Optional[typing.Union[str, Path]] = None,
):
    """Mix a large pre-built FST language model with a small ARPA model."""
    with tempfile.NamedTemporaryFile(suffix=".fst", mode="w+") as small_fst:
        # Convert small ARPA to FST
        if not small_fst_path:
            small_fst_path = small_fst.name

        small_command = [
            "ngramread",
            "--ARPA",
            str(small_arpa_path),
            str(small_fst_path),
        ]
        _LOGGER.debug(small_command)
        subprocess.check_call(small_command)

        with tempfile.NamedTemporaryFile(suffix=".fst", mode="w+") as mixed_fst:
            # Merge ngram FSTs
            small_fst.seek(0)

            if not mixed_fst_path:
                mixed_fst_path = mixed_fst.name

            merge_command = [
                "ngrammerge",
                f"--alpha={mix_weight}",
                str(large_fst_path),
                str(small_fst_path),
                str(mixed_fst_path),
            ]
            _LOGGER.debug(merge_command)
            subprocess.check_call(merge_command)

            # Write to final ARPA
            mixed_fst.seek(0)
            print_command = [
                "ngramprint",
                "--ARPA",
                str(mixed_fst_path),
                str(mixed_arpa_path),
            ]
            _LOGGER.debug(print_command)
            subprocess.check_call(print_command)


# -----------------------------------------------------------------------------


def get_perplexity(
    text: str, language_model_fst: typing.Union[str, Path], debug: bool = False
) -> typing.Optional[float]:
    """Compute perplexity of text with ngramperplexity."""

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_path = Path(temp_dir_name)
        text_path = temp_path / "sentence.txt"
        text_path.write_text(text)

        symbols_path = temp_path / "sentence.syms"
        symbols_command = ["ngramsymbols", str(text_path), str(symbols_path)]
        _LOGGER.debug(symbols_command)
        subprocess.check_call(symbols_command)

        far_path = temp_path / "sentence.far"
        compile_command = [
            "farcompilestrings",
            f"-symbols={symbols_path}",
            "-keep_symbols=1",
            str(text_path),
            str(far_path),
        ]
        _LOGGER.debug(compile_command)
        subprocess.check_call(compile_command)

        verbosity = 1 if debug else 0
        perplexity_command = [
            "ngramperplexity",
            f"--v={verbosity}",
            str(language_model_fst),
            str(far_path),
        ]
        _LOGGER.debug(perplexity_command)
        output = subprocess.check_output(perplexity_command).decode()

        _LOGGER.debug(output)

        last_line = output.strip().splitlines()[-1]
        match = re.match(r"^.*perplexity\s*=\s*(.+)$", last_line)

        if match:
            perplexity = float(match.group(1))
            return perplexity

        return None


# -----------------------------------------------------------------------------


def run_task(task: typing.Dict[str, typing.Any]):
    """Execute a doit compatible task."""
    name = task.get("name", "task")
    for action in task["actions"]:
        file_dep = " ".join(f'"{d}"' for d in task.get("file_dep", []))
        targets = " ".join(f'"{t}"' for t in task.get("targets", []))
        command = action % {"dependencies": file_dep, "targets": targets}
        _LOGGER.debug("%s: %s", name, command)
        subprocess.check_call(command, shell=True)
