"""Test cases for JSGF graph functions."""
import unittest

from rhasspynlu.ini_jsgf import parse_ini
from rhasspynlu.jsgf_graph import (
    GraphFst,
    GraphFsts,
    graph_to_fst,
    graph_to_fsts,
    intents_to_graph,
)


class FstTestCase(unittest.TestCase):
    """Test cases for OpenFST conversion."""

    def test_single_sentence(self):
        """Test one intent, one sentence."""
        intents = parse_ini(
            """
        [TestIntent]
        this is a test
        """
        )

        graph = intents_to_graph(intents)
        fsts = graph_to_fsts(graph)
        self.assertEqual(
            fsts,
            GraphFsts(
                intent_fsts={
                    "TestIntent": "0 1 this this 0\n"
                    "1 2 is is 0\n"
                    "2 3 a a 0\n"
                    "3 4 test test 0\n"
                    "4 5 <eps> <eps> 0\n"
                    "5\n"
                },
                symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4},
                input_symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4},
                output_symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4},
            ),
        )

    def test_substitution(self):
        """Test one intent, one sentence with a substitution."""
        intents = parse_ini(
            """
        [TestIntent]
        this is a test:sub
        """
        )

        graph = intents_to_graph(intents)
        fsts = graph_to_fsts(graph)
        self.assertEqual(
            fsts,
            GraphFsts(
                intent_fsts={
                    "TestIntent": "0 1 this this 0\n"
                    "1 2 is is 0\n"
                    "2 3 a a 0\n"
                    "3 4 test <eps> 0\n"
                    "4 5 <eps> sub 0\n"
                    "5 6 <eps> <eps> 0\n"
                    "6\n"
                },
                symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4, "sub": 5},
                input_symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4},
                output_symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "sub": 5},
            ),
        )

    def test_optional(self):
        """Test one intent, one sentence with an optional word."""
        intents = parse_ini(
            """
        [TestIntent]
        this is [a] test
        """
        )

        graph = intents_to_graph(intents)
        fsts = graph_to_fsts(graph)
        self.assertEqual(
            fsts,
            GraphFsts(
                intent_fsts={
                    "TestIntent": "0 1 this this 0\n"
                    "1 2 is is 0\n"
                    "2 3 a a 0\n"
                    "2 4 <eps> <eps> 0\n"
                    "3 5 <eps> <eps> 0\n"
                    "4 5 <eps> <eps> 0\n"
                    "5 6 test test 0\n"
                    "6 7 <eps> <eps> 0\n"
                    "7\n"
                },
                symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4},
                input_symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4},
                output_symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4},
            ),
        )

    def test_multiple_sentences(self):
        """Test multiple intents."""
        intents = parse_ini(
            """
        [TestIntent1]
        this is a test

        [TestIntent2]
        this is another test
        """
        )

        graph = intents_to_graph(intents)
        fsts = graph_to_fsts(graph)
        self.assertEqual(
            fsts,
            GraphFsts(
                intent_fsts={
                    "TestIntent1": "0 1 this this 0\n"
                    "1 2 is is 0\n"
                    "2 3 a a 0\n"
                    "3 4 test test 0\n"
                    "4 5 <eps> <eps> 0\n"
                    "5\n",
                    "TestIntent2": "0 1 this this 0\n"
                    "1 2 is is 0\n"
                    "2 3 another another 0\n"
                    "3 4 test test 0\n"
                    "4 5 <eps> <eps> 0\n"
                    "5\n",
                },
                symbols={
                    "<eps>": 0,
                    "this": 1,
                    "is": 2,
                    "a": 3,
                    "test": 4,
                    "another": 5,
                },
                input_symbols={
                    "<eps>": 0,
                    "this": 1,
                    "is": 2,
                    "a": 3,
                    "test": 4,
                    "another": 5,
                },
                output_symbols={
                    "<eps>": 0,
                    "this": 1,
                    "is": 2,
                    "a": 3,
                    "test": 4,
                    "another": 5,
                },
            ),
        )

    # -------------------------------------------------------------------------

    def test_one_weight(self):
        """Single intent should have an edge weight of 0."""
        intents = parse_ini(
            """
        [TestIntent]
        this is a test
        """
        )

        graph = intents_to_graph(intents)
        fst = graph_to_fst(graph)
        self.assertEqual(
            fst,
            GraphFst(
                intent_fst="0 1 <eps> __label__TestIntent 0\n"
                "1 2 this this 0\n"
                "2 3 is is 0\n"
                "3 4 a a 0\n"
                "4 5 test test 0\n"
                "5 6 <eps> <eps> 0\n"
                "6\n",
                symbols={
                    "<eps>": 0,
                    "__label__TestIntent": 1,
                    "this": 2,
                    "is": 3,
                    "a": 4,
                    "test": 5,
                },
                input_symbols={"<eps>": 0, "this": 2, "is": 3, "a": 4, "test": 5},
                output_symbols={
                    "<eps>": 0,
                    "__label__TestIntent": 1,
                    "this": 2,
                    "is": 3,
                    "a": 4,
                    "test": 5,
                },
            ),
        )

    def test_multiple_weights(self):
        """Multiple intents should have balanced weights."""
        intents = parse_ini(
            """
        [TestIntent1]
        this is a test

        [TestIntent2]
        this is a test
        """
        )

        graph = intents_to_graph(intents)
        fst = graph_to_fst(graph)
        self.assertEqual(
            fst,
            GraphFst(
                intent_fst="0 1 <eps> __label__TestIntent1 0.5\n"
                "0 2 <eps> __label__TestIntent2 0.5\n"
                "1 3 this this 0\n"
                "2 4 this this 0\n"
                "3 5 is is 0\n"
                "4 6 is is 0\n"
                "5 7 a a 0\n"
                "6 8 a a 0\n"
                "7 9 test test 0\n"
                "8 10 test test 0\n"
                "9 11 <eps> <eps> 0\n"
                "10 11 <eps> <eps> 0\n"
                "11\n",
                symbols={
                    "<eps>": 0,
                    "__label__TestIntent1": 1,
                    "__label__TestIntent2": 2,
                    "this": 3,
                    "is": 4,
                    "a": 5,
                    "test": 6,
                },
                input_symbols={"<eps>": 0, "this": 3, "is": 4, "a": 5, "test": 6},
                output_symbols={
                    "<eps>": 0,
                    "__label__TestIntent1": 1,
                    "__label__TestIntent2": 2,
                    "this": 3,
                    "is": 4,
                    "a": 5,
                    "test": 6,
                },
            ),
        )
