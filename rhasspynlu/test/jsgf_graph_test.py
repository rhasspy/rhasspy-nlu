"""Test cases for JSGF graph functions."""
import unittest

from rhasspynlu.ini_jsgf import parse_ini
from rhasspynlu.jsgf_graph import intents_to_graph, graph_to_fsts, GraphFsts


class FstTestCase(unittest.TestCase):
    """Test cases for OpenFST conversion."""

    def test_single_sentence(self):
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
                    "TestIntent": "0 1 this this\n"
                    "1 2 is is\n"
                    "2 3 a a\n"
                    "3 4 test test\n"
                    "4 5 <eps> <eps>\n"
                    "5\n"
                },
                symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4},
                input_symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4},
                output_symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4},
            ),
        )

    def test_substitution(self):
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
                    "TestIntent": "0 1 this this\n"
                    "1 2 is is\n"
                    "2 3 a a\n"
                    "3 4 test <eps>\n"
                    "4 5 <eps> sub\n"
                    "5 6 <eps> <eps>\n"
                    "6\n"
                },
                symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4, "sub": 5},
                input_symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4},
                output_symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "sub": 5},
            ),
        )

    def test_optional(self):
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
                    "TestIntent": "0 1 this this\n"
                    "1 2 is is\n"
                    "2 3 a a\n"
                    "2 4 <eps> <eps>\n"
                    "3 5 <eps> <eps>\n"
                    "4 5 <eps> <eps>\n"
                    "5 6 test test\n"
                    "6 7 <eps> <eps>\n"
                    "7\n"
                },
                symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4},
                input_symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4},
                output_symbols={"<eps>": 0, "this": 1, "is": 2, "a": 3, "test": 4},
            ),
        )

    def test_multiple_sentences(self):
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
                    "TestIntent1": "0 1 this this\n"
                    "1 2 is is\n"
                    "2 3 a a\n"
                    "3 4 test test\n"
                    "4 5 <eps> <eps>\n"
                    "5\n",
                    "TestIntent2": "0 1 this this\n"
                    "1 2 is is\n"
                    "2 3 another another\n"
                    "3 4 test test\n"
                    "4 5 <eps> <eps>\n"
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
