"""Test cases for JSGF graph functions."""
import unittest

from .ini_jsgf import parse_ini
from .jsgf_graph import intents_to_graph, graph_to_fsts


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
            {
                "TestIntent": "0 1 this this\n"
                "1 2 is is\n"
                "2 3 a a\n"
                "3 4 test test\n"
                "4 5 <eps> <eps>\n"
                "5\n"
            },
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
            {
                "TestIntent": "0 1 this this\n"
                "1 2 is is\n"
                "2 3 a a\n"
                "3 4 test <eps>\n"
                "4 5 <eps> sub\n"
                "5 6 <eps> <eps>\n"
                "6\n"
            },
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
            {
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
        )
