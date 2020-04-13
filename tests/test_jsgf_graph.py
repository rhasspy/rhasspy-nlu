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
        print(fst)
        self.assertEqual(
            fst,
            GraphFst(
                intent_fst="0 1 <eps> __label__TestIntent1 0.5\n"
                "1 2 this this 0\n"
                "2 3 is is 0\n"
                "3 4 a a 0\n"
                "4 5 test test 0\n"
                "5 6 <eps> <eps> 0\n"
                "0 7 <eps> __label__TestIntent2 0.5\n"
                "7 8 this this 0\n"
                "8 9 is is 0\n"
                "9 10 a a 0\n"
                "10 11 test test 0\n"
                "11 6 <eps> <eps> 0\n"
                "6\n",
                symbols={
                    "<eps>": 0,
                    "__label__TestIntent1": 1,
                    "this": 2,
                    "is": 3,
                    "a": 4,
                    "test": 5,
                    "__label__TestIntent2": 6,
                },
                input_symbols={"<eps>": 0, "this": 2, "is": 3, "a": 4, "test": 5},
                output_symbols={
                    "<eps>": 0,
                    "__label__TestIntent1": 1,
                    "this": 2,
                    "is": 3,
                    "a": 4,
                    "test": 5,
                    "__label__TestIntent2": 6,
                },
            ),
        )

    # -------------------------------------------------------------------------

    def test_intent_filter_single_fst(self):
        """Test multiple intents, single FST with an intent filter."""
        intents = parse_ini(
            """
        [TestIntent1]
        this is a test

        [TestIntent2]
        this is another test
        """
        )

        graph = intents_to_graph(intents)
        fst = graph_to_fst(graph, intent_filter=lambda intent: intent == "TestIntent1")
        print(fst)
        self.assertEqual(
            fst,
            GraphFst(
                intent_fst="0 1 <eps> __label__TestIntent1 0.5\n"
                "1 2 this this 0\n"
                "2 3 is is 0\n"
                "3 4 a a 0\n"
                "4 5 test test 0\n"
                "5 6 <eps> <eps> 0\n"
                "6\n",
                symbols={
                    "<eps>": 0,
                    "__label__TestIntent1": 1,
                    "this": 2,
                    "is": 3,
                    "a": 4,
                    "test": 5,
                },
                input_symbols={"<eps>": 0, "this": 2, "is": 3, "a": 4, "test": 5},
                output_symbols={
                    "<eps>": 0,
                    "__label__TestIntent1": 1,
                    "this": 2,
                    "is": 3,
                    "a": 4,
                    "test": 5,
                },
            ),
        )

    def test_intent_filter_multiple_fsts(self):
        """Test multiple intents, multiple FSTs with an intent filter."""
        intents = parse_ini(
            """
        [TestIntent1]
        this is a test

        [TestIntent2]
        this is another test
        """
        )

        graph = intents_to_graph(intents)
        fsts = graph_to_fsts(
            graph, intent_filter=lambda intent: intent == "TestIntent1"
        )
        self.assertEqual(
            fsts,
            GraphFsts(
                intent_fsts={
                    "TestIntent1": "0 1 this this 0\n"
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


# -----------------------------------------------------------------------------


class MiscellaneousTestCase(unittest.TestCase):
    """Miscellaneous JSGF graph test cases"""

    # pylint: disable=R0201
    def test_nested_remote_rule(self):
        """Test a nested rule reference from a separate grammar."""
        intents = parse_ini(
            """
        [TestIntent1]
        test_rule_1 = <test_rule_2>
        test_rule_2 = test
        this is a test

        [TestIntent2]
        this is another <TestIntent1.test_rule_1>
        """
        )

        # Will fail to parse if nested rule references are broken
        intents_to_graph(intents)
