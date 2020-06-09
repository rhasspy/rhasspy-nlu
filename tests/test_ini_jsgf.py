"""Test cases for ini/JSGF grammar parser."""
import unittest

from rhasspynlu.ini_jsgf import get_intent_counts, parse_ini, split_rules
from rhasspynlu.jsgf import Sentence, Sequence, SequenceType, Tag, Word, walk_expression


class IniJsgfTestCase(unittest.TestCase):
    """Test cases for ini/JSGF grammar parser."""

    def test_parse(self):
        """Test ini/JSGF parser."""
        ini_text = """
        [TestIntent1]
        this is a test

        [TestIntent2]
        this is another test
        """

        intents = parse_ini(ini_text)
        self.assertEqual(
            intents,
            {
                "TestIntent1": [
                    Sentence(
                        text="this is a test",
                        items=[Word("this"), Word("is"), Word("a"), Word("test")],
                    )
                ],
                "TestIntent2": [
                    Sentence(
                        text="this is another test",
                        items=[Word("this"), Word("is"), Word("another"), Word("test")],
                    )
                ],
            },
        )

    def test_escape(self):
        """Test escaped optional."""
        ini_text = """
        [TestIntent1]
        \\[this] is a test
        """

        intents = parse_ini(ini_text)
        self.assertEqual(
            intents,
            {
                "TestIntent1": [
                    Sentence(
                        text="[this] is a test",
                        items=[
                            Sequence(
                                text="this",
                                type=SequenceType.ALTERNATIVE,
                                items=[Word("this"), Word("")],
                            ),
                            Word("is"),
                            Word("a"),
                            Word("test"),
                        ],
                    )
                ]
            },
        )

    def test_intent_counts(self):
        """Test sentence counts by intent."""
        ini_text = """
        [TestIntent1]
        this [is] [a] test
        this is [another] test

        [TestIntent2]
        this is (my | your| another) test
        """

        intents = parse_ini(ini_text)
        intent_counts = get_intent_counts(intents)
        self.assertEqual(
            intent_counts,
            {
                "TestIntent1": (1 * 2 * 2 * 1) + (1 * 1 * 2 * 1),
                "TestIntent2": (1 * 1 * 3 * 1),
            },
        )

    def test_transform(self):
        """Test sentence transform."""
        ini_text = """
        [TestIntent1]
        THIS IS A TEST
        """

        intents = parse_ini(ini_text, sentence_transform=str.lower)
        self.assertEqual(
            intents,
            {
                "TestIntent1": [
                    Sentence(
                        text="this is a test",
                        items=[Word("this"), Word("is"), Word("a"), Word("test")],
                    )
                ]
            },
        )

    def test_intent_filter(self):
        """Test filtering intents."""
        ini_text = """
        [TestIntent1]
        this is a test

        [TestIntent2]
        this is another test
        """

        intents = parse_ini(ini_text, intent_filter=lambda n: n != "TestIntent2")
        self.assertEqual(
            intents,
            {
                "TestIntent1": [
                    Sentence(
                        text="this is a test",
                        items=[Word("this"), Word("is"), Word("a"), Word("test")],
                    )
                ]
            },
        )

    def test_walk(self):
        """Test Expression.walk with rule and slot reference."""
        ini_text = """
        [SetAlarm]
        minutes = $minute minutes
        set alarm for <minutes>
        """

        intents = parse_ini(ini_text)
        sentences, replacements = split_rules(intents)
        replacements["$minute"] = [Sentence.parse("2 | 3")]

        def num2words(word):
            if not isinstance(word, Word):
                return

            try:
                n = int(word.text)
                if n == 2:
                    word.text = "two"
                    word.substitution = "2"
                elif n == 3:
                    word.text = "three"
                    word.substitution = "3"
            except ValueError:
                pass

        for s in sentences["SetAlarm"]:
            walk_expression(s, num2words, replacements)

        # Verify minute digits were replaced
        minute = replacements["$minute"][0]
        self.assertEqual(
            minute,
            Sentence(
                text="2 | 3",
                type=SequenceType.GROUP,
                items=[
                    Sequence(
                        text="2 | 3",
                        type=SequenceType.ALTERNATIVE,
                        items=[
                            Word("two", substitution="2"),
                            Word("three", substitution="3"),
                        ],
                    )
                ],
            ),
        )

    def test_final_optional_entity(self):
        """Ensure final optional entity has tag."""

        ini_text = """
        [ChangeDisplay]
        display [(page | layer){layout}]
        """

        intents = parse_ini(ini_text)
        self.assertEqual(
            intents,
            {
                "ChangeDisplay": [
                    Sentence(
                        text="display [(page | layer){layout}]",
                        items=[
                            Word("display"),
                            Sequence(
                                text="(page | layer){layout}",
                                type=SequenceType.ALTERNATIVE,
                                items=[
                                    Sequence(
                                        text="page | layer",
                                        type=SequenceType.GROUP,
                                        tag=Tag(tag_text="layout"),
                                        items=[
                                            Sequence(
                                                text="page | layer",
                                                type=SequenceType.ALTERNATIVE,
                                                items=[Word("page"), Word("layer")],
                                            )
                                        ],
                                    ),
                                    Word(""),
                                ],
                            ),
                        ],
                    )
                ]
            },
        )

    def test_rule_with_same_name(self):
        """Ensure referenced rule names don't conflict with local names."""

        ini_text = """
        [Test1]
        rule = is a test
        this <rule>

        [Test2]
        rule = <Test1.rule>
        this <rule>
        """

        _intents = parse_ini(ini_text)
        intents, replacements = split_rules(_intents)

        # Walk all sentences to make sure there's no infinite recursion
        for intent, sentences in intents.items():
            for sentence in sentences:
                walk_expression(sentence, lambda x: x, replacements)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
