"""Test cases for JSGF parser."""
import unittest

from rhasspynlu.jsgf import (
    Sentence,
    Word,
    Sequence,
    SequenceType,
    RuleReference,
    SlotReference,
    Tag,
    get_expression_count,
)


class BasicJsgfTestCase(unittest.TestCase):
    """Basic JSGF test cases."""

    def test_words(self):
        """Basic word sequence."""
        s = Sentence.parse("this is a test")
        self.assertEqual(s.items, [Word("this"), Word("is"), Word("a"), Word("test")])

    def test_optional(self):
        """Basic optional."""
        s = Sentence.parse("this is [a] test")
        self.assertEqual(
            s.items,
            [
                Word("this"),
                Word("is"),
                Sequence(
                    text="a", type=SequenceType.ALTERNATIVE, items=[Word("a"), Word("")]
                ),
                Word("test"),
            ],
        )

    def test_alternative(self):
        """Basic alternative."""
        s = Sentence.parse("this (is | a) test")
        self.assertEqual(
            s.items,
            [
                Word("this"),
                Sequence(
                    text="is | a",
                    type=SequenceType.ALTERNATIVE,
                    items=[Word("is"), Word("a")],
                ),
                Word("test"),
            ],
        )

    def test_rule_reference(self):
        """Basic rule references."""
        s = Sentence.parse("this <is-a> test")
        self.assertEqual(
            s.items,
            [
                Word("this"),
                RuleReference(text="<is-a>", rule_name="is-a"),
                Word("test"),
            ],
        )

        s = Sentence.parse("this <is.a> test")
        self.assertEqual(
            s.items,
            [
                Word("this"),
                RuleReference(text="<is.a>", grammar_name="is", rule_name="a"),
                Word("test"),
            ],
        )

    def test_slot_reference(self):
        """Basic slot reference."""
        s = Sentence.parse("this $is-a test")
        self.assertEqual(
            s.items,
            [Word("this"), SlotReference(text="$is-a", slot_name="is-a"), Word("test")],
        )

    def test_tag_word(self):
        """Tag a word."""
        s = Sentence.parse("this{is} a{test}")
        self.assertEqual(
            s.items,
            [Word("this", tag=Tag(tag_text="is")), Word("a", tag=Tag(tag_text="test"))],
        )

    def test_tag_group(self):
        """Tag a group."""
        s = Sentence.parse("(this is a){test}")
        self.assertEqual(s.tag, Tag(tag_text="test"))
        self.assertEqual(s.items, [Word("this"), Word("is"), Word("a")])

    def test_tag_alternative(self):
        """Tag an alternative."""
        s = Sentence.parse("[this is a]{test}")
        self.assertEqual(s.tag, Tag(tag_text="test"))
        self.assertEqual(
            s.items,
            [
                Sequence(
                    text="this is a",
                    type=SequenceType.GROUP,
                    items=[Word("this"), Word("is"), Word("a")],
                ),
                Word(""),
            ],
        )

    def test_expression_count(self):
        """Test counting number of expressions."""
        s = Sentence.parse("[this] [is] [a] [test]")
        expected_count = 2 * 2 * 2 * 2
        self.assertEqual(get_expression_count(s), expected_count)


# -----------------------------------------------------------------------------


class AdvancedJsgfTestCase(unittest.TestCase):
    """More advanced JSGF test cases."""

    def test_optional_alternative(self):
        """Combined optional and alternative."""
        s = Sentence.parse("this [is | a] test")
        self.assertEqual(
            s.items,
            [
                Word("this"),
                Sequence(
                    text="is | a",
                    type=SequenceType.ALTERNATIVE,
                    items=[Word("is"), Word("a"), Word("")],
                ),
                Word("test"),
            ],
        )

    def test_word_substitution(self):
        """Single word substitutions."""
        s = Sentence.parse("this: :is a:test")
        self.assertEqual(
            s.items,
            [
                Word("this", substitution=""),
                Word("", substitution="is"),
                Word("a", substitution="test"),
            ],
        )

    def test_group_substitution(self):
        """Group substitution."""
        s = Sentence.parse("(this is a):test")
        self.assertEqual(s.substitution, "test")
        self.assertEqual(s.items, [Word("this"), Word("is"), Word("a")])

    def test_alternative_substitution(self):
        """Alternative substitution."""
        s = Sentence.parse("this [is a]:isa test")
        self.assertEqual(
            s.items,
            [
                Word("this"),
                Sequence(
                    text="is a",
                    type=SequenceType.ALTERNATIVE,
                    substitution="isa",
                    items=[
                        Sequence(
                            text="is a",
                            type=SequenceType.GROUP,
                            items=[Word("is"), Word("a")],
                        ),
                        Word(""),
                    ],
                ),
                Word("test"),
            ],
        )

    def test_tag_substitution(self):
        """Tag substitution."""
        s = Sentence.parse("(this is){a:test}")
        self.assertEqual(s.tag, Tag(tag_text="a", substitution="test"))
        self.assertEqual(s.items, [Word("this"), Word("is")])

    def test_tag_and_word_substitution(self):
        """Tag and word substitutions."""
        s = Sentence.parse("(this:is){a:test}")
        self.assertEqual(s.tag, Tag(tag_text="a", substitution="test"))
        self.assertEqual(s.items, [Word("this", substitution="is")])

    def test_nested_optionals(self):
        """Optional inside an optional."""
        s = Sentence.parse("this [[is] a] test")
        self.assertEqual(
            s.items,
            [
                Word("this"),
                Sequence(
                    text="[is] a",
                    type=SequenceType.ALTERNATIVE,
                    items=[
                        Sequence(
                            text="[is] a",
                            type=SequenceType.GROUP,
                            items=[
                                Sequence(
                                    text="is",
                                    type=SequenceType.ALTERNATIVE,
                                    items=[Word("is"), Word("")],
                                ),
                                Word("a"),
                            ],
                        ),
                        Word(""),
                    ],
                ),
                Word("test"),
            ],
        )

    def test_implicit_sequences(self):
        """Implicit sequences around alternative."""
        s = Sentence.parse("this is | a test")
        self.assertEqual(s.type, SequenceType.ALTERNATIVE)
        self.assertEqual(
            s.items,
            [
                Sequence(
                    text="this is",
                    type=SequenceType.GROUP,
                    items=[Word("this"), Word("is")],
                ),
                Sequence(
                    text="a test",
                    type=SequenceType.GROUP,
                    items=[Word("a"), Word("test")],
                ),
            ],
        )

    def test_implicit_sequence_with_rule(self):
        """Implicit sequence around alternative with a rule reference."""
        s = Sentence.parse("this | is a <test>")
        self.assertEqual(s.type, SequenceType.ALTERNATIVE)
        self.assertEqual(
            s.items,
            [
                Word("this"),
                Sequence(
                    text="is a <test>",
                    type=SequenceType.GROUP,
                    items=[
                        Word("is"),
                        Word("a"),
                        RuleReference(text="<test>", rule_name="test"),
                    ],
                ),
            ],
        )


# -----------------------------------------------------------------------------


class OtherJsgfTestCase(unittest.TestCase):
    """Corner cases exposed by testing and users."""

    def test_den_overhead(self):
        """Sequence of optionals/alternatives plus tag substitution."""
        s = Sentence.parse(
            "toggle [the] (den | playroom) [light] {light_name:den_overhead}"
        )
        self.assertEqual(
            s.items,
            [
                Word("toggle"),
                Sequence(
                    text="the",
                    type=SequenceType.ALTERNATIVE,
                    items=[Word("the"), Word("")],
                ),
                Sequence(
                    text="den | playroom",
                    type=SequenceType.ALTERNATIVE,
                    items=[Word("den"), Word("playroom")],
                ),
                Sequence(
                    text="light",
                    type=SequenceType.ALTERNATIVE,
                    items=[Word("light"), Word("")],
                    tag=Tag(tag_text="light_name", substitution="den_overhead"),
                ),
            ],
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
