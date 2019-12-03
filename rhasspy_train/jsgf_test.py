"""Test cases for JSGF parser."""
import unittest

from jsgf import (
    Sentence,
    Word,
    Sequence,
    SequenceType,
    RuleReference,
    SlotReference,
    Tag,
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
        self.assertEqual(s.items, [Word("this"), Word("is"), Word("a"), Word("")])


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
                    items=[Word("is"), Word("a"), Word("")],
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
