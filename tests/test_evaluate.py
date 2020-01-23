"""Test cases for evaluation."""
import unittest

from rhasspynlu.intent import Recognition, Intent, Entity
from rhasspynlu.evaluate import evaluate_intents, get_word_error


class WordErrorEvaluationTests(unittest.TestCase):
    """Test cases for word error."""

    def test_wer_perfect(self):
        """Test word error rate calculation with perfect match."""
        reference = "this is a test".split()
        hypothesis = reference

        result = get_word_error(reference, hypothesis)
        self.assertEqual(4, result.matches)
        self.assertEqual(0, result.errors)
        self.assertEqual(0, result.substitutions)
        self.assertEqual(0, result.insertions)
        self.assertEqual(0, result.deletions)
        self.assertEqual(0, result.error_rate)

        self.assertEqual(reference, result.reference)
        self.assertEqual(hypothesis, result.hypothesis)
        self.assertEqual(reference, result.differences)

    def test_wer_substitution(self):
        """Test word error rate calculation with a substitution."""
        reference = "this is a test".split()
        hypothesis = "this is bad test".split()

        result = get_word_error(reference, hypothesis)
        self.assertEqual(1, result.errors)
        self.assertEqual(3, result.matches)
        self.assertEqual(1, result.substitutions)
        self.assertEqual(0, result.insertions)
        self.assertEqual(0, result.deletions)
        self.assertEqual(0.25, result.error_rate)

        self.assertEqual(reference, result.reference)
        self.assertEqual(hypothesis, result.hypothesis)
        self.assertEqual(["this", "is", "a:bad", "test"], result.differences)

    def test_wer_multiple(self):
        """Test word error rate calculation with a multiple insertions/deletions/substitutions."""
        reference = "this is a test".split()
        hypothesis = "this bad test test opps".split()

        result = get_word_error(reference, hypothesis)
        self.assertEqual(3, result.errors)
        self.assertEqual(2, result.matches)
        self.assertEqual(2, result.substitutions)
        self.assertEqual(1, result.insertions)
        self.assertEqual(0, result.deletions)
        self.assertEqual(0.75, result.error_rate)

        self.assertEqual(reference, result.reference)
        self.assertEqual(hypothesis, result.hypothesis)
        self.assertEqual(
            ["this", "is:bad", "a:test", "test", "+opps"], result.differences
        )


# -----------------------------------------------------------------------------


class IntentEvaluationTests(unittest.TestCase):
    """Test cases for intent evaluation."""

    def test_evaluate_perfect(self):
        """Test intent evaluation with a perfect match."""
        expected = {
            "test1": Recognition(
                intent=Intent(name="TestIntent"),
                entities=[Entity(entity="testEntity", value="testValue")],
                text="this is a test",
            )
        }

        actual = expected
        report = evaluate_intents(expected, actual)

        self.assertEqual(1, report.num_wavs)
        self.assertEqual(1, report.num_intents)
        self.assertEqual(1, report.num_entities)

        self.assertEqual(1, report.correct_intent_names)
        self.assertEqual(1, report.correct_entities)
        self.assertEqual(1, report.correct_intent_and_entities)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
