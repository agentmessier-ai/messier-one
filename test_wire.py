"""python -m unittest test_wire.py   (no GPU, no model)"""
import unittest

import wire


class OptionText(unittest.TestCase):
    def test_description_is_what_the_model_reads(self):
        self.assertEqual(wire.options({"a": "refund the order", "b": "escalate"}), {"a": "refund the order", "b": "escalate"})

    def test_null_description_falls_back_to_the_option_name(self):
        self.assertEqual(wire.options({"refund": None, "escalate": "send to a human"}), {"refund": "refund", "escalate": "send to a human"})

    def test_blank_description_falls_back_to_the_option_name(self):
        self.assertEqual(wire.options({"refund": "  ", "escalate": ""}), {"refund": "refund", "escalate": "escalate"})

    def test_array_of_names_is_accepted(self):
        self.assertEqual(wire.options(["red", "green"]), {"red": "red", "green": "green"})

    def test_structured_description_is_kept_for_the_serialiser(self):
        self.assertEqual(wire.options({"a": {"x": 1}}), {"a": {"x": 1}})


class Confidence(unittest.TestCase):
    def test_choice_is_top_probability_rescaled_above_chance(self):
        self.assertAlmostEqual(wire.confidence("choice", {"a": 0.7, "b": 0.2, "c": 0.1}), (0.7 - 1 / 3) / (1 - 1 / 3))

    def test_choice_at_chance_is_zero_and_certain_is_one(self):
        self.assertAlmostEqual(wire.confidence("choice", {"a": 0.5, "b": 0.5}), 0.0)
        self.assertAlmostEqual(wire.confidence("choice", {"a": 1.0, "b": 0.0}), 1.0)

    def test_single_option_is_certain(self):
        self.assertEqual(wire.confidence("choice", {"only": 1.0}), 1.0)

    def test_score_is_one_minus_spread_around_the_mode_relative_to_uniform(self):
        # mode is level 1; mean distance 0.2*1 + 0.2*1 = 0.4; uniform over 3 levels: (1 + 0 + 1) / 3
        self.assertAlmostEqual(wire.confidence("score", {"0": 0.2, "1": 0.6, "2": 0.2}), 1 - 0.4 / (2 / 3))

    def test_score_levels_are_ordered_by_number_not_by_text(self):
        p = {str(i): 0.0 for i in range(11)}; p["10"] = 0.9; p["9"] = 0.1
        self.assertAlmostEqual(wire.confidence("score", p), 1 - 0.1 / (sum(range(11)) / 11))

    def test_score_never_goes_below_zero(self):
        self.assertEqual(wire.confidence("score", {"0": 0.34, "1": 0.0, "2": 0.33, "3": 0.33}), 0.0)


class TopProbability(unittest.TestCase):
    def test_reads_the_probabilities_not_the_confidence(self):
        self.assertEqual(wire.top_probability({"type": "choice", "choice": "a", "confidence": 0.1, "probabilities": {"a": 0.55, "b": 0.45}}), 0.55)

    def test_noul_is_the_likelier_side(self):
        self.assertAlmostEqual(wire.top_probability({"type": "noul", "noul": 0.2}), 0.8)


class Answer(unittest.TestCase):
    def test_choice_answer(self):
        a = wire.answer("choice", {"a": 0.25, "b": 0.75})
        self.assertEqual((a["type"], a["choice"], a["probabilities"]), ("choice", "b", {"a": 0.25, "b": 0.75}))
        self.assertAlmostEqual(a["confidence"], 0.5)

    def test_score_answer_is_the_expected_level(self):
        a = wire.answer("score", {"0": 0.5, "1": 0.0, "2": 0.5})
        self.assertEqual(a["type"], "score"); self.assertAlmostEqual(a["score"], 1.0)

    def test_noul_answer_has_no_confidence(self):
        self.assertEqual(wire.answer("noul", {"true": 0.9, "false": 0.1}), {"type": "noul", "noul": 0.9})

    def test_boolean_is_the_internal_name_of_noul(self):
        self.assertEqual(wire.answer("boolean", {"true": 0.9, "false": 0.1}), {"type": "noul", "noul": 0.9})


if __name__ == "__main__":
    unittest.main()
