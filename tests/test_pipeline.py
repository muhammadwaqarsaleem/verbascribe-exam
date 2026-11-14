"""
tests/test_pipeline.py
Automated End-to-End Integration Tests for VerbaScribe-Exam.
Validates: MockASR -> IntentParser -> StateMachine execution loop.
"""
import os
import unittest
from src.schemas import IntentSlotResult, ActionType
from src.mock_asr import MockASR
from src.intent_parser import IntentParser
from src.state_machine import StateMachine


class TestVerbaScribePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fallback to current directory or models folder
        model_dir = "models/intent_slot_model"
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model weights not found at {model_dir}. "
                "Download them from Google Drive into models/intent_slot_model/ first."
            )
        cls.parser = IntentParser(model_dir=model_dir)
        cls.mock_asr = MockASR()

    def setUp(self):
        self.state_machine = StateMachine()

    def test_01_plain_academic_dictation(self):
        """Verifies scientific dictation appends cleanly to the buffer."""
        utterance = "kinetic energy equals half mass times velocity squared"
        intent_res = self.parser.parse(utterance)

        self.assertEqual(intent_res.intent, "DICTATION")
        self.assertFalse(intent_res.is_collision_risk)

        response = self.state_machine.apply(intent_res, raw_transcript=utterance)
        self.assertIn("kinetic energy equals half mass", response.buffer_state)

    def test_02_lexical_collision_immunity(self):
        """Verifies sentences containing command words (e.g. 'delete') are NOT executed as commands."""
        utterance = "the delete operation in red-black trees preserves balance through rotations"
        intent_res = self.parser.parse(utterance)

        self.assertEqual(intent_res.intent, "DICTATION")
        self.assertTrue(intent_res.is_collision_risk)

        response = self.state_machine.apply(intent_res, raw_transcript=utterance)
        # Buffer must preserve the sentence, not wipe the text
        self.assertIn("delete operation in red-black trees", response.buffer_state)

    def test_03_delete_last_word(self):
        """Verifies verbal backspacing removes only the preceding word."""
        self.state_machine.buffer = "the quicksort average case complexity evaluates to"
        
        command = "scratch that last word"
        intent_res = self.parser.parse(command)

        self.assertEqual(intent_res.intent, "EDIT_ACTION")
        self.assertEqual(intent_res.action_type, "DELETE_LAST_WORD")

        response = self.state_machine.apply(intent_res, raw_transcript=command)
        self.assertEqual(response.buffer_state.strip(), "the quicksort average case complexity evaluates")

    def test_04_delete_last_sentence(self):
        """Verifies deleting previous sentence preserves earlier paragraphs."""
        self.state_machine.buffer = "First question is completed. The velocity is negative."
        
        command = "delete the previous sentence"
        intent_res = self.parser.parse(command)

        self.assertEqual(intent_res.intent, "EDIT_ACTION")
        self.assertEqual(intent_res.action_type, "DELETE_LAST_SENTENCE")

        response = self.state_machine.apply(intent_res, raw_transcript=command)
        self.assertEqual(response.buffer_state.strip(), "First question is completed.")

    def test_05_targeted_span_substitution(self):
        """Verifies targeted slot extraction replaces specified tokens correctly."""
        self.state_machine.buffer = "the particle exhibits linear momentum in this system"
        
        command = "replace momentum with force"
        intent_res = self.parser.parse(command)

        self.assertEqual(intent_res.intent, "EDIT_ACTION")
        self.assertEqual(intent_res.action_type, "SUBSTITUTE_SPAN")
        self.assertEqual(intent_res.target_text.lower(), "momentum")
        self.assertEqual(intent_res.replacement_text.lower(), "force")

        response = self.state_machine.apply(intent_res, raw_transcript=command)
        self.assertIn("linear force in this system", response.buffer_state)


if __name__ == "__main__":
    unittest.main()