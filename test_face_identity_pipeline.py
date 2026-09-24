"""Regressões de identidade: texto do Ollama não é uma comparação biométrica."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from db import get_connection, get_known_people, init_db, insert_images_batch, register_known_person
from describe_images import process_images, recognize_face_crop_with_ai
from face_cropper import extract_face_crops_in_memory


class TestFaceDescriptionSafety(unittest.TestCase):
    @patch("describe_images.query_ollama_vision")
    def test_model_cannot_assign_existing_identity(self, query):
        query.return_value = json.dumps({
            "person_label": "Pessoa 7", "description": "Rosto oval com óculos."
        })
        known = [{"person_label": "Pessoa 7", "description": "Rosto oval com óculos."}]
        result = recognize_face_crop_with_ai("crop", known)
        self.assertEqual(result["person_label"], "Pessoa 8")
        self.assertTrue(result["is_new"])
        self.assertEqual(result["description"], "Rosto oval com óculos.")
        self.assertNotIn("Pessoa 7", query.call_args.kwargs["prompt"])

    @patch("describe_images.query_ollama_vision")
    def test_invalid_description_does_not_invent_profile(self, query):
        for response in ["", "Não é Pessoa 1", "{}", '{"description": []}', '{"description": "  "}']:
            with self.subTest(response=response):
                query.return_value = response
                with self.assertRaises(ValueError):
                    recognize_face_crop_with_ai("crop", [])


class TestFaceIdentityPipeline(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db_path = str(Path(directory.name) / "images.db")
        self.image_path = Path(directory.name) / "photo.jpg"
        Image.new("RGB", (200, 200), "white").save(self.image_path)
        self.crops = extract_face_crops_in_memory(str(self.image_path), [(40, 40, 120, 120)])
        init_db(self.db_path)
        insert_images_batch(self.db_path, [{
            "file_path": str(self.image_path), "file_name": self.image_path.name, "file_size": 100
        }])
        self.events = []

    def run_pipeline(self, match, responses):
        with patch("describe_images.FaceIdentityMatcher") as matcher, \
                patch("describe_images.extract_face_crops_in_memory", return_value=self.crops), \
                patch("describe_images.query_ollama_vision", side_effect=responses) as query:
            matcher.return_value.match.return_value = match
            result = process_images(self.db_path, progress_callback=self.events.append)
        return result, matcher, query

    def test_visual_match_reuses_identity_without_face_ollama_call(self):
        register_known_person(self.db_path, "Ana", "Rosto oval.")
        result, matcher, query = self.run_pipeline(
            {"person_label": "Ana", "status": "matched", "score": 0.82}, ["Uma pessoa no jardim."])
        self.assertEqual(result["processed"], 1)
        self.assertEqual(query.call_count, 1)
        self.assertEqual(len(get_known_people(self.db_path)), 1)
        self.assertEqual(matcher.return_value.match.call_args.kwargs["excluded_labels"], [])
        event = next(e for e in self.events if e["event"] == "face_recognized")
        self.assertEqual(event["match_status"], "matched")
        self.assertEqual(event["match_score"], 0.82)
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM images").fetchone()
        self.assertEqual(json.loads(row["people_present"]), ["Ana"])
        self.assertEqual(row["description"], "Uma pessoa no jardim.")

    def test_ambiguous_match_creates_provisional_not_model_identity(self):
        register_known_person(self.db_path, "Pessoa 9", "Rosto oval.")
        result, _, query = self.run_pipeline(
            {"person_label": None, "status": "ambiguous", "score": 0.65},
            ["Uma pessoa no parque."])
        self.assertEqual(result["processed"], 1)
        self.assertEqual(query.call_count, 1)
        people = get_known_people(self.db_path)
        self.assertEqual({p["person_label"] for p in people}, {"Pessoa 9", "Pessoa 10"})
        new_person = next(p for p in people if p["person_label"] == "Pessoa 10")
        self.assertTrue(Path(new_person["face_crop_path"]).is_file())
        self.assertTrue(Path(new_person["face_crop_path"]).is_relative_to(Path(self.db_path).parent))
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM images").fetchone()
        self.assertEqual(json.loads(row["people_present"]), ["Pessoa 10"])
        self.assertEqual(row["description"], "Uma pessoa no parque.")

    def test_model_failure_is_error_not_new_person(self):
        with patch("describe_images.FaceIdentityMatcher") as matcher, \
                patch("describe_images.extract_face_crops_in_memory", return_value=self.crops), \
                patch("describe_images.query_ollama_vision") as query:
            matcher.return_value.match.side_effect = RuntimeError("SFace indisponível")
            result = process_images(self.db_path)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(get_known_people(self.db_path), [])
        query.assert_not_called()

    def test_cancel_after_description_does_not_register_person(self):
        stop = threading.Event()
        def describe(**kwargs):
            stop.set()
            return '{"description": "Rosto oval."}'
        with patch("describe_images.FaceIdentityMatcher") as matcher, \
                patch("describe_images.extract_face_crops_in_memory", return_value=self.crops), \
                patch("describe_images.query_ollama_vision", side_effect=describe):
            matcher.return_value.match.return_value = {"person_label": None, "status": "new", "score": None}
            result = process_images(self.db_path, stop_event=stop)
        self.assertEqual(result["processed"], 0)
        self.assertEqual(get_known_people(self.db_path), [])

    def test_process_images_without_ai_description_runs_biometrics_only(self):
        with patch("describe_images.FaceIdentityMatcher") as matcher, \
                patch("describe_images.extract_face_crops_in_memory", return_value=self.crops), \
                patch("describe_images.query_ollama_vision") as query:
            matcher.return_value.match.return_value = {"person_label": None, "status": "new", "score": None}
            result = process_images(self.db_path, describe_ai=False)

        self.assertEqual(result["processed"], 1)
        self.assertEqual(result["errors"], 0)
        query.assert_not_called()

        people = get_known_people(self.db_path)
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["person_label"], "Pessoa 1")

        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM images").fetchone()
        self.assertEqual(json.loads(row["people_present"]), ["Pessoa 1"])
        self.assertIn("biométrico", row["description"])
        self.assertEqual(row["model_used"], "biometric_sface")

    def test_process_images_without_ai_and_without_face_detect_marks_processed(self):
        with patch("describe_images.query_ollama_vision") as query:
            result = process_images(self.db_path, use_face_detection=False, describe_ai=False)

        self.assertEqual(result["processed"], 1)
        query.assert_not_called()

        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT * FROM images").fetchone()
        self.assertEqual(row["status"], "processed")
        self.assertEqual(row["model_used"], "none")
        self.assertEqual(row["people_present"], "[]")


if __name__ == "__main__":
    unittest.main()