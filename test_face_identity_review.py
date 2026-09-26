"""Testes da sugestão local de pares de pessoas para revisão manual."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from db import (
    get_known_people,
    init_db,
    insert_images_batch,
    register_known_person,
    update_image_description,
)
from face_identity_review import (
    find_similar_people,
    get_people_cooccurrence_and_counts,
    run_cli_similar_people_review,
)
from face_recognition import FaceIdentityMatcher


class TestSimilarPeopleSuggestions(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db_path = str(Path(self.directory.name) / "images.db")
        init_db(self.db_path)
        for label in ("Ana", "Bia", "Caio"):
            register_known_person(
                self.db_path,
                label,
                f"Descrição de {label}",
                face_crop_path=f"{label}.jpg"
            )

    def test_suggestions_are_filtered_sorted_and_do_not_merge(self):
        scores = {
            frozenset(("Ana", "Bia")): 0.49,
            frozenset(("Ana", "Caio")): 0.72,
            frozenset(("Bia", "Caio")): 0.34,
        }

        with patch("face_identity_review.FaceIdentityMatcher") as matcher_type:
            matcher_type.return_value.compare_people.side_effect = lambda first, second: scores[
                frozenset((first["person_label"], second["person_label"]))
            ]
            candidates = find_similar_people(self.db_path, min_similarity=0.35)

        self.assertEqual(
            [(item["first"]["person_label"], item["second"]["person_label"]) for item in candidates],
            [("Ana", "Caio"), ("Ana", "Bia")]
        )
        self.assertEqual([item["score"] for item in candidates], [0.72, 0.49])
        self.assertEqual(candidates[0]["confidence"], "Alta")
        self.assertEqual(candidates[1]["confidence"], "Média")
        self.assertEqual({person["person_label"] for person in get_known_people(self.db_path)}, {"Ana", "Bia", "Caio"})

    def test_cooccurrence_and_filtering(self):
        insert_images_batch(self.db_path, [
            {"file_path": "img1.jpg", "file_name": "img1.jpg", "file_size": 100},
            {"file_path": "img2.jpg", "file_name": "img2.jpg", "file_size": 100},
        ])
        update_image_description(self.db_path, 1, "Foto 1", "test", people_present='["Ana", "Bia"]')
        update_image_description(self.db_path, 2, "Foto 2", "test", people_present='["Ana"]')

        cooccur, counts = get_people_cooccurrence_and_counts(self.db_path)
        self.assertIn(frozenset(("Ana", "Bia")), cooccur)
        self.assertEqual(counts.get("Ana"), 2)
        self.assertEqual(counts.get("Bia"), 1)

        scores = {
            frozenset(("Ana", "Bia")): 0.80,
            frozenset(("Ana", "Caio")): 0.70,
            frozenset(("Bia", "Caio")): 0.20,
        }
        with patch("face_identity_review.FaceIdentityMatcher") as matcher_type:
            matcher_type.return_value.compare_people.side_effect = lambda first, second: scores[
                frozenset((first["person_label"], second["person_label"]))
            ]
            all_candidates = find_similar_people(self.db_path, min_similarity=0.35, filter_cooccurring=False)
            filtered_candidates = find_similar_people(self.db_path, min_similarity=0.35, filter_cooccurring=True)

        self.assertEqual(len(all_candidates), 2)
        self.assertTrue(all_candidates[0]["co_occurs"])
        self.assertEqual(len(filtered_candidates), 1)
        self.assertEqual(filtered_candidates[0]["first"]["person_label"], "Ana")
        self.assertEqual(filtered_candidates[0]["second"]["person_label"], "Caio")

    def test_invalid_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            find_similar_people(self.db_path, min_similarity=float("nan"))

    def test_people_without_face_crops_are_not_compared(self):
        db_path = str(Path(self.directory.name) / "no_crops.db")
        init_db(db_path)
        register_known_person(db_path, "Sem recorte A", "Descrição A")
        register_known_person(db_path, "Sem recorte B", "Descrição B")
        with patch("face_identity_review.FaceIdentityMatcher") as matcher_type:
            self.assertEqual(find_similar_people(db_path), [])
        matcher_type.assert_not_called()

    def test_matcher_returns_canonical_cosine_similarity(self):
        matcher = FaceIdentityMatcher()
        matcher._np = np
        references = {
            "first.jpg": np.array([1.0, 0.0]),
            "second.jpg": np.array([0.6, 0.8]),
        }
        first = {"face_crop_path": "first.jpg"}
        second = {"face_crop_path": "second.jpg"}
        with patch.object(matcher, "_load_models"), patch.object(
            matcher, "_reference_embedding", side_effect=references.get
        ):
            self.assertAlmostEqual(matcher.compare_people(first, second), 0.6)

        with patch.object(matcher, "_load_models"), patch.object(
            matcher, "_reference_embedding", side_effect=[references["first.jpg"], None]
        ):
            self.assertIsNone(matcher.compare_people(first, second))

    def test_cli_review_runs_without_error(self):
        with patch("face_identity_review.find_similar_people", return_value=[]), \
             patch("builtins.input", return_value=""):
            run_cli_similar_people_review(self.db_path)


if __name__ == "__main__":
    unittest.main()
