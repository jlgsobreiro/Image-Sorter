"""Testes unitários dedicados para o módulo de reconhecimento facial OpenCV SFace."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from face_cropper import extract_face_crops_in_memory
from face_recognition import FaceIdentityMatcher, SFACE_MODEL_PATH, SFACE_SHA256


class TestFaceIdentityMatcher(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.astronaut_path = Path(__file__).resolve().parent / "test_data" / "astronaut.png"
        self.assertTrue(self.astronaut_path.is_file(), "Imagem de teste astronaut.png necessária.")

    def test_init_parameter_validation(self):
        matcher = FaceIdentityMatcher(threshold=0.6, margin=0.15)
        self.assertEqual(matcher.threshold, 0.6)
        self.assertEqual(matcher.margin, 0.15)

        with self.assertRaises(ValueError):
            FaceIdentityMatcher(threshold=1.5)
        with self.assertRaises(ValueError):
            FaceIdentityMatcher(threshold=float("nan"))
        with self.assertRaises(ValueError):
            FaceIdentityMatcher(margin=-0.1)
        with self.assertRaises(ValueError):
            FaceIdentityMatcher(margin=float("inf"))

    def test_real_sface_match_with_astronaut(self):
        matcher = FaceIdentityMatcher(threshold=0.5, margin=0.1)
        crops = extract_face_crops_in_memory(str(self.astronaut_path))
        self.assertGreaterEqual(len(crops), 1, "YuNet deve encontrar o rosto na imagem astronaut.png")
        crop_pil = crops[0]["pil_image"]

        # Salva o recorte de referência em pasta temporária
        ref_path = Path(self.temp_dir.name) / "astronaut_face.jpg"
        crop_pil.save(ref_path, "JPEG")

        known_people = [{
            "person_label": "Astronauta",
            "face_crop_path": str(ref_path),
            "description": "Retrato de astronauta com capacete ao lado."
        }]

        # 1. Match exato com o mesmo recorte
        res = matcher.match(crop=crop_pil, known_people=known_people)
        self.assertEqual(res["status"], "matched")
        self.assertEqual(res["person_label"], "Astronauta")
        self.assertGreater(res["score"], 0.85)

        # 2. Match com rótulo excluído (já presente na mesma foto)
        res_excluded = matcher.match(
            crop=crop_pil,
            known_people=known_people,
            excluded_labels=["Astronauta"]
        )
        self.assertEqual(res_excluded["status"], "ambiguous")
        self.assertIsNone(res_excluded["person_label"])

        # 3. Imagem sem pessoas cadastradas
        res_empty = matcher.match(crop=crop_pil, known_people=[])
        self.assertEqual(res_empty["status"], "new")
        self.assertIsNone(res_empty["person_label"])

        # 4. Recorte de baixa qualidade / sem rosto (imagem em branco)
        blank_crop = Image.new("RGB", (60, 60), "black")
        res_blank = matcher.match(crop=blank_crop, known_people=known_people)
        self.assertEqual(res_blank["status"], "low_quality")
        self.assertIsNone(res_blank["score"])

    def test_ambiguity_within_margin(self):
        matcher = FaceIdentityMatcher(threshold=0.5, margin=0.1)
        crops = extract_face_crops_in_memory(str(self.astronaut_path))
        crop_pil = crops[0]["pil_image"]

        ref1 = Path(self.temp_dir.name) / "ref1.jpg"
        ref2 = Path(self.temp_dir.name) / "ref2.jpg"
        crop_pil.save(ref1, "JPEG")
        crop_pil.save(ref2, "JPEG")

        known_people = [
            {"person_label": "Pessoa A", "face_crop_path": str(ref1)},
            {"person_label": "Pessoa B", "face_crop_path": str(ref2)},
        ]

        # Ambas as referências têm pontuações idênticas -> deve retornar ambiguous
        res = matcher.match(crop=crop_pil, known_people=known_people)
        self.assertEqual(res["status"], "ambiguous")
        self.assertIsNone(res["person_label"])

    def test_cache_invalidation_on_file_change(self):
        matcher = FaceIdentityMatcher()
        crops = extract_face_crops_in_memory(str(self.astronaut_path))
        crop_pil = crops[0]["pil_image"]

        ref_path = Path(self.temp_dir.name) / "ref_dynamic.jpg"
        crop_pil.save(ref_path, "JPEG")

        known_people = [{"person_label": "Pessoa X", "face_crop_path": str(ref_path)}]
        res1 = matcher.match(crop=crop_pil, known_people=known_people)
        self.assertEqual(res1["status"], "matched")

        # Sobrescreve a referência com imagem em branco
        Image.new("RGB", (80, 80), "white").save(ref_path, "JPEG")
        # Garante atualização do stat (mtime)
        os.utime(ref_path, (os.path.getatime(ref_path) + 10, os.path.getmtime(ref_path) + 10))

        res2 = matcher.match(crop=crop_pil, known_people=known_people)
        self.assertEqual(res2["status"], "new")


if __name__ == "__main__":
    unittest.main()
