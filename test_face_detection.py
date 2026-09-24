"""Testes do detector local, coordenadas e tratamento de falhas sem Ollama."""
import io
import hashlib
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import face_cropper
from db import init_db, insert_images_batch, get_statistics
from describe_images import process_images


class TestDetectionFailures(unittest.TestCase):
    def test_detection_failure_is_not_saved_as_no_faces(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "images.db")
            image_path = Path(directory) / "image.jpg"
            Image.new("RGB", (100, 100)).save(image_path)
            init_db(db_path)
            insert_images_batch(db_path, [{
                "file_path": str(image_path), "file_name": image_path.name, "file_size": 100
            }])
            with patch("describe_images.extract_face_crops_in_memory", side_effect=RuntimeError("detector indisponível")), \
                    patch("describe_images.query_ollama_vision") as query:
                result = process_images(db_path=db_path)
            self.assertEqual(result["errors"], 1)
            self.assertEqual(get_statistics(db_path)["processed"], 0)
            query.assert_not_called()


class TestYuNetDetection(unittest.TestCase):
    def test_bundled_model_integrity(self):
        digest = hashlib.sha256(face_cropper.YUNET_MODEL_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4")

    def test_model_cache_is_thread_local(self):
        with patch.object(face_cropper, "_YUNET_LOCAL", threading.local()):
            detector = face_cropper._get_yunet_detector()
            self.assertIsNotNone(detector)
            self.assertIs(face_cropper._get_yunet_detector(), detector)
            with ThreadPoolExecutor(max_workers=1) as executor:
                other = executor.submit(face_cropper._get_yunet_detector).result()
            self.assertIsNotNone(other)
            self.assertIsNot(other, detector)

    @patch("face_cropper._get_yunet_detector")
    def test_multiscale_coordinates_clipping_and_nms(self, get_detector):
        detector = get_detector.return_value
        detector.detect.side_effect = [
            (1, np.array([[32, 16, 64, 64, *([0] * 10), 0.95]], dtype=np.float32)),
            (1, np.array([
                [64, 32, 128, 128, *([0] * 10), 0.94],
                [1250, 600, 80, 80, *([0] * 10), 0.98],
                [400, 20, 100, 100, *([0] * 10), 0.2],
                [700, 20, 1, 1, *([0] * 10), 0.99]
            ], dtype=np.float32))
        ]
        boxes = face_cropper.detect_faces_local(Image.new("RGB", (2000, 1000)))
        self.assertEqual(boxes, [(100, 50, 300, 250), (1953, 937, 2000, 1000)])
        self.assertEqual(detector.setInputSize.call_args_list[0].args, ((640, 320),))
        self.assertEqual(detector.setInputSize.call_args_list[1].args, ((1280, 640),))

    @patch("face_cropper._get_cascade")
    @patch("face_cropper._get_yunet_detector")
    def test_empty_neural_result_does_not_add_haar_false_positives(self, get_detector, cascade):
        get_detector.return_value.detect.return_value = (0, None)
        self.assertEqual(face_cropper.detect_faces_local(Image.new("RGB", (100, 100))), [])
        cascade.assert_not_called()

    @patch("face_cropper._get_cascade")
    @patch("face_cropper._get_yunet_detector", return_value=None)
    def test_missing_model_uses_haar(self, get_detector, get_cascade):
        get_cascade.return_value.detectMultiScale.return_value = [(10, 20, 40, 50)]
        self.assertEqual(face_cropper.detect_faces_local(Image.new("RGB", (100, 100))), [(10, 20, 50, 70)])

    @patch("face_cropper._get_cascade")
    @patch("face_cropper._get_yunet_detector")
    def test_broken_neural_detector_uses_haar(self, get_detector, get_cascade):
        get_detector.return_value.detect.side_effect = RuntimeError("modelo inválido")
        get_cascade.return_value.detectMultiScale.return_value = [(10, 20, 40, 50)]
        self.assertEqual(face_cropper.detect_faces_local(Image.new("RGB", (100, 100))), [(10, 20, 50, 70)])

    @patch("face_cropper._get_cascade", return_value=None)
    @patch("face_cropper._get_yunet_detector", return_value=None)
    def test_no_detector_raises(self, get_detector, get_cascade):
        with self.assertRaises(RuntimeError):
            face_cropper.extract_face_crops_in_memory(Image.new("RGB", (100, 100)))

    def test_invalid_image_raises(self):
        with self.assertRaises(RuntimeError):
            face_cropper.extract_face_crops_in_memory(b"invalid image")

    @patch("face_cropper._get_cascade", side_effect=AssertionError("YuNet não deve usar fallback neste teste"))
    def test_real_model_on_blank_inputs(self, cascade):
        self.assertIsNotNone(face_cropper._get_yunet_detector())
        image = Image.new("RGB", (200, 100))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        for value in [image, buffer.getvalue(), np.zeros((100, 200), dtype=np.uint8),
                      np.zeros((100, 200, 4), dtype=np.uint8), Image.new("RGB", (1, 1))]:
            with self.subTest(input_type=type(value)):
                self.assertEqual(face_cropper.detect_faces_local(value), [])

    @patch("face_cropper._get_cascade", side_effect=AssertionError("YuNet não deve usar fallback neste teste"))
    def test_real_model_detects_face_and_crops(self, cascade):
        image_path = Path(__file__).parent / "test_data" / "astronaut.png"
        crops = face_cropper.extract_face_crops_in_memory(str(image_path))
        # Rosto da astronauta na região superior central da fotografia pública da NASA.
        matches = [crop for crop in crops if
                   170 <= crop["original_face_bbox"][0] <= 200 and
                   50 <= crop["original_face_bbox"][1] <= 90]
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0]["base64"])
        self.assertLessEqual(max(matches[0]["pil_image"].size), 400)

    @patch("face_cropper._get_cascade", side_effect=AssertionError("YuNet não deve usar fallback neste teste"))
    def test_real_model_finds_two_smaller_faces_in_large_image(self, cascade):
        image_path = Path(__file__).parent / "test_data" / "astronaut.png"
        with Image.open(image_path) as source:
            small = source.convert("RGB").resize((256, 256))
        group = Image.new("RGB", (1600, 900))
        group.paste(small, (100, 100))
        group.paste(small, (1100, 500))
        boxes = face_cropper.detect_faces_local(group)
        self.assertEqual(len(boxes), 2)
        for box, (offset_x, offset_y) in zip(boxes, [(100, 100), (1100, 500)]):
            left, top, right, bottom = box
            self.assertTrue(offset_x + 80 <= left <= offset_x + 110)
            self.assertTrue(offset_y + 20 <= top <= offset_y + 50)
            self.assertGreater(right - left, 30)
            self.assertGreater(bottom - top, 30)


if __name__ == "__main__":
    unittest.main()