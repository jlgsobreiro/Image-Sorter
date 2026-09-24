"""
Testes unitários automatizados para o classificador de tipos de imagem com IA
(IA Local Integrada ultrarrápida e Ollama VLM) e exportação exclusiva de fotos reais.
"""
import os
import io
import json
import shutil
import tempfile
import sqlite3
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from db import (
    init_db,
    insert_images_batch,
    get_statistics,
    update_image_type,
    get_images_for_classification,
    get_image_type_statistics,
    get_real_photos,
    update_image_hash,
    update_image_duplicate_status,
    get_all_images
)
from image_classifier import (
    FastImageClassifier,
    classify_image_fast,
    encode_image_for_classification,
    parse_classification_response,
    classify_image_with_ai,
    classify_images_batch,
    export_real_photos,
    run_interactive_classification_cli
)
from photo_viewer import PhotoViewerGUI


class TestImageClassifierDB(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_images.db")
        init_db(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_schema_and_migration_image_type(self):
        # Cria um banco antigo sem as colunas adicionadas por migração
        old_db = os.path.join(self.temp_dir, "old.db")
        conn = sqlite3.connect(old_db)
        conn.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                file_size INTEGER,
                status TEXT DEFAULT 'pending',
                description TEXT,
                model_used TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP
            );
        """)
        conn.commit()
        conn.close()

        # Executa init_db para disparar migração transparente
        init_db(old_db)

        conn = sqlite3.connect(old_db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(images);")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()

        self.assertIn("image_type", columns)
        self.assertIn("image_hash", columns)
        self.assertIn("is_duplicate", columns)
        self.assertIn("people_present", columns)

    def test_update_and_get_image_types(self):
        insert_images_batch(self.db_path, [
            {"file_path": "/img1.jpg", "file_name": "img1.jpg", "file_size": 100},
            {"file_path": "/img2.png", "file_name": "img2.png", "file_size": 200},
            {"file_path": "/img3.ico", "file_name": "img3.ico", "file_size": 50},
            {"file_path": "/img4.txt", "file_name": "img4.txt", "file_size": 80},
        ])

        update_image_type(self.db_path, 1, "photo")
        update_image_type(self.db_path, 2, "screenshot")
        update_image_type(self.db_path, 3, "icon_or_graphic")
        # img4 permanece unclassified

        stats = get_image_type_statistics(self.db_path)
        self.assertEqual(stats["photo"], 1)
        self.assertEqual(stats["screenshot"], 1)
        self.assertEqual(stats["icon_or_graphic"], 1)
        self.assertEqual(stats["unclassified"], 1)
        self.assertEqual(stats["total"], 4)

        # get_images_for_classification
        unclassified = get_images_for_classification(self.db_path, force=False)
        self.assertEqual(len(unclassified), 1)
        self.assertEqual(unclassified[0]["id"], 4)

        all_imgs = get_images_for_classification(self.db_path, force=True)
        self.assertEqual(len(all_imgs), 4)

    def test_tolerant_type_normalization(self):
        insert_images_batch(self.db_path, [
            {"file_path": "/test_norm.jpg", "file_name": "test_norm.jpg", "file_size": 100}
        ])
        # Normalização de strings variantes
        update_image_type(self.db_path, 1, "Fotografia Real")
        row = get_all_images(self.db_path)[0]
        self.assertEqual(row["image_type"], "photo")

        update_image_type(self.db_path, 1, "Print")
        row = get_all_images(self.db_path)[0]
        self.assertEqual(row["image_type"], "screenshot")

        update_image_type(self.db_path, 1, "Ícone")
        row = get_all_images(self.db_path)[0]
        self.assertEqual(row["image_type"], "icon_or_graphic")

    def test_get_real_photos_only_unique(self):
        insert_images_batch(self.db_path, [
            {"file_path": "/p1.jpg", "file_name": "p1.jpg", "file_size": 100},
            {"file_path": "/p1_dup.jpg", "file_name": "p1_dup.jpg", "file_size": 100},
            {"file_path": "/p2.jpg", "file_name": "p2.jpg", "file_size": 120},
            {"file_path": "/s1.png", "file_name": "s1.png", "file_size": 300},
        ])

        update_image_type(self.db_path, 1, "photo")
        update_image_type(self.db_path, 2, "photo")
        update_image_type(self.db_path, 3, "photo")
        update_image_type(self.db_path, 4, "screenshot")

        # p1 e p1_dup têm o mesmo hash
        update_image_hash(self.db_path, 1, "hash_aaa")
        update_image_hash(self.db_path, 2, "hash_aaa")
        update_image_duplicate_status(self.db_path, 2, 1)  # duplicata confirmada

        update_image_hash(self.db_path, 3, "hash_bbb")

        # 1. only_unique=True -> deve retornar p1 e p2 (2 fotos únicas), descartando p1_dup e s1
        unique_photos = get_real_photos(self.db_path, only_unique=True)
        self.assertEqual(len(unique_photos), 2)
        unique_ids = {p["id"] for p in unique_photos}
        self.assertIn(1, unique_ids)
        self.assertIn(3, unique_ids)
        self.assertNotIn(2, unique_ids)
        self.assertNotIn(4, unique_ids)

        # 2. only_unique=False -> retorna todas as 3 fotos
        all_photos = get_real_photos(self.db_path, only_unique=False)
        self.assertEqual(len(all_photos), 3)


class TestFastLocalClassifier(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # 1. Ícone com canal alfa
        self.icon_path = os.path.join(self.temp_dir, "icon.png")
        icon_img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw_ico = ImageDraw.Draw(icon_img)
        draw_ico.ellipse((10, 10, 54, 54), fill=(255, 120, 0, 255))
        icon_img.save(self.icon_path)

        # 2. Screenshot com resolução de tela e elementos de UI
        self.screenshot_path = os.path.join(self.temp_dir, "screenshot.png")
        ss_img = Image.new("RGB", (1920, 1080), (240, 240, 240))
        draw_ss = ImageDraw.Draw(ss_img)
        draw_ss.rectangle((0, 0, 1920, 40), fill=(50, 50, 50))
        draw_ss.rectangle((0, 40, 250, 1080), fill=(220, 220, 220))
        for y in range(80, 800, 30):
            draw_ss.line((280, y, 1100, y), fill=(20, 20, 20), width=2)
        ss_img.save(self.screenshot_path)

        # 3. Foto real com gradientes e variações naturais
        self.photo_path = os.path.join(self.temp_dir, "photo.jpg")
        arr = np.zeros((400, 600, 3), dtype=np.uint8)
        x = np.linspace(0, 1, 600)
        y = np.linspace(0, 1, 400)
        xv, yv = np.meshgrid(x, y)
        arr[:, :, 0] = np.clip(np.sin(xv * 3.14) * 200 + 30 + np.random.normal(0, 5, (400, 600)), 0, 255).astype(np.uint8)
        arr[:, :, 1] = np.clip(np.cos(yv * 3.14) * 180 + 50 + np.random.normal(0, 5, (400, 600)), 0, 255).astype(np.uint8)
        arr[:, :, 2] = np.clip((xv + yv) * 100 + 40 + np.random.normal(0, 5, (400, 600)), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(self.photo_path, quality=95)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_classify_icon(self):
        res = classify_image_fast(self.icon_path)
        self.assertEqual(res["image_type"], "icon_or_graphic")
        self.assertGreaterEqual(res["confidence"], 0.85)
        self.assertEqual(res["engine"], "fast_local")

    def test_classify_screenshot(self):
        res = classify_image_fast(self.screenshot_path)
        self.assertEqual(res["image_type"], "screenshot")
        self.assertGreaterEqual(res["confidence"], 0.70)

    def test_classify_photo(self):
        res = classify_image_fast(self.photo_path)
        self.assertEqual(res["image_type"], "photo")
        self.assertGreaterEqual(res["confidence"], 0.70)

    def test_classify_pil_image_in_memory(self):
        img = Image.open(self.photo_path)
        res = FastImageClassifier.classify(img)
        self.assertEqual(res["image_type"], "photo")

    def test_classify_nonexistent_file(self):
        res = classify_image_fast("/caminho/inexistente_12345.jpg")
        self.assertEqual(res["image_type"], "other")
        self.assertEqual(res["confidence"], 0.0)


class TestImageClassifierAI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_img_path = os.path.join(self.temp_dir, "sample.jpg")
        img = Image.new("RGB", (200, 200), color="blue")
        img.save(self.test_img_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_encode_image_for_classification(self):
        b64 = encode_image_for_classification(self.test_img_path, max_dimension=100)
        self.assertIsNotNone(b64)
        self.assertTrue(len(b64) > 50)

        # Imagem RGBA
        rgba = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        b64_rgba = encode_image_for_classification(rgba)
        self.assertIsNotNone(b64_rgba)

    def test_parse_classification_response_json(self):
        resp1 = 'Aqui está a classificação:\n```json\n{"image_type": "photo", "confidence": 0.98, "reason": "Foto de praia."}\n```'
        parsed1 = parse_classification_response(resp1)
        self.assertEqual(parsed1["image_type"], "photo")
        self.assertAlmostEqual(parsed1["confidence"], 0.98)
        self.assertEqual(parsed1["reason"], "Foto de praia.")

        resp2 = '{"image_type": "screenshot", "confidence": 0.92, "reason": "Print de tela de celular."}'
        parsed2 = parse_classification_response(resp2)
        self.assertEqual(parsed2["image_type"], "screenshot")

        resp3 = '{"image_type": "icon_or_graphic", "confidence": 0.95, "reason": "Ícone de aplicativo com fundo transparente."}'
        parsed3 = parse_classification_response(resp3)
        self.assertEqual(parsed3["image_type"], "icon_or_graphic")

    def test_parse_classification_response_fallback(self):
        resp_photo = "Esta imagem parece ser uma fotografia real com câmera digital."
        self.assertEqual(parse_classification_response(resp_photo)["image_type"], "photo")

        resp_screen = "Identificado como captura de tela do WhatsApp."
        self.assertEqual(parse_classification_response(resp_screen)["image_type"], "screenshot")

        resp_icon = "Trata-se de um ícone de software vetorial."
        self.assertEqual(parse_classification_response(resp_icon)["image_type"], "icon_or_graphic")

        self.assertEqual(parse_classification_response("")["image_type"], "other")

    @patch("urllib.request.urlopen")
    def test_classify_image_with_ai_ollama(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "response": '```json\n{"image_type": "photo", "confidence": 0.96, "reason": "Retrato de pessoa."}\n```'
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        res = classify_image_with_ai(self.test_img_path, engine="ollama", model="llava")
        self.assertEqual(res["image_type"], "photo")
        self.assertAlmostEqual(res["confidence"], 0.96)
        self.assertEqual(res["engine"], "ollama")

    def test_classify_image_with_ai_auto_routes_to_fast_local(self):
        res = classify_image_with_ai(self.test_img_path, engine="auto", model="fast")
        self.assertEqual(res["engine"], "fast_local")
        self.assertIn(res["image_type"], ["photo", "screenshot", "icon_or_graphic", "other"])


class TestBatchClassificationAndExport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        init_db(self.db_path)

        # Cria imagens no disco
        self.img1 = os.path.join(self.temp_dir, "foto_viagem.jpg")
        self.img2 = os.path.join(self.temp_dir, "print_zap.png")
        self.img3 = os.path.join(self.temp_dir, "icone_app.png")

        # 1. Foto com cores naturais
        arr = np.zeros((300, 300, 3), dtype=np.uint8)
        arr[:, :, 0] = np.random.randint(50, 200, (300, 300))
        arr[:, :, 1] = np.random.randint(40, 180, (300, 300))
        arr[:, :, 2] = np.random.randint(30, 220, (300, 300))
        Image.fromarray(arr).save(self.img1)

        # 2. Print de tela
        ss = Image.new("RGB", (1920, 1080), (250, 250, 250))
        draw_ss = ImageDraw.Draw(ss)
        draw_ss.rectangle((0, 0, 1920, 40), fill=(30, 30, 30))
        for y in range(80, 600, 30):
            draw_ss.line((50, y, 900, y), fill=(0, 0, 0), width=2)
        ss.save(self.img2)

        # 3. Ícone com transparência
        ico = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw_ico = ImageDraw.Draw(ico)
        draw_ico.ellipse((10, 10, 54, 54), fill=(255, 0, 0, 255))
        ico.save(self.img3)

        insert_images_batch(self.db_path, [
            {"file_path": self.img1, "file_name": "foto_viagem.jpg", "file_size": 500},
            {"file_path": self.img2, "file_name": "print_zap.png", "file_size": 400},
            {"file_path": self.img3, "file_name": "icone_app.png", "file_size": 100},
        ])

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_classify_images_batch_fast_local(self):
        events = []
        summary = classify_images_batch(
            db_path=self.db_path,
            engine="fast_local",
            progress_callback=lambda d: events.append(d)
        )

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["classified_count"], 3)
        self.assertGreaterEqual(summary["photos_found"] + summary["screenshots_found"] + summary["icons_found"], 2)

        # Verifica banco de dados
        stats = get_image_type_statistics(self.db_path)
        self.assertEqual(stats["unclassified"], 0)
        self.assertEqual(stats["total"], 3)

        # Verifica eventos disparados
        event_types = [e.get("event") for e in events]
        self.assertIn("classification_start", event_types)
        self.assertIn("image_classification_complete", event_types)
        self.assertIn("classification_finished", event_types)

    @patch("image_classifier.open_folder_in_explorer")
    def test_export_real_photos(self, mock_open_folder):
        mock_open_folder.return_value = True

        # Marca imagem 1 como foto real e imagem 2 como print
        update_image_type(self.db_path, 1, "photo")
        update_image_type(self.db_path, 2, "screenshot")
        update_image_type(self.db_path, 3, "icon_or_graphic")

        dest_dir = os.path.join(self.temp_dir, "fotos_exportadas")
        events = []

        summary = export_real_photos(
            db_path=self.db_path,
            destination_dir=dest_dir,
            only_unique=True,
            open_explorer_on_complete=True,
            progress_callback=lambda d: events.append(d)
        )

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["exported_count"], 1)
        self.assertTrue(os.path.exists(os.path.join(dest_dir, "foto_viagem.jpg")))
        self.assertFalse(os.path.exists(os.path.join(dest_dir, "print_zap.png")))
        self.assertFalse(os.path.exists(os.path.join(dest_dir, "icone_app.png")))
        mock_open_folder.assert_called_once()


if __name__ == "__main__":
    unittest.main()
