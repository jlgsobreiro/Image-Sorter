"""
Testes automatizados para validação do ImageSorter:
- Testes do banco de dados SQLite (db.py)
- Testes da varredura de diretórios e discos (scan_images.py)
- Testes da integração com o Ollama e tratamento de erros (describe_images.py)
"""
import os
import tempfile
import unittest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

from db import (
    init_db,
    insert_images_batch,
    get_pending_images,
    update_image_description,
    update_image_error,
    get_statistics,
    get_connection,
    get_known_people,
    register_known_person,
    rename_or_merge_known_person,
    get_person_images,
    update_person_face_crop,
    search_images,
    get_people_summary,
    update_image_hash,
    update_image_duplicate_status,
    get_duplicate_groups,
    get_unique_images
)
from scan_images import scan_and_save_images, interactive_scan_prompt
from describe_images import (
    encode_image_to_base64,
    query_ollama_vision,
    extract_people_from_response,
    extract_new_people_from_response,
    build_prompt_with_catalog,
    build_face_crop_recognition_prompt,
    recognize_face_crop_with_ai,
    process_images,
    reset_errors_to_pending,
    get_available_ollama_models,
    interactive_describe_prompt
)
from face_cropper import (
    parse_bounding_box,
    crop_face,
    ensure_face_crop_for_person,
    detect_faces_local,
    extract_face_crops_in_memory,
    encode_pil_to_base64
)
from interactive_namer import run_cli_namer
from interactive_menu import (
    run_interactive_cli,
    run_interactive_explorer,
    draw_face_boxes_on_image,
    InteractiveLiveEvaluatorGUI,
    InteractiveDashboardGUI,
    start_live_evaluator_gui,
    start_interactive_dashboard_gui
)
from duplicate_finder import (
    compute_file_hash,
    compute_perceptual_hash,
    compute_image_fingerprint,
    calculate_and_store_hashes,
    find_duplicates,
    set_keeper_in_duplicate_group,
    export_unique_images,
    select_directory_via_explorer,
    DuplicateReviewGUI
)
from PIL import Image


class TestDatabaseOperations(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_images.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_and_insert(self):
        init_db(self.db_path)
        records = [
            {"file_path": "/path/to/img1.jpg", "file_name": "img1.jpg", "file_size": 1024},
            {"file_path": "/path/to/img2.png", "file_name": "img2.png", "file_size": 2048},
        ]
        inserted = insert_images_batch(self.db_path, records)
        self.assertEqual(inserted, 2)

        # Inserção duplicada deve ser ignorada
        duplicate_records = [
            {"file_path": "/path/to/img1.jpg", "file_name": "img1.jpg", "file_size": 1024},
            {"file_path": "/path/to/img3.webp", "file_name": "img3.webp", "file_size": 4096},
        ]
        inserted_dup = insert_images_batch(self.db_path, duplicate_records)
        self.assertEqual(inserted_dup, 1)

        stats = get_statistics(self.db_path)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["pending"], 3)
        self.assertEqual(stats["processed"], 0)

    def test_update_status_and_errors(self):
        records = [
            {"file_path": "/path/to/img1.jpg", "file_name": "img1.jpg", "file_size": 100},
            {"file_path": "/path/to/img2.jpg", "file_name": "img2.jpg", "file_size": 200},
        ]
        insert_images_batch(self.db_path, records)
        pending = get_pending_images(self.db_path)
        self.assertEqual(len(pending), 2)

        # Atualizar sucesso com pessoas presentes
        update_image_description(
            self.db_path,
            image_id=pending[0]["id"],
            description="Uma foto com duas pessoas sorrindo no parque.",
            model_used="llama3.2-vision",
            people_present=json.dumps(["Pessoa 1", "Pessoa 2"])
        )

        # Atualizar erro
        update_image_error(
            self.db_path,
            image_id=pending[1]["id"],
            error_message="Conexão recusada"
        )

        stats = get_statistics(self.db_path)
        self.assertEqual(stats["processed"], 1)
        self.assertEqual(stats["errors"], 1)
        self.assertEqual(stats["pending"], 0)

        with get_connection(self.db_path) as conn:
            row = conn.cursor().execute("SELECT * FROM images WHERE id = ?", (pending[0]["id"],)).fetchone()
            self.assertEqual(row["people_present"], json.dumps(["Pessoa 1", "Pessoa 2"]))

        # Retentar erros
        reopened = reset_errors_to_pending(self.db_path)
        self.assertEqual(reopened, 1)
        stats_after = get_statistics(self.db_path)
        self.assertEqual(stats_after["pending"], 1)
        self.assertEqual(stats_after["errors"], 0)

    def test_schema_migration_adds_people_present_column(self):
        # Simula criação de tabela antiga sem a coluna people_present
        with get_connection(self.db_path) as conn:
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

        # Executa init_db que deve migrar a tabela adicionando a coluna
        init_db(self.db_path)

        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(images);")
            cols = [row["name"] for row in cursor.fetchall()]
            self.assertIn("people_present", cols)

    def test_known_people_registry(self):
        init_db(self.db_path)
        # Inserção inicial
        success1 = register_known_person(
            self.db_path,
            person_label="Pessoa 1",
            description="Homem de barba, óculos pretos, aproximadamente 35 anos",
            first_seen_image_id=1
        )
        self.assertTrue(success1)

        # Inserção duplicada deve falhar/retornar False
        success_dup = register_known_person(
            self.db_path,
            person_label="Pessoa 1",
            description="Outra descrição para Pessoa 1",
            first_seen_image_id=2
        )
        self.assertFalse(success_dup)

        # Inserir segunda pessoa
        success2 = register_known_person(
            self.db_path,
            person_label="Pessoa 2",
            description="Mulher de cabelos castanhos longos, aproximadamente 28 anos",
            first_seen_image_id=2
        )
        self.assertTrue(success2)

        known = get_known_people(self.db_path)
        self.assertEqual(len(known), 2)
        self.assertEqual(known[0]["person_label"], "Pessoa 1")
        self.assertEqual(known[1]["person_label"], "Pessoa 2")

        stats = get_statistics(self.db_path)
        self.assertEqual(stats["known_people"], 2)

    def test_rename_and_merge_known_person(self):
        init_db(self.db_path)
        # Insere duas imagens com pessoas
        records = [
            {"file_path": "/photos/pic1.jpg", "file_name": "pic1.jpg", "file_size": 100},
            {"file_path": "/photos/pic2.jpg", "file_name": "pic2.jpg", "file_size": 100},
            {"file_path": "/photos/pic3.jpg", "file_name": "pic3.jpg", "file_size": 100},
        ]
        insert_images_batch(self.db_path, records)

        update_image_description(self.db_path, 1, "Desc 1", "llama3.2-vision", people_present=json.dumps(["Pessoa 1"]))
        update_image_description(self.db_path, 2, "Desc 2", "llama3.2-vision", people_present=json.dumps(["Pessoa 2"]))
        update_image_description(self.db_path, 3, "Desc 3", "llama3.2-vision", people_present=json.dumps(["Pessoa 1", "Pessoa 2"]))

        register_known_person(self.db_path, "Pessoa 1", "Homem de óculos", first_seen_image_id=1)
        register_known_person(self.db_path, "Pessoa 2", "Homem com chapéu", first_seen_image_id=2)

        # 1. Renomeação simples: "Pessoa 1" -> "Carlos"
        res_rename = rename_or_merge_known_person(self.db_path, "Pessoa 1", "Carlos")
        self.assertEqual(res_rename["action"], "renamed")
        self.assertEqual(res_rename["affected_images"], 2) # pic1 e pic3

        people = get_known_people(self.db_path)
        labels = [p["person_label"] for p in people]
        self.assertIn("Carlos", labels)
        self.assertNotIn("Pessoa 1", labels)

        carlos_imgs = get_person_images(self.db_path, "Carlos")
        self.assertEqual(len(carlos_imgs), 2)

        # 2. Unificação (Merge): "Pessoa 2" era na verdade "Carlos"
        res_merge = rename_or_merge_known_person(self.db_path, "Pessoa 2", "Carlos")
        self.assertEqual(res_merge["action"], "merged")
        self.assertEqual(res_merge["affected_images"], 2) # pic2 e pic3

        people_after_merge = get_known_people(self.db_path)
        labels_after = [p["person_label"] for p in people_after_merge]
        self.assertIn("Carlos", labels_after)
        self.assertNotIn("Pessoa 2", labels_after)
        self.assertEqual(len(people_after_merge), 1)

        # Na imagem 3, que tinha ["Pessoa 1", "Pessoa 2"], agora deve ter apenas ["Carlos"] sem duplicação
        with get_connection(self.db_path) as conn:
            row3 = conn.cursor().execute("SELECT people_present FROM images WHERE id = 3").fetchone()
            self.assertEqual(json.loads(row3["people_present"]), ["Carlos"])

        all_carlos_imgs = get_person_images(self.db_path, "Carlos")
        self.assertEqual(len(all_carlos_imgs), 3)


class TestScanImages(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_images.db")

        # Cria estrutura de pastas aninhadas com arquivos
        self.scan_root = os.path.join(self.temp_dir.name, "media")
        os.makedirs(os.path.join(self.scan_root, "sub1", "sub2"), exist_ok=True)

        with open(os.path.join(self.scan_root, "photo1.JPG"), "wb") as f:
            f.write(b"fake-image-1")
        with open(os.path.join(self.scan_root, "sub1", "photo2.png"), "wb") as f:
            f.write(b"fake-image-2")
        with open(os.path.join(self.scan_root, "sub1", "sub2", "photo3.WEBP"), "wb") as f:
            f.write(b"fake-image-3")
        with open(os.path.join(self.scan_root, "document.pdf"), "wb") as f:
            f.write(b"fake-pdf")
        with open(os.path.join(self.scan_root, "notes.txt"), "wb") as f:
            f.write(b"fake-txt")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scan_finds_all_images_recursively(self):
        res = scan_and_save_images(
            root_path=self.scan_root,
            db_path=self.db_path,
            verbose=False
        )
        self.assertEqual(res["images_found"], 3)
        self.assertEqual(res["new_inserted"], 3)

        pending = get_pending_images(self.db_path)
        self.assertEqual(len(pending), 3)
        paths = [row["file_name"] for row in pending]
        self.assertIn("photo1.JPG", paths)
        self.assertIn("photo2.png", paths)
        self.assertIn("photo3.WEBP", paths)


class TestDescribeImages(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_images.db")

        self.img_file = os.path.join(self.temp_dir.name, "test.png")
        Image.new("RGB", (100, 100), color=(200, 100, 100)).save(self.img_file, format="PNG")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_base64_encoding(self):
        b64 = encode_image_to_base64(self.img_file)
        self.assertTrue(len(b64) > 0)

    @patch("urllib.request.urlopen")
    def test_query_ollama_vision(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "response": "Uma foto com fundo azul e texto claro."
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        desc = query_ollama_vision(
            image_path=self.img_file,
            model="llama3.2-vision",
            prompt="Descreva a imagem"
        )
        self.assertEqual(desc, "Uma foto com fundo azul e texto claro.")

    def test_extract_people_from_response(self):
        # 1. Formato padrão com seção PESSOAS_PRESENTES
        text_standard = """Esta é uma imagem em uma sala de reuniões.
Pessoa 1: homem de camisa azul, aproximadamente 35 anos.
Pessoa 2: mulher de blazer preto, sentada à direita.

PESSOAS_PRESENTES: ["Pessoa 1", "Pessoa 2"]
"""
        extracted = extract_people_from_response(text_standard)
        self.assertEqual(json.loads(extracted), ["Pessoa 1", "Pessoa 2"])

        # 2. Formato com lista vazia
        text_empty = """Paisagem montanhosa sem ninguém.
PESSOAS_PRESENTES: []"""
        extracted_empty = extract_people_from_response(text_empty)
        self.assertEqual(json.loads(extracted_empty), [])

        # 3. Formato textual de ausência
        text_none = "Foto de um carro. PESSOAS_PRESENTES: Nenhuma pessoa"
        extracted_none = extract_people_from_response(text_none)
        self.assertEqual(json.loads(extracted_none), [])

        # 4. Fallback baseado no texto quando a seção explícita não está presente
        text_fallback = "Na foto vemos Pessoa 1 ao lado de Pessoa 2 e mais adiante Pessoa 3."
        extracted_fallback = extract_people_from_response(text_fallback)
        self.assertEqual(json.loads(extracted_fallback), ["Pessoa 1", "Pessoa 2", "Pessoa 3"])

    def test_build_prompt_with_catalog(self):
        # 1. Catálogo vazio
        prompt_empty = build_prompt_with_catalog("Descreva a imagem", [])
        self.assertIn("Descreva a imagem", prompt_empty)
        self.assertIn("NOVO CATÁLOGO", prompt_empty)
        self.assertIn("Pessoa 1", prompt_empty)

        # 2. Catálogo preenchido
        catalog = [
            {"person_label": "Pessoa 1", "description": "Homem alto de óculos"},
            {"person_label": "Pessoa 2", "description": "Mulher com vestido azul"}
        ]
        prompt_with_cat = build_prompt_with_catalog("Descreva a imagem", catalog)
        self.assertIn("Pessoa 1: Homem alto de óculos", prompt_with_cat)
        self.assertIn("Pessoa 2: Mulher com vestido azul", prompt_with_cat)
        self.assertIn("Pessoa 3", prompt_with_cat)
        self.assertIn("A 'Pessoa 1' deve ser SEMPRE o mesmo indivíduo", prompt_with_cat)

    def test_extract_new_people_from_response(self):
        # 1. Formato estruturado JSON
        text_json = """Descrição detalhada.
PESSOAS_PRESENTES: ["Pessoa 1", "Pessoa 2"]
NOVAS_PESSOAS: [
    {"id": "Pessoa 1", "descricao": "Homem de barba ruiva e olhos verdes."},
    {"id": "Pessoa 2", "descricao": "Mulher jovem com cabelos crespos."}
]
"""
        new_people = extract_new_people_from_response(text_json, known_labels=["Pessoa 1"])
        self.assertEqual(len(new_people), 1)
        self.assertEqual(new_people[0]["id"], "Pessoa 2")
        self.assertIn("cabelos crespos", new_people[0]["descricao"])

        # 2. Fallback textual quando NOVAS_PESSOAS não é um JSON válido
        text_fallback = """
Pessoa 1: Homem de óculos.
Pessoa 2: Mulher com casaco vermelho e cabelos loiros.
PESSOAS_PRESENTES: ["Pessoa 1", "Pessoa 2"]
"""
        new_people_fallback = extract_new_people_from_response(text_fallback, known_labels=["Pessoa 1"])
        self.assertEqual(len(new_people_fallback), 1)
        self.assertEqual(new_people_fallback[0]["id"], "Pessoa 2")
        self.assertIn("casaco vermelho", new_people_fallback[0]["descricao"])

    @patch("urllib.request.urlopen")
    def test_process_images_workflow(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "response": "Descrição com pessoas.\nPessoa 1: homem sorrindo.\nPESSOAS_PRESENTES: [\"Pessoa 1\"]\nNOVAS_PESSOAS: [{\"id\": \"Pessoa 1\", \"descricao\": \"homem sorrindo\"}]"
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # Insere a imagem no banco
        insert_images_batch(self.db_path, [{
            "file_path": self.img_file,
            "file_name": "test.png",
            "file_size": 100
        }])

        stats = process_images(
            db_path=self.db_path,
            model="llama3.2-vision",
            verbose=False,
            use_face_detection=False
        )
        self.assertEqual(stats["processed"], 1)
        self.assertEqual(stats["errors"], 0)

        db_stats = get_statistics(self.db_path)
        self.assertEqual(db_stats["processed"], 1)
        self.assertEqual(db_stats["pending"], 0)
        self.assertEqual(db_stats["known_people"], 1)

        with get_connection(self.db_path) as conn:
            row = conn.cursor().execute("SELECT * FROM images WHERE id = 1").fetchone()
            self.assertIn("Pessoa 1", row["description"])
            self.assertEqual(row["people_present"], json.dumps(["Pessoa 1"]))
            self.assertEqual(row["model_used"], "llama3.2-vision")
            self.assertEqual(row["status"], "processed")

            person_row = conn.cursor().execute("SELECT * FROM known_people WHERE person_label = 'Pessoa 1'").fetchone()
            self.assertIsNotNone(person_row)
            self.assertEqual(person_row["first_seen_image_id"], 1)

    @patch("urllib.request.urlopen")
    def test_multi_image_person1_consistency(self, mock_urlopen):
        # Cria duas imagens
        img1 = os.path.join(self.temp_dir.name, "foto1.jpg")
        img2 = os.path.join(self.temp_dir.name, "foto2.jpg")
        Image.new("RGB", (200, 200), color=(100, 150, 200)).save(img1, format="JPEG")
        Image.new("RGB", (200, 200), color=(150, 200, 100)).save(img2, format="JPEG")

        insert_images_batch(self.db_path, [
            {"file_path": img1, "file_name": "foto1.jpg", "file_size": 50},
            {"file_path": img2, "file_name": "foto2.jpg", "file_size": 50}
        ])

        # Primeira chamada (foto 1) cadastra Pessoa 1
        resp1 = json.dumps({
            "response": "Foto de formatura.\nPessoa 1: Homem jovem de barba curta e terno azul.\nPESSOAS_PRESENTES: [\"Pessoa 1\"]\nNOVAS_PESSOAS: [{\"id\": \"Pessoa 1\", \"descricao\": \"Homem jovem de barba curta e terno azul.\"}]"
        }).encode("utf-8")

        # Segunda chamada (foto 2) reconhece Pessoa 1 e detecta Pessoa 2
        resp2 = json.dumps({
            "response": "Foto no parque.\nPessoa 1 está ao lado de Pessoa 2.\nPESSOAS_PRESENTES: [\"Pessoa 1\", \"Pessoa 2\"]\nNOVAS_PESSOAS: [{\"id\": \"Pessoa 2\", \"descricao\": \"Mulher de vestido floral.\"}]"
        }).encode("utf-8")

        mock_resp_obj1 = MagicMock()
        mock_resp_obj1.read.return_value = resp1
        mock_resp_obj2 = MagicMock()
        mock_resp_obj2.read.return_value = resp2

        mock_urlopen.return_value.__enter__.side_effect = [mock_resp_obj1, mock_resp_obj2]

        process_images(db_path=self.db_path, verbose=False, use_face_detection=False)

        known = get_known_people(self.db_path)
        self.assertEqual(len(known), 2)
        self.assertEqual(known[0]["person_label"], "Pessoa 1")
        self.assertEqual(known[1]["person_label"], "Pessoa 2")

        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            rows = cursor.execute("SELECT id, people_present FROM images ORDER BY id ASC").fetchall()
            self.assertEqual(rows[0]["people_present"], json.dumps(["Pessoa 1"]))
            self.assertEqual(rows[1]["people_present"], json.dumps(["Pessoa 1", "Pessoa 2"]))


class TestFaceCropperAndInteractiveNamer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_images.db")
        self.crops_dir = os.path.join(self.temp_dir.name, "crops")
        os.makedirs(self.crops_dir, exist_ok=True)

        # Cria uma imagem de teste real em RGB via PIL
        self.sample_img_path = os.path.join(self.temp_dir.name, "sample_person.jpg")
        img = Image.new("RGB", (800, 600), color=(120, 150, 200))
        img.save(self.sample_img_path, format="JPEG")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_bounding_box(self):
        # 1. Escala 0-1000
        bbox_1000 = [100, 200, 500, 600]
        parsed_1000 = parse_bounding_box(bbox_1000, img_width=1000, img_height=1000)
        self.assertEqual(parsed_1000, (200, 100, 600, 500))

        # 2. Escala 0.0-1.0
        bbox_float = [0.1, 0.2, 0.5, 0.6]
        parsed_float = parse_bounding_box(bbox_float, img_width=1000, img_height=1000)
        self.assertEqual(parsed_float, (200, 100, 600, 500))

        # 3. String JSON
        json_bbox = json.dumps({"box_2d": [100, 200, 500, 600]})
        parsed_json = parse_bounding_box(json_bbox, img_width=1000, img_height=1000)
        self.assertEqual(parsed_json, (200, 100, 600, 500))

    def test_crop_face_generation(self):
        out_crop = os.path.join(self.crops_dir, "test_face_crop.jpg")
        crop_res = crop_face(
            image_path=self.sample_img_path,
            bbox=[100, 100, 400, 400],
            output_path=out_crop
        )
        self.assertIsNotNone(crop_res)
        self.assertTrue(os.path.exists(out_crop))

        with Image.open(out_crop) as cropped:
            self.assertTrue(cropped.width > 0)
            self.assertTrue(cropped.height > 0)

    def test_ensure_face_crop_for_person(self):
        init_db(self.db_path)
        insert_images_batch(self.db_path, [{
            "file_path": self.sample_img_path,
            "file_name": "sample_person.jpg",
            "file_size": 1024
        }])

        register_known_person(
            self.db_path,
            person_label="Pessoa 1",
            description="Pessoa sorrindo",
            first_seen_image_id=1
        )

        crop_path = ensure_face_crop_for_person(
            self.db_path,
            person_label="Pessoa 1",
            crops_dir=self.crops_dir
        )
        self.assertIsNotNone(crop_path)
        self.assertTrue(os.path.exists(crop_path))

        # Verifica se o caminho foi persistido no banco
        people = get_known_people(self.db_path)
        self.assertEqual(people[0]["face_crop_path"], crop_path)

    @patch("builtins.input")
    def test_cli_namer_interactive_flow(self, mock_input):
        init_db(self.db_path)
        insert_images_batch(self.db_path, [{
            "file_path": self.sample_img_path,
            "file_name": "sample_person.jpg",
            "file_size": 1024
        }])
        update_image_description(self.db_path, 1, "Desc", "llama3.2-vision", people_present=json.dumps(["Pessoa 1"]))
        register_known_person(self.db_path, "Pessoa 1", "Pessoa de óculos", first_seen_image_id=1)

        # Simula usuário escolhendo opção 1 (digitar novo nome) e inserindo "Alice"
        mock_input.side_effect = ["1", "Alice", "q"]

        run_cli_namer(db_path=self.db_path, crops_dir=self.crops_dir)

        people = get_known_people(self.db_path)
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["person_label"], "Alice")

        carlos_imgs = get_person_images(self.db_path, "Alice")
        self.assertEqual(len(carlos_imgs), 1)


class TestInteractiveFeatures(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_images.db")
        init_db(self.db_path)

        # Cria arquivos fictícios para varredura
        self.scan_folder = os.path.join(self.temp_dir.name, "photos")
        os.makedirs(self.scan_folder, exist_ok=True)
        self.f1 = os.path.join(self.scan_folder, "f1.jpg")
        self.f2 = os.path.join(self.scan_folder, "f2.png")
        with open(self.f1, "wb") as f:
            f.write(b"fake image 1")
        with open(self.f2, "wb") as f:
            f.write(b"fake image 2")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_search_images_and_summary(self):
        insert_images_batch(self.db_path, [
            {"file_path": self.f1, "file_name": "f1.jpg", "file_size": 100},
            {"file_path": self.f2, "file_name": "f2.png", "file_size": 200}
        ])
        update_image_description(
            self.db_path, 1, "Foto na praia com Alice", "model-x", people_present=json.dumps(["Alice"])
        )
        register_known_person(self.db_path, "Alice", "Pessoa na praia", first_seen_image_id=1)

        # Teste de busca por termo
        results = search_images(self.db_path, query_text="praia")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)

        # Teste de busca por pessoa
        res_person = search_images(self.db_path, person_label="Alice")
        self.assertEqual(len(res_person), 1)

        # Teste de resumo de pessoas
        summary = get_people_summary(self.db_path)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["person_label"], "Alice")
        self.assertEqual(summary[0]["photo_count"], 1)

    @patch("builtins.input")
    def test_interactive_scan_prompt(self, mock_input):
        # Simula: caminho self.scan_folder, opção 1 (todas as extensões padrão), lote 500
        mock_input.side_effect = [self.scan_folder, "1", "500"]

        result = interactive_scan_prompt(db_path=self.db_path)
        self.assertIsNotNone(result)
        self.assertEqual(result["images_found"], 2)
        self.assertEqual(result["new_inserted"], 2)

    @patch("urllib.request.urlopen")
    def test_get_available_ollama_models(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "models": [{"name": "llama3.2-vision:latest"}, {"name": "llava:13b"}]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        models = get_available_ollama_models()
        self.assertEqual(len(models), 2)
        self.assertIn("llama3.2-vision:latest", models)

    @patch("describe_images.get_available_ollama_models")
    @patch("describe_images.process_images")
    @patch("builtins.input")
    def test_interactive_describe_prompt(self, mock_input, mock_process, mock_models):
        insert_images_batch(self.db_path, [
            {"file_path": self.f1, "file_name": "f1.jpg", "file_size": 100}
        ])
        mock_models.return_value = ["llama3.2-vision:latest"]
        mock_process.return_value = {"processed": 1, "errors": 0, "total": 1}

        # Simula: modo 1 (IA completa), escolha modelo 1, limite 10, prompt 1 (padrão)
        mock_input.side_effect = ["1", "1", "10", "1"]

        interactive_describe_prompt(db_path=self.db_path)
        mock_process.assert_called_once()
        self.assertTrue(mock_process.call_args.kwargs.get("describe_ai"))

    @patch("describe_images.process_images")
    @patch("builtins.input")
    def test_interactive_describe_prompt_biometric_only(self, mock_input, mock_process):
        insert_images_batch(self.db_path, [
            {"file_path": self.f1, "file_name": "f1.jpg", "file_size": 100}
        ])
        mock_process.return_value = {"processed": 1, "errors": 0, "total": 1}

        # Simula: modo 2 (apenas biometria facial, sem IA), limite 10
        mock_input.side_effect = ["2", "10"]

        interactive_describe_prompt(db_path=self.db_path)
        mock_process.assert_called_once()
        self.assertFalse(mock_process.call_args.kwargs.get("describe_ai"))

    @patch("builtins.input")
    def test_interactive_cli_menu_exit(self, mock_input):
        # Simula escolha 0 (sair)
        mock_input.side_effect = ["0"]
        # Não deve lançar exceção
        run_interactive_cli(db_path=self.db_path)

    @patch("builtins.input")
    def test_interactive_explorer_listing(self, mock_input):
        register_known_person(self.db_path, "Bob", "Descrição do Bob")
        # Simula escolha 1 (listar) e depois 0 (voltar)
        mock_input.side_effect = ["1", "0"]
        run_interactive_explorer(db_path=self.db_path)


class TestFaceDetectionAndInMemoryRecognition(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_faces.db")
        init_db(self.db_path)

        # Cria imagem sintética
        self.img_path = os.path.join(self.temp_dir.name, "face_sample.jpg")
        img = Image.new("RGB", (300, 300), color=(200, 200, 200))
        img.save(self.img_path, format="JPEG")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_encode_pil_to_base64(self):
        img = Image.new("RGB", (100, 100), color="blue")
        b64 = encode_pil_to_base64(img)
        self.assertIsInstance(b64, str)
        self.assertTrue(len(b64) > 50)

    def test_extract_face_crops_in_memory_with_provided_bbox(self):
        crops = extract_face_crops_in_memory(
            self.img_path,
            bboxes=[(50, 50, 150, 150)]
        )
        self.assertEqual(len(crops), 1)
        crop = crops[0]
        self.assertIn("base64", crop)
        self.assertIn("pil_image", crop)
        self.assertIn("image_bytes", crop)
        self.assertIn("box_2d_norm", crop)
        self.assertEqual(crop["original_face_bbox"], (50, 50, 150, 150))
        self.assertTrue(len(crop["base64"]) > 0)
        self.assertIsInstance(crop["pil_image"], Image.Image)

    def test_build_face_crop_recognition_prompt(self):
        # Prompt de descrição fisionômica objetiva para recorte facial
        prompt_new = build_face_crop_recognition_prompt([], next_id_num=1, face_idx=1, total_faces=1)
        self.assertIn("Rosto 1 de 1", prompt_new)
        self.assertIn("description", prompt_new)

    @patch("describe_images.query_ollama_vision")
    def test_recognize_face_crop_with_ai_existing_person(self, mock_query):
        mock_query.return_value = json.dumps({
            "description": "Rosto oval, barba castanha curta, óculos retangulares e pele clara."
        })
        known = [{"person_label": "Pessoa 1", "description": "Homem de óculos"}]
        res = recognize_face_crop_with_ai("fake_base64_data", known_people=known)

        self.assertEqual(res["person_label"], "Pessoa 2")
        self.assertTrue(res["is_new"])
        self.assertIn("barba castanha", res["description"])

    @patch("describe_images.query_ollama_vision")
    def test_recognize_face_crop_with_ai_new_person(self, mock_query):
        mock_query.return_value = json.dumps({
            "description": "Mulher jovem com cabelo loiro comprido e olhos azuis."
        })
        known = [{"person_label": "Pessoa 1", "description": "Homem de óculos"}]
        res = recognize_face_crop_with_ai("fake_base64_data", known_people=known, next_id_num=2)

        self.assertEqual(res["person_label"], "Pessoa 2")
        self.assertTrue(res["is_new"])
        self.assertIn("cabelo loiro", res["description"])

    @patch("describe_images.FaceIdentityMatcher")
    @patch("describe_images.extract_face_crops_in_memory")
    @patch("describe_images.query_ollama_vision")
    def test_process_images_with_face_detection_pipeline(self, mock_query, mock_crops, mock_matcher):
        insert_images_batch(self.db_path, [
            {"file_path": self.img_path, "file_name": "face_sample.jpg", "file_size": 1024}
        ])

        # Simula 1 rosto detectado pelo algoritmo de visão computacional
        mock_crops.return_value = [{
            "bbox": (20, 20, 120, 120),
            "original_face_bbox": (30, 30, 110, 110),
            "pil_image": Image.new("RGB", (100, 100)),
            "base64": "fake_b64",
            "image_bytes": b"fake_bytes",
            "box_2d_norm": [100, 100, 400, 400]
        }]
        mock_matcher.return_value.match.return_value = {
            "person_label": None, "status": "new", "score": None
        }

        # Chamada única à IA para descrever o cenário geral da imagem com pessoas identificadas
        mock_query.return_value = "Um homem em uma sala iluminada olhando para a câmera."

        result = process_images(
            db_path=self.db_path,
            use_face_detection=True
        )

        self.assertEqual(result["processed"], 1)
        self.assertEqual(result["errors"], 0)
        self.assertEqual(mock_query.call_count, 1)

        # Verifica se cadastrou a pessoa no catálogo e na imagem
        people = get_known_people(self.db_path)
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0]["person_label"], "Pessoa 1")

        with get_connection(self.db_path) as conn:
            row = conn.cursor().execute("SELECT people_present, description FROM images WHERE id = 1").fetchone()
            self.assertEqual(json.loads(row["people_present"]), ["Pessoa 1"])
            self.assertIn("homem em uma sala", row["description"])

    @patch("describe_images.extract_face_crops_in_memory")
    @patch("describe_images.query_ollama_vision")
    def test_process_images_skips_when_no_faces_detected(self, mock_query, mock_crops):
        insert_images_batch(self.db_path, [
            {"file_path": self.img_path, "file_name": "no_face.jpg", "file_size": 1024}
        ])

        # Simula 0 rostos detectados pelo algoritmo de visão computacional
        mock_crops.return_value = []

        result = process_images(
            db_path=self.db_path,
            use_face_detection=True
        )

        self.assertEqual(result["processed"], 1)
        self.assertEqual(result["errors"], 0)

        # A IA multimodal NÃO deve ser chamada quando não há rostos
        mock_query.assert_not_called()

        with get_connection(self.db_path) as conn:
            row = conn.cursor().execute("SELECT people_present, description, status, model_used FROM images WHERE id = 1").fetchone()
            self.assertEqual(json.loads(row["people_present"]), [])
            self.assertIn("Nenhum rosto detectado", row["description"])
            self.assertEqual(row["status"], "processed")
            self.assertEqual(row["model_used"], "face_detection_skip")


class TestLiveVisualEvaluator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_live.db")
        init_db(self.db_path)

        self.img_path = os.path.join(self.temp_dir.name, "sample_live.jpg")
        img = Image.new("RGB", (200, 200), color=(150, 150, 150))
        img.save(self.img_path, format="JPEG")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_draw_face_boxes_on_image(self):
        base_img = Image.new("RGB", (300, 300), color="white")
        face_crops = [{
            "original_face_bbox": (30, 30, 120, 120),
            "person_label": "Pessoa 1"
        }]
        annotated = draw_face_boxes_on_image(base_img, face_crops)
        self.assertIsInstance(annotated, Image.Image)
        self.assertEqual(annotated.size, (300, 300))

    @patch("describe_images.FaceIdentityMatcher")
    @patch("describe_images.extract_face_crops_in_memory")
    @patch("describe_images.query_ollama_vision")
    def test_process_images_with_progress_callback(self, mock_query, mock_crops, mock_matcher):
        insert_images_batch(self.db_path, [
            {"file_path": self.img_path, "file_name": "sample_live.jpg", "file_size": 1024}
        ])

        mock_crops.return_value = [{
            "bbox": (10, 10, 80, 80),
            "original_face_bbox": (10, 10, 80, 80),
            "pil_image": Image.new("RGB", (70, 70)),
            "base64": "fake_b64",
            "image_bytes": b"fake_bytes",
            "box_2d_norm": [50, 50, 400, 400]
        }]
        mock_matcher.return_value.match.return_value = {
            "person_label": None, "status": "new", "score": None
        }

        mock_query.return_value = "Uma pessoa sentada em um ambiente claro."

        received_events = []
        def callback(evt):
            received_events.append(evt)

        res = process_images(
            db_path=self.db_path,
            use_face_detection=True,
            progress_callback=callback
        )

        self.assertEqual(res["processed"], 1)
        event_types = [e["event"] for e in received_events]
        self.assertIn("start_image", event_types)
        self.assertIn("faces_detected", event_types)
        self.assertIn("face_recognized", event_types)
        self.assertIn("image_completed", event_types)
        self.assertIn("finished", event_types)

    def test_gui_event_handling(self):
        try:
            import tkinter as tk
            gui = InteractiveLiveEvaluatorGUI(db_path=self.db_path)
            gui.root.withdraw()

            # Simula evento start_image
            gui._handle_worker_event({
                "event": "start_image",
                "image_id": 1,
                "file_path": self.img_path,
                "index": 1,
                "total": 1
            })
            self.assertIn("sample_live.jpg", gui.lbl_img_info.cget("text"))

            # Simula evento faces_detected
            gui._handle_worker_event({
                "event": "faces_detected",
                "image_id": 1,
                "file_path": self.img_path,
                "face_crops": [{
                    "pil_image": Image.new("RGB", (60, 60)),
                    "person_label": "Pessoa 1",
                    "bbox": (10, 10, 50, 50)
                }]
            })
            self.assertEqual(gui.lbl_faces_count.cget("text"), "Rostos detectados: 1")

            # Simula evento image_completed
            gui._handle_worker_event({
                "event": "image_completed",
                "image_id": 1,
                "file_path": self.img_path,
                "description": "Foto de teste da IA.",
                "people_present": json.dumps(["Pessoa 1"]),
                "duration": 1.25,
                "processed_count": 1,
                "error_count": 0,
                "total": 1
            })
            self.assertIn("Foto de teste da IA", gui.txt_ai_output.get("1.0", tk.END))

            gui.root.destroy()
        except tk.TclError:
            # Em ambientes headless sem display X11/Tk
            pass


class TestDuplicateFinderAndExporter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_duplicates.db")
        init_db(self.db_path)

        # Cria imagens de teste (2 imagens com o mesmo conteúdo idêntico e 1 diferente)
        self.img1_path = os.path.join(self.temp_dir.name, "photo_a.jpg")
        self.img2_path = os.path.join(self.temp_dir.name, "photo_a_copy.jpg")
        self.img3_path = os.path.join(self.temp_dir.name, "photo_b.jpg")

        # Imagem A com padrão diagonal
        img_a = Image.new("RGB", (100, 100), color=(255, 255, 255))
        for x in range(50):
            for y in range(50):
                img_a.putpixel((x, y), (0, 0, 0))
        img_a.save(self.img1_path, format="JPEG")
        img_a.save(self.img2_path, format="JPEG")

        # Imagem B com padrão vertical
        img_b = Image.new("RGB", (100, 100), color=(255, 255, 255))
        for x in range(50, 100):
            for y in range(100):
                img_b.putpixel((x, y), (128, 128, 128))
        img_b.save(self.img3_path, format="JPEG")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_hashes(self):
        h1 = compute_file_hash(self.img1_path)
        h2 = compute_file_hash(self.img2_path)
        h3 = compute_file_hash(self.img3_path)

        self.assertIsNotNone(h1)
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)

        dh1 = compute_perceptual_hash(self.img1_path)
        dh2 = compute_perceptual_hash(self.img2_path)
        dh3 = compute_perceptual_hash(self.img3_path)

        self.assertIsNotNone(dh1)
        self.assertEqual(dh1, dh2)
        self.assertNotEqual(dh1, dh3)

    def test_calculate_and_store_hashes(self):
        records = [
            {"file_path": self.img1_path, "file_name": "photo_a.jpg", "file_size": 1000},
            {"file_path": self.img2_path, "file_name": "photo_a_copy.jpg", "file_size": 1000},
            {"file_path": self.img3_path, "file_name": "photo_b.jpg", "file_size": 1000},
        ]
        insert_images_batch(self.db_path, records)

        res = calculate_and_store_hashes(self.db_path, method="sha256")
        self.assertEqual(res["processed"], 3)
        self.assertEqual(res["errors"], 0)
        self.assertEqual(res["duplicate_groups"], 1)
        self.assertEqual(res["duplicate_images_total"], 2)

        groups = find_duplicates(self.db_path)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["count"], 2)
        self.assertEqual(len(groups[0]["images"]), 2)

    def test_set_keeper_in_duplicate_group(self):
        records = [
            {"file_path": self.img1_path, "file_name": "photo_a.jpg", "file_size": 1000},
            {"file_path": self.img2_path, "file_name": "photo_a_copy.jpg", "file_size": 1000},
        ]
        insert_images_batch(self.db_path, records)
        calculate_and_store_hashes(self.db_path)

        groups = find_duplicates(self.db_path)
        images = groups[0]["images"]
        keeper_id = images[1]["id"]  # Escolhe a segunda como keeper

        set_keeper_in_duplicate_group(self.db_path, keeper_id, images)

        with get_connection(self.db_path) as conn:
            row_keeper = conn.execute("SELECT is_duplicate FROM images WHERE id = ?", (keeper_id,)).fetchone()
            other_id = images[0]["id"]
            row_other = conn.execute("SELECT is_duplicate FROM images WHERE id = ?", (other_id,)).fetchone()

            self.assertEqual(row_keeper["is_duplicate"], 0)
            self.assertEqual(row_other["is_duplicate"], 1)

    def test_get_unique_images(self):
        records = [
            {"file_path": self.img1_path, "file_name": "photo_a.jpg", "file_size": 1000},
            {"file_path": self.img2_path, "file_name": "photo_a_copy.jpg", "file_size": 1000},
            {"file_path": self.img3_path, "file_name": "photo_b.jpg", "file_size": 1000},
        ]
        insert_images_batch(self.db_path, records)
        calculate_and_store_hashes(self.db_path)

        unique_imgs = get_unique_images(self.db_path)
        # Deve retornar apenas 2 imagens únicas (1 de photo_a e 1 de photo_b)
        self.assertEqual(len(unique_imgs), 2)
        unique_paths = [u["file_path"] for u in unique_imgs]
        self.assertIn(self.img3_path, unique_paths)
        self.assertTrue(self.img1_path in unique_paths or self.img2_path in unique_paths)

    def test_export_unique_images_workflow(self):
        records = [
            {"file_path": self.img1_path, "file_name": "photo_a.jpg", "file_size": 1000},
            {"file_path": self.img2_path, "file_name": "photo_a_copy.jpg", "file_size": 1000},
            {"file_path": self.img3_path, "file_name": "photo_b.jpg", "file_size": 1000},
        ]
        insert_images_batch(self.db_path, records)

        export_dir = os.path.join(self.temp_dir.name, "exported_uniques")

        events = []
        def cb(evt):
            events.append(evt)

        stats = export_unique_images(
            db_path=self.db_path,
            destination_dir=export_dir,
            open_explorer_on_complete=False,
            progress_callback=cb
        )

        self.assertEqual(stats["status"], "success")
        self.assertEqual(stats["exported_count"], 2)
        self.assertEqual(stats["duplicates_avoided"], 1)

        # Verifica que exatamente 2 arquivos existem na pasta de exportação
        exported_files = os.listdir(export_dir)
        self.assertEqual(len(exported_files), 2)

        evt_types = [e["event"] for e in events]
        self.assertIn("export_start", evt_types)
        self.assertIn("export_completed", evt_types)

    def test_export_unique_images_filename_collision(self):
        # Cria outra foto com mesmo nome base mas em subpasta diferente
        sub_dir = os.path.join(self.temp_dir.name, "sub")
        os.makedirs(sub_dir, exist_ok=True)
        img4_path = os.path.join(sub_dir, "photo_b.jpg")
        img4 = Image.new("RGB", (100, 100), color=(0, 0, 255))
        img4.save(img4_path, format="JPEG")

        records = [
            {"file_path": self.img3_path, "file_name": "photo_b.jpg", "file_size": 1000},
            {"file_path": img4_path, "file_name": "photo_b.jpg", "file_size": 1000},
        ]
        insert_images_batch(self.db_path, records)

        export_dir = os.path.join(self.temp_dir.name, "exported_collision")
        stats = export_unique_images(
            db_path=self.db_path,
            destination_dir=export_dir,
            open_explorer_on_complete=False
        )

        self.assertEqual(stats["exported_count"], 2)
        exported_files = os.listdir(export_dir)
        self.assertEqual(len(exported_files), 2)
        self.assertIn("photo_b.jpg", exported_files)
        self.assertIn("photo_b_2.jpg", exported_files)

    def test_duplicate_review_gui_init(self):
        try:
            import tkinter as tk
            records = [
                {"file_path": self.img1_path, "file_name": "photo_a.jpg", "file_size": 1000},
                {"file_path": self.img2_path, "file_name": "photo_a_copy.jpg", "file_size": 1000},
            ]
            insert_images_batch(self.db_path, records)
            calculate_and_store_hashes(self.db_path)

            gui = DuplicateReviewGUI(db_path=self.db_path)
            gui.root.withdraw()
            self.assertEqual(len(gui.groups), 1)
            self.assertIn("Grupo 1 de 1", gui.lbl_group_info.cget("text"))

            gui._next_group()
            gui._prev_group()
            gui.root.destroy()
        except tk.TclError:
            pass


class TestProgressBarAndInterruptControls(unittest.TestCase):
    """Testes automatizados para barras de progresso e controle de interrupção."""

    def setUp(self):
        import threading
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_prog.db")
        self.threading = threading

        # Cria imagens para testes de varredura e hash
        self.images_dir = os.path.join(self.temp_dir.name, "imgs")
        os.makedirs(self.images_dir, exist_ok=True)
        self.test_img_paths = []
        for i in range(5):
            p = os.path.join(self.images_dir, f"test_{i}.jpg")
            img = Image.new("RGB", (60, 60), color=(i * 20, 50, 100))
            img.save(p, format="JPEG")
            self.test_img_paths.append(p)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scan_and_save_images_progress_callback(self):
        events = []
        def cb(e):
            events.append(e)

        res = scan_and_save_images(
            root_path=self.images_dir,
            db_path=self.db_path,
            progress_callback=cb
        )

        self.assertEqual(res["images_found"], 5)
        self.assertFalse(res["interrupted"])
        ev_types = [e["event"] for e in events]
        self.assertIn("scan_start", ev_types)
        self.assertIn("scan_completed", ev_types)

    def test_scan_and_save_images_stop_event(self):
        stop_evt = self.threading.Event()
        stop_evt.set()  # Já começa com sinal de interrupção

        res = scan_and_save_images(
            root_path=self.images_dir,
            db_path=self.db_path,
            stop_event=stop_evt
        )

        self.assertTrue(res["interrupted"])

    def test_calculate_hashes_with_stop_event(self):
        init_db(self.db_path)
        records = [
            {"file_path": p, "file_name": os.path.basename(p), "file_size": 100}
            for p in self.test_img_paths
        ]
        insert_images_batch(self.db_path, records)

        stop_evt = self.threading.Event()
        stop_evt.set()

        res = calculate_and_store_hashes(self.db_path, stop_event=stop_evt)
        self.assertTrue(res["interrupted"])
        self.assertEqual(res["processed"], 0)

    def test_export_unique_images_with_stop_event(self):
        init_db(self.db_path)
        records = [
            {"file_path": p, "file_name": os.path.basename(p), "file_size": 100}
            for p in self.test_img_paths
        ]
        insert_images_batch(self.db_path, records)

        stop_evt = self.threading.Event()
        stop_evt.set()

        export_dir = os.path.join(self.temp_dir.name, "out_stop")
        res = export_unique_images(
            db_path=self.db_path,
            destination_dir=export_dir,
            open_explorer_on_complete=False,
            stop_event=stop_evt
        )

        self.assertTrue(res["interrupted"])
        self.assertEqual(res["status"], "cancelled")

    def test_dashboard_gui_widgets_and_interrupt(self):
        try:
            import tkinter as tk
            gui = InteractiveDashboardGUI(db_path=self.db_path)
            gui.root.withdraw()

            self.assertIsNotNone(gui.progressbar)
            self.assertIsNotNone(gui.lbl_progress_status)
            self.assertIsNotNone(gui.btn_stop_op)
            self.assertIsNotNone(gui.lbl_progress_pct)

            # Testa acionamento de interrupção
            gui.is_busy = True
            gui.btn_stop_op.configure(state=gui.tk.NORMAL)
            gui.stop_current_operation()
            self.assertTrue(gui.stop_event.is_set())
            self.assertIn("Solicitando interrupção", gui.lbl_progress_status.cget("text"))

            gui.root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    unittest.main()
