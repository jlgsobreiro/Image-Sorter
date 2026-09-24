"""
Testes automatizados para o visualizador de fotos (PhotoViewerGUI) e nomeação manual de pessoas.
"""
import os
import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from db import (
    init_db,
    insert_images_batch,
    get_connection,
    get_known_people,
    register_known_person,
    update_image_people_present,
    get_all_images
)
from photo_viewer import (
    PhotoViewerGUI,
    start_photo_viewer_gui,
    run_cli_photo_viewer,
    open_file_in_system,
    rotate_image_file,
    create_manual_face_crop
)
from interactive_namer import InteractiveNamerGUI, start_interactive_namer
from face_cropper import ensure_face_crop_for_person


class TestPhotoViewerAndNamer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = os.path.join(self.temp_dir.name, "test_viewer.db")
        self.crops_dir = os.path.join(self.temp_dir.name, "face_crops")
        os.makedirs(self.crops_dir, exist_ok=True)
        init_db(self.db_path)

        # Cria imagens de teste sintéticas
        self.images_data = []
        for i in range(1, 4):
            img_path = os.path.join(self.temp_dir.name, f"test_img_{i}.jpg")
            with Image.new("RGB", (200, 200), color=(50 * i, 70, 90)) as img:
                img.save(img_path)
            self.images_data.append({
                "file_path": img_path,
                "file_name": f"test_img_{i}.jpg",
                "file_size": os.path.getsize(img_path)
            })

        insert_images_batch(self.db_path, self.images_data)

    def _create_viewer_gui(self, initial_image_id=None, filter_person=None, parent_root=None):
        gui = PhotoViewerGUI.__new__(PhotoViewerGUI)

        def cleanup():
            if hasattr(gui, "root"):
                try:
                    gui.root.destroy()
                except tk.TclError:
                    pass

        self.addCleanup(cleanup)
        gui.__init__(
            db_path=self.db_path,
            initial_image_id=initial_image_id,
            filter_person=filter_person,
            parent_root=parent_root
        )
        gui.root.withdraw()
        return gui

    def _create_namer_gui(self, parent_root=None):
        gui = InteractiveNamerGUI.__new__(InteractiveNamerGUI)

        def cleanup():
            if hasattr(gui, "root"):
                try:
                    gui.root.destroy()
                except tk.TclError:
                    pass

        self.addCleanup(cleanup)
        gui.__init__(
            db_path=self.db_path,
            crops_dir=self.crops_dir,
            parent_root=parent_root
        )
        gui.root.withdraw()
        return gui

    def test_get_all_images_helper(self):
        all_imgs = get_all_images(self.db_path)
        self.assertEqual(len(all_imgs), 3)

        update_image_people_present(self.db_path, 1, ["Alice", "Bob"])
        alice_imgs = get_all_images(self.db_path, person_label="Alice")
        self.assertEqual(len(alice_imgs), 1)
        self.assertEqual(alice_imgs[0]["id"], 1)

    def test_photo_viewer_gui_initialization_and_navigation(self):
        gui = self._create_viewer_gui()
        self.assertEqual(len(gui.images_list), 3)
        self.assertEqual(gui.current_idx, 0)
        self.assertIn("Foto 1 de 3", gui.lbl_counter.cget("text"))

        # Navega para próxima foto
        gui._next_image()
        self.assertEqual(gui.current_idx, 1)
        self.assertIn("Foto 2 de 3", gui.lbl_counter.cget("text"))

        # Navega para foto anterior
        gui._prev_image()
        self.assertEqual(gui.current_idx, 0)

    def test_photo_viewer_gui_with_parent_root(self):
        parent = tk.Tk()
        parent.withdraw()
        self.addCleanup(parent.destroy)

        gui = self._create_viewer_gui(parent_root=parent)
        self.assertTrue(gui.is_toplevel)
        self.assertIsInstance(gui.root, tk.Toplevel)

    def test_photo_viewer_assign_face_name_locally_and_globally(self):
        update_image_people_present(self.db_path, 1, ["Pessoa 1"])
        gui = self._create_viewer_gui(initial_image_id=1)

        # Atribui novo nome "Carlos" apenas localmente
        gui._assign_face_name(
            image_id=1,
            face_idx=0,
            new_name="Carlos",
            old_name="Pessoa 1",
            crop_data=None
        )

        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT people_present FROM images WHERE id = 1").fetchone()
            pp = json.loads(row["people_present"])
            self.assertEqual(pp, ["Carlos"])

        # Testa remoção de pessoa
        gui._remove_person_from_image(1, "Carlos")
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT people_present FROM images WHERE id = 1").fetchone()
            pp = json.loads(row["people_present"])
            self.assertEqual(pp, [])

    def test_photo_viewer_description_edit(self):
        gui = self._create_viewer_gui(initial_image_id=2)
        gui.txt_description.delete("1.0", tk.END)
        gui.txt_description.insert("1.0", "Nova descrição personalizada da foto.")

        with patch.object(gui.messagebox, "showinfo"):
            gui._save_description_edit()

        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT description FROM images WHERE id = 2").fetchone()
            self.assertEqual(row["description"], "Nova descrição personalizada da foto.")

    def test_photo_viewer_detect_faces_on_image(self):
        gui = self._create_viewer_gui(initial_image_id=1)
        dummy_crops = [{
            "bbox": (10, 10, 50, 50),
            "original_face_bbox": (10, 10, 50, 50),
            "pil_image": Image.new("RGB", (40, 40), "pink"),
            "base64": "",
            "image_bytes": b"fake",
            "box_2d_norm": [50, 50, 250, 250]
        }]

        with patch("photo_viewer.extract_face_crops_in_memory", return_value=dummy_crops), \
             patch("photo_viewer.FaceIdentityMatcher") as mock_matcher, \
             patch.object(gui.messagebox, "showinfo"):
            mock_matcher.return_value.match.return_value = {"person_label": None, "status": "new", "score": None}
            gui._detect_faces_on_current_image()

        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT people_present FROM images WHERE id = 1").fetchone()
            pp = json.loads(row["people_present"])
            self.assertEqual(len(pp), 1)

    def test_interactive_namer_crop_rendering_and_parent_root(self):
        parent = tk.Tk()
        parent.withdraw()
        self.addCleanup(parent.destroy)

        # Registra pessoa e atualiza crop
        crop_file = os.path.join(self.crops_dir, "crop_pessoa_1.jpg")
        Image.new("RGB", (60, 60), color="blue").save(crop_file)

        register_known_person(
            self.db_path,
            person_label="Pessoa 1",
            description="Pessoa teste",
            first_seen_image_id=1,
            face_crop_path=crop_file
        )

        namer_gui = self._create_namer_gui(parent_root=parent)
        self.assertTrue(namer_gui.is_toplevel)
        self.assertIsNotNone(namer_gui.photo)
        self.assertEqual(namer_gui.img_label.image, namer_gui.photo)

    def test_ensure_face_crop_for_person_resolution(self):
        # Quando o arquivo não existe inicialmente mas a foto de origem existe
        register_known_person(
            self.db_path,
            person_label="Pessoa 2",
            description="Perfil",
            first_seen_image_id=2
        )

        resolved_crop = ensure_face_crop_for_person(
            self.db_path,
            person_label="Pessoa 2",
            crops_dir=self.crops_dir
        )
        self.assertIsNotNone(resolved_crop)
        self.assertTrue(os.path.isfile(resolved_crop))

    def test_rotate_image_file_utility(self):
        test_img = os.path.join(self.temp_dir.name, "rot_test.jpg")
        with Image.new("RGB", (300, 150), color="red") as im:
            im.save(test_img)

        # Gira 90 graus horario -> dimensoes devem virar 150x300
        ok = rotate_image_file(test_img, 90)
        self.assertTrue(ok)
        with Image.open(test_img) as im:
            self.assertEqual(im.size, (150, 300))

        # Gira 180 graus -> dimensoes permanecem 150x300
        ok = rotate_image_file(test_img, 180)
        self.assertTrue(ok)
        with Image.open(test_img) as im:
            self.assertEqual(im.size, (150, 300))

        # Gira -90 graus (270) -> volta a 300x150
        ok = rotate_image_file(test_img, -90)
        self.assertTrue(ok)
        with Image.open(test_img) as im:
            self.assertEqual(im.size, (300, 150))

        # Arquivo inexistente
        self.assertFalse(rotate_image_file("nonexistent_path_12345.jpg", 90))

    def test_create_manual_face_crop_utility(self):
        test_img = os.path.join(self.temp_dir.name, "crop_source.jpg")
        with Image.new("RGB", (400, 300), color="green") as im:
            im.save(test_img)

        out_crop = os.path.join(self.crops_dir, "crop_manual_test.jpg")
        crop_pil, crop_bytes, box_norm = create_manual_face_crop(
            image_path=test_img,
            box=(50, 60, 150, 180),
            output_crop_path=out_crop
        )

        self.assertIsNotNone(crop_pil)
        self.assertIsNotNone(crop_bytes)
        self.assertEqual(crop_pil.size, (100, 120))
        self.assertTrue(os.path.isfile(out_crop))

        # [ymin, xmin, ymax, xmax] normalizados (0-1000)
        # 60/300 = 200, 50/400 = 125, 180/300 = 600, 150/400 = 375
        self.assertEqual(box_norm, [200, 125, 600, 375])

    def test_photo_viewer_gui_rotation(self):
        gui = self._create_viewer_gui(initial_image_id=1)
        file_path = self.images_data[0]["file_path"]

        with patch.object(gui, "_show_current_image"):
            gui._rotate_current_image(90)

        self.assertIn("Foto rotacionada", gui.lbl_status_msg.cget("text"))

    def test_photo_viewer_gui_manual_crop_and_add(self):
        gui = self._create_viewer_gui(initial_image_id=1)

        # Configura geometria simulada renderizada
        gui._rendered_img_rect = (50, 50, 250, 250)  # disp_w=200, disp_h=200
        gui._rendered_orig_size = (200, 200)

        crop_pil = Image.new("RGB", (60, 60), color="yellow")
        crop_bytes = b"fake_jpeg_bytes"

        # Adiciona pessoa manualmente a partir do crop
        gui._add_manual_crop_to_image_and_catalog(
            image_id=1,
            person_name="Mariana",
            crop_data={
                "image_bytes": crop_bytes,
                "box_2d_norm": [100, 100, 400, 400],
                "original_face_bbox": (20, 20, 80, 80),
                "pil_image": crop_pil,
                "person_label": "Mariana"
            }
        )

        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT people_present FROM images WHERE id = 1").fetchone()
            pp = json.loads(row["people_present"])
            self.assertIn("Mariana", pp)

            kp = conn.execute("SELECT * FROM known_people WHERE person_label = 'Mariana'").fetchone()
            self.assertIsNotNone(kp)
            self.assertEqual(kp["person_label"], "Mariana")

    def test_photo_viewer_toggle_manual_crop_mode(self):
        gui = self._create_viewer_gui(initial_image_id=1)
        self.assertFalse(gui.manual_crop_active)

        gui._toggle_manual_crop_mode()
        self.assertTrue(gui.manual_crop_active)
        self.assertEqual(gui.canvas_photo.cget("cursor"), "crosshair")

        gui._toggle_manual_crop_mode()
        self.assertFalse(gui.manual_crop_active)

    def test_assign_face_name_preserves_current_index_and_filter(self):
        # Configura imagens 2 e 3 com "Pessoa 1"
        register_known_person(self.db_path, "Pessoa 1", description="Teste")
        update_image_people_present(self.db_path, 2, ["Pessoa 1"])
        update_image_people_present(self.db_path, 3, ["Pessoa 1"])

        # Abre visualizador filtrando por "Pessoa 1" e posiciona na imagem 3 (idx 1)
        gui = self._create_viewer_gui(initial_image_id=3, filter_person="Pessoa 1")
        self.assertEqual(len(gui.images_list), 2)
        self.assertEqual(gui.current_idx, 1)
        self.assertEqual(gui.images_list[gui.current_idx]["id"], 3)

        # Simula resposta "Sim" (True) no askyesnocancel para renomear globalmente
        with patch.object(gui.messagebox, "askyesnocancel", return_value=True):
            gui._assign_face_name(
                image_id=3,
                face_idx=0,
                new_name="Roberto",
                old_name="Pessoa 1",
                crop_data=None
            )

        # O filtro deve ser atualizado para "Roberto", o índice mantido em 1 (imagem 3)
        self.assertEqual(gui.filter_person, "Roberto")
        self.assertEqual(len(gui.images_list), 2)
        self.assertEqual(gui.current_idx, 1)
        self.assertEqual(gui.images_list[gui.current_idx]["id"], 3)

    def test_assign_face_name_in_all_images_preserves_index(self):
        # Abre visualizador na foto 2 de 3 (sem filtro de pessoa)
        gui = self._create_viewer_gui(initial_image_id=2)
        self.assertEqual(gui.current_idx, 1)

        # Atribui nome na foto 2
        with patch.object(gui.messagebox, "askyesnocancel", return_value=False):
            gui._assign_face_name(
                image_id=2,
                face_idx=None,
                new_name="Fernanda",
                old_name="",
                crop_data=None
            )

        # Permanece na foto 2 (idx 1), não retorna para o início da lista
        self.assertEqual(gui.current_idx, 1)
        self.assertEqual(gui.images_list[gui.current_idx]["id"], 2)

    def test_shortcuts_ignored_when_typing_in_text_widgets(self):
        gui = self._create_viewer_gui(initial_image_id=1)

        # Cria mock e widgets de texto reais
        entry = gui.ttk.Entry(gui.root)
        entry.pack()
        combo = gui.ttk.Combobox(gui.root)
        combo.pack()
        text = gui.scrolledtext.ScrolledText(gui.root)
        text.pack()

        class DummyEvent:
            def __init__(self, widget):
                self.widget = widget

        with patch.object(gui, "_rotate_current_image") as mock_rotate, \
             patch.object(gui, "_prev_image") as mock_prev, \
             patch.object(gui, "_next_image") as mock_next, \
             patch.object(gui, "_detect_faces_on_current_image") as mock_f5:

            # 1. Quando o evento vem de um Combobox (ex: digitando o nome 'Sandra' que contém 'r')
            ev_combo = DummyEvent(combo)
            self.assertTrue(gui._is_text_widget_active(ev_combo))

            # 2. Quando o evento vem de um Entry
            ev_entry = DummyEvent(entry)
            self.assertTrue(gui._is_text_widget_active(ev_entry))

            # 3. Quando o evento vem de um ScrolledText / Text
            ev_text = DummyEvent(text)
            self.assertTrue(gui._is_text_widget_active(ev_text))

            # 4. Quando o evento vem do Canvas ou da janela raiz (sem foco de texto)
            ev_canvas = DummyEvent(gui.canvas_photo)
            with patch.object(gui.root, "focus_get", return_value=gui.canvas_photo):
                self.assertFalse(gui._is_text_widget_active(ev_canvas))

            # Testa a ativação dos atalhos via manipulação direta
            # Com foco no combobox, não deve rotacionar nem navegar
            with patch.object(gui.root, "focus_get", return_value=combo):
                # Simula execução dos bindings com e.widget = combo
                for key_event in ["<r>", "<R>", "<l>", "<L>", "<Control-Right>", "<Control-Left>"]:
                    gui.root.event_generate(key_event)
                mock_rotate.assert_not_called()
                mock_prev.assert_not_called()
                mock_next.assert_not_called()
                mock_f5.assert_not_called()

            # Com foco no canvas (visualização normal), atalhos funcionam
            with patch.object(gui.root, "focus_get", return_value=gui.canvas_photo):
                # Rotacionar com 'r'
                ev = DummyEvent(gui.canvas_photo)
                if not gui._is_text_widget_active(ev):
                    gui._rotate_current_image(90)
                mock_rotate.assert_called_with(90)

    @patch("builtins.input")
    def test_run_cli_photo_viewer(self, mock_input):
        update_image_people_present(self.db_path, 1, ["Daniel"])
        # Simula opções: [n] adicionar "Eduardo", [r] rotacionar [1], [ENTER] próxima, [q] sair
        mock_input.side_effect = ["n", "Eduardo", "r", "1", "", "q"]

        run_cli_photo_viewer(self.db_path)

        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT people_present FROM images WHERE id = 1").fetchone()
            pp = json.loads(row["people_present"])
            self.assertIn("Eduardo", pp)


if __name__ == "__main__":
    unittest.main()
