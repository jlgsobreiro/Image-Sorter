"""
Módulo de Visualização de Fotos e Nomeação Manual de Pessoas para o ImageSorter.
Permite navegar pelas fotos cadastradas no banco, inspecionar visualmente os rostos detectados,
atribuir ou renomear pessoas diretamente na foto e detectar rostos sob demanda.
"""
import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageOps

logger = logging.getLogger("ImageSorter.PhotoViewer")

try:
    from db import (
        DEFAULT_DB_PATH,
        get_connection,
        get_all_images,
        get_image_by_id,
        get_known_people,
        register_known_person,
        update_image_people_present,
        rename_or_merge_known_person,
        update_person_face_crop,
        update_image_hash,
        search_images
    )
    from face_cropper import (
        ensure_face_crop_for_person,
        extract_face_crops_in_memory,
        detect_faces_local,
        draw_face_boxes_on_image,
        DEFAULT_CROPS_DIR
    )
    from face_recognition import FaceIdentityMatcher
except ImportError:
    from .db import (
        DEFAULT_DB_PATH,
        get_connection,
        get_all_images,
        get_image_by_id,
        get_known_people,
        register_known_person,
        update_image_people_present,
        rename_or_merge_known_person,
        update_person_face_crop,
        update_image_hash,
        search_images
    )
    from .face_cropper import (
        ensure_face_crop_for_person,
        extract_face_crops_in_memory,
        detect_faces_local,
        draw_face_boxes_on_image,
        DEFAULT_CROPS_DIR
    )
    from .face_recognition import FaceIdentityMatcher


def open_file_in_system(file_path: str) -> None:
    """Abre o arquivo no visualizador padrão do sistema operacional."""
    if not file_path or not os.path.exists(file_path):
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(file_path)
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", file_path], check=False)
        else:
            subprocess.run(["xdg-open", file_path], check=False)
    except Exception as e:
        logger.warning(f"Não foi possível abrir o arquivo {file_path}: {e}")


def rotate_image_file(file_path: str, angle: int = 90) -> bool:
    """
    Rotaciona um arquivo de imagem em disco pelo ângulo especificado (sentido horário para valores positivos).
    Suporta 90, 180, 270 (ou -90).
    Aplica exif_transpose previamente para normalizar a orientação.
    """
    if not file_path or not os.path.isfile(file_path):
        return False
    try:
        with Image.open(file_path) as im:
            fmt = im.format or "JPEG"
            im_trans = ImageOps.exif_transpose(im)
            if im_trans is None:
                im_trans = im

            # No Pillow rotate gira no sentido anti-horário; para positivo=horário usa (-angle) % 360
            rot_angle = (-angle) % 360
            if rot_angle == 0:
                return True

            rotated = im_trans.rotate(rot_angle, expand=True)

            if fmt.upper() in ("JPEG", "JPG"):
                rotated.convert("RGB").save(file_path, format="JPEG", quality=95)
            elif fmt.upper() in ("PNG", "WEBP", "BMP", "GIF"):
                rotated.save(file_path, format=fmt)
            else:
                rotated.save(file_path)
        logger.info(f"Imagem rotacionada {angle}° com sucesso: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Erro ao rotacionar imagem {file_path}: {e}")
        return False


def create_manual_face_crop(
    image_path: str,
    box: Tuple[int, int, int, int],
    output_crop_path: Optional[str] = None
) -> Tuple[Optional[Image.Image], Optional[bytes], List[int]]:
    """
    Recorta manualmente uma região da imagem original a partir das coordenadas em pixels (left, top, right, bottom).
    Retorna (PIL.Image, bytes_jpeg, box_2d_norm).
    """
    if not os.path.isfile(image_path):
        return None, None, []
    try:
        import io
        with Image.open(image_path) as im:
            orig_w, orig_h = im.size
            l, t, r, b = box
            l = max(0, min(orig_w - 1, int(l)))
            t = max(0, min(orig_h - 1, int(t)))
            r = max(l + 1, min(orig_w, int(r)))
            b = max(t + 1, min(orig_h, int(b)))

            crop_pil = im.convert("RGB").crop((l, t, r, b))

            ymin = int(round((t / orig_h) * 1000))
            xmin = int(round((l / orig_w) * 1000))
            ymax = int(round((b / orig_h) * 1000))
            xmax = int(round((r / orig_w) * 1000))
            box_2d_norm = [ymin, xmin, ymax, xmax]

            buf = io.BytesIO()
            crop_pil.save(buf, format="JPEG", quality=95)
            crop_bytes = buf.getvalue()

            if output_crop_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_crop_path)), exist_ok=True)
                crop_pil.save(output_crop_path, format="JPEG", quality=95)

            return crop_pil, crop_bytes, box_2d_norm
    except Exception as e:
        logger.error(f"Erro ao criar recorte manual para {image_path}: {e}")
        return None, None, []


class PhotoViewerGUI:
    """
    Interface gráfica Tkinter para visualização completa de fotos com
    detecção facial sobreposta e painel de nomeação/edição manual de pessoas.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        initial_image_id: Optional[int] = None,
        filter_person: Optional[str] = None,
        parent_root: Optional[Any] = None
    ):
        import tkinter as tk
        from tkinter import ttk, messagebox, scrolledtext
        from PIL import ImageTk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.scrolledtext = scrolledtext
        self.ImageTk = ImageTk

        self.db_path = db_path
        self.filter_person = filter_person
        self.images_list: List[Dict[str, Any]] = []
        self.current_idx = 0
        self.current_face_crops: List[Dict[str, Any]] = []
        self._keep_refs: List[Any] = []
        self.main_photo_tk: Optional[Any] = None

        # Geometria da foto no Canvas e controle de seleção/recorte manual
        self._rendered_img_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._rendered_orig_size: Tuple[int, int] = (0, 0)
        self._drag_start: Optional[Tuple[int, int]] = None
        self._drag_rect_id: Optional[Any] = None
        self.manual_crop_active: bool = False

        self.is_toplevel = parent_root is not None
        if self.is_toplevel:
            self.root = self.tk.Toplevel(parent_root)
        else:
            self.root = self.tk.Tk()

        self.root.title("ImageSorter — Visualizador de Fotos e Identificação de Pessoas")
        self.root.geometry("1180x820")
        self.root.minsize(920, 640)

        self._build_ui()
        self._load_images(initial_image_id=initial_image_id)
        self._bind_shortcuts()

    def _build_ui(self):
        # Frame Principal
        main_frame = self.ttk.Frame(self.root, padding="8")
        main_frame.pack(fill=self.tk.BOTH, expand=True)

        # -----------------------------------------------------------------
        # BARRA SUPERIOR: FILTROS E CONTROLES DE NAVEGAÇÃO
        # -----------------------------------------------------------------
        top_bar = self.ttk.Frame(main_frame, padding="4")
        top_bar.pack(fill=self.tk.X, pady=(0, 6))

        # Filtro
        self.ttk.Label(top_bar, text="Filtro:", font=("Segoe UI", 9, "bold")).pack(side=self.tk.LEFT, padx=(0, 4))
        self.filter_var = self.tk.StringVar(value="all")
        self.combo_filter = self.ttk.Combobox(
            top_bar,
            textvariable=self.filter_var,
            state="readonly",
            values=[
                "Todas as Fotos",
                "📷 Apenas Fotos Reais",
                "📱 Apenas Prints de Tela",
                "🎨 Apenas Ícones/Assets",
                "Com Pessoas Identificadas",
                "Sem Pessoas",
                "Processadas",
                "Pendentes"
            ],
            width=26
        )
        self.combo_filter.current(0)
        self.combo_filter.pack(side=self.tk.LEFT, padx=(0, 10))
        self.combo_filter.bind("<<ComboboxSelected>>", lambda e: self._on_filter_changed())

        # Busca por termo
        self.search_var = self.tk.StringVar()
        entry_search = self.ttk.Entry(top_bar, textvariable=self.search_var, width=18)
        entry_search.pack(side=self.tk.LEFT, padx=(0, 4))
        entry_search.bind("<Return>", lambda e: self._on_filter_changed())

        btn_search = self.ttk.Button(top_bar, text="🔍 Buscar", command=self._on_filter_changed)
        btn_search.pack(side=self.tk.LEFT, padx=(0, 15))

        # Navegação
        self.btn_prev = self.ttk.Button(top_bar, text="◀ Anterior", command=self._prev_image)
        self.btn_prev.pack(side=self.tk.LEFT, padx=(0, 4))

        self.lbl_counter = self.ttk.Label(
            top_bar,
            text="Foto 0 de 0",
            font=("Segoe UI", 10, "bold")
        )
        self.lbl_counter.pack(side=self.tk.LEFT, padx=8)

        self.btn_next = self.ttk.Button(top_bar, text="Próxima ▶", command=self._next_image)
        self.btn_next.pack(side=self.tk.LEFT, padx=(4, 15))

        # Controles de Rotação de Imagem
        self.ttk.Label(top_bar, text="Girar:", font=("Segoe UI", 9, "bold")).pack(side=self.tk.LEFT, padx=(0, 4))
        self.btn_rot_left = self.ttk.Button(
            top_bar,
            text="⟲ -90°",
            width=6,
            command=lambda: self._rotate_current_image(-90)
        )
        self.btn_rot_left.pack(side=self.tk.LEFT, padx=(0, 2))

        self.btn_rot_right = self.ttk.Button(
            top_bar,
            text="⟳ +90°",
            width=6,
            command=lambda: self._rotate_current_image(90)
        )
        self.btn_rot_right.pack(side=self.tk.LEFT, padx=(0, 2))

        self.btn_rot_180 = self.ttk.Button(
            top_bar,
            text="🔄 180°",
            width=6,
            command=lambda: self._rotate_current_image(180)
        )
        self.btn_rot_180.pack(side=self.tk.LEFT, padx=(0, 15))

        # Botão abrir no sistema
        btn_open_ext = self.ttk.Button(
            top_bar,
            text="👁️ Abrir no Sistema",
            command=self._open_current_in_system
        )
        btn_open_ext.pack(side=self.tk.RIGHT)

        # -----------------------------------------------------------------
        # CONTAINER CENTRAL: DIVISÃO EM DOIS PAINÉIS (FOTO + PESSOAS)
        # -----------------------------------------------------------------
        paned = self.ttk.PanedWindow(main_frame, orient=self.tk.HORIZONTAL)
        paned.pack(fill=self.tk.BOTH, expand=True)

        # PAINEL ESQUERDO: EXIBIÇÃO DA FOTO COM RETÂNGULOS DELIMITADORES
        self.frame_photo_container = self.ttk.LabelFrame(
            paned,
            text=" 🖼️ Imagem Selecionada (Arraste sobre o rosto para criar um recorte manual) ",
            padding="6"
        )
        paned.add(self.frame_photo_container, weight=5)

        self.canvas_photo = self.tk.Canvas(
            self.frame_photo_container,
            background="#1E1E1E",
            highlightthickness=0
        )
        self.canvas_photo.pack(fill=self.tk.BOTH, expand=True)
        self.canvas_photo.bind("<Configure>", lambda e: self._render_current_photo_deferred())
        self.canvas_photo.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas_photo.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas_photo.bind("<ButtonRelease-1>", self._on_canvas_release)

        self.lbl_crop_hint = self.ttk.Label(
            self.frame_photo_container,
            text="💡 Dica: Arraste o mouse sobre qualquer rosto na foto para selecioná-lo e nomeá-lo manualmente.",
            font=("Segoe UI", 8, "italic"),
            foreground="#888888"
        )
        self.lbl_crop_hint.pack(side=self.tk.BOTTOM, fill=self.tk.X, pady=(2, 0))

        # PAINEL DIREITO: GESTÃO E NOMEAÇÃO DE ROSTOS / METADADOS
        self.frame_sidebar = self.ttk.Frame(paned, padding="4")
        paned.add(self.frame_sidebar, weight=3)

        # Subpainel 1: Pessoas e Rostos nesta Foto
        self.frame_faces = self.ttk.LabelFrame(
            self.frame_sidebar,
            text=" 👤 Pessoas e Rostos nesta Foto ",
            padding="6"
        )
        self.frame_faces.pack(fill=self.tk.BOTH, expand=True, pady=(0, 6))

        # Canvas com Scrollbar para os cards de pessoas nesta foto
        faces_scroll_frame = self.ttk.Frame(self.frame_faces)
        faces_scroll_frame.pack(fill=self.tk.BOTH, expand=True)

        self.faces_canvas = self.tk.Canvas(faces_scroll_frame, highlightthickness=0, background="#FAFAFA")
        self.faces_scrollbar = self.ttk.Scrollbar(faces_scroll_frame, orient=self.tk.VERTICAL, command=self.faces_canvas.yview)
        self.faces_inner_frame = self.ttk.Frame(self.faces_canvas)

        self.faces_inner_frame.bind(
            "<Configure>",
            lambda e: self.faces_canvas.configure(scrollregion=self.faces_canvas.bbox("all"))
        )
        self.faces_canvas_window = self.faces_canvas.create_window((0, 0), window=self.faces_inner_frame, anchor="nw")
        self.faces_canvas.configure(yscrollcommand=self.faces_scrollbar.set)

        self.faces_canvas.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        self.faces_scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)

        # Botões de Ação para a foto
        actions_box = self.ttk.Frame(self.frame_faces, padding="4")
        actions_box.pack(fill=self.tk.X, pady=(4, 0))

        btn_detect = self.ttk.Button(
            actions_box,
            text="🔍 Detectar Rostos",
            command=self._detect_faces_on_current_image
        )
        btn_detect.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=(0, 2))

        btn_crop_manual = self.ttk.Button(
            actions_box,
            text="✂️ Recorte Manual",
            command=self._toggle_manual_crop_mode
        )
        btn_crop_manual.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=(0, 2))

        btn_add_manual = self.ttk.Button(
            actions_box,
            text="➕ Nomear",
            command=self._add_person_manually_dialog
        )
        btn_add_manual.pack(side=self.tk.RIGHT, fill=self.tk.X, expand=True)

        # Subpainel 2: Informações da Imagem e Descrição
        self.frame_meta = self.ttk.LabelFrame(
            self.frame_sidebar,
            text=" 📋 Detalhes & Descrição da Foto ",
            padding="6"
        )
        self.frame_meta.pack(fill=self.tk.BOTH, expand=False)

        self.lbl_meta_filename = self.ttk.Label(
            self.frame_meta,
            text="Arquivo: -",
            font=("Segoe UI", 9, "bold")
        )
        self.lbl_meta_filename.pack(anchor=self.tk.W)

        self.lbl_meta_status = self.ttk.Label(
            self.frame_meta,
            text="Status: - | Dimensões: -",
            font=("Segoe UI", 8)
        )
        self.lbl_meta_status.pack(anchor=self.tk.W, pady=(2, 4))

        self.txt_description = self.scrolledtext.ScrolledText(
            self.frame_meta,
            wrap=self.tk.WORD,
            height=4,
            font=("Segoe UI", 9),
            background="#FFFFFF"
        )
        self.txt_description.pack(fill=self.tk.X, pady=(2, 4))

        btn_save_desc = self.ttk.Button(
            self.frame_meta,
            text="💾 Salvar Alteração na Descrição",
            command=self._save_description_edit
        )
        btn_save_desc.pack(anchor=self.tk.E)

        # -----------------------------------------------------------------
        # BARRA DE STATUS INFERIOR
        # -----------------------------------------------------------------
        status_bar = self.ttk.Frame(main_frame, padding="2")
        status_bar.pack(fill=self.tk.X, pady=(4, 0))

        self.lbl_status_msg = self.ttk.Label(
            status_bar,
            text="Atalhos: [← / →]: Navegar | [R]: Rotacionar 90° | [L]: Rotacionar -90° | [F5]: Redetectar Rostos | [Arraste na foto]: Recorte manual",
            font=("Segoe UI", 8, "italic"),
            foreground="#555555"
        )
        self.lbl_status_msg.pack(side=self.tk.LEFT)

    def _is_text_widget_active(self, event=None) -> bool:
        """Verifica se o foco atual do teclado ou o widget de origem do evento é um campo de texto/edição."""
        candidates = []
        if event is not None and getattr(event, "widget", None) is not None:
            candidates.append(event.widget)
        try:
            focused = self.root.focus_get()
            if focused is not None and focused not in candidates:
                candidates.append(focused)
        except Exception:
            pass

        text_classes = {"Entry", "TEntry", "Combobox", "TCombobox", "Text", "Spinbox", "TSpinbox"}

        for widget in candidates:
            try:
                if widget.winfo_class() in text_classes:
                    return True
            except Exception:
                pass

            types_to_check = []
            for cls_name in ("Entry", "Combobox", "Text", "Spinbox"):
                if hasattr(self.ttk, cls_name):
                    types_to_check.append(getattr(self.ttk, cls_name))
                if hasattr(self.tk, cls_name):
                    types_to_check.append(getattr(self.tk, cls_name))
            if hasattr(self, "scrolledtext") and hasattr(self.scrolledtext, "ScrolledText"):
                types_to_check.append(self.scrolledtext.ScrolledText)

            if types_to_check and isinstance(widget, tuple(types_to_check)):
                return True

        return False

    def _bind_shortcuts(self):
        self.root.bind("<Left>", lambda e: None if self._is_text_widget_active(e) else self._prev_image())
        self.root.bind("<Right>", lambda e: None if self._is_text_widget_active(e) else self._next_image())
        self.root.bind("<F5>", lambda e: None if self._is_text_widget_active(e) else self._detect_faces_on_current_image())
        self.root.bind("<r>", lambda e: None if self._is_text_widget_active(e) else self._rotate_current_image(90))
        self.root.bind("<R>", lambda e: None if self._is_text_widget_active(e) else self._rotate_current_image(90))
        self.root.bind("<l>", lambda e: None if self._is_text_widget_active(e) else self._rotate_current_image(-90))
        self.root.bind("<L>", lambda e: None if self._is_text_widget_active(e) else self._rotate_current_image(-90))
        self.root.bind("<Control-Right>", lambda e: None if self._is_text_widget_active(e) else self._rotate_current_image(90))
        self.root.bind("<Control-Left>", lambda e: None if self._is_text_widget_active(e) else self._rotate_current_image(-90))
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def _load_images(self, initial_image_id: Optional[int] = None):
        prev_idx = getattr(self, "current_idx", 0)
        target_id = initial_image_id
        if target_id is None and getattr(self, "images_list", None) and 0 <= prev_idx < len(self.images_list):
            target_id = self.images_list[prev_idx].get("id")

        filter_text = self.filter_var.get()
        query = self.search_var.get().strip()

        status_filter = None
        type_filter = None
        if filter_text == "Processadas":
            status_filter = "processed"
        elif filter_text == "Pendentes":
            status_filter = "pending"
        elif "Fotos Reais" in filter_text:
            type_filter = "photo"
        elif "Prints" in filter_text:
            type_filter = "screenshot"
        elif "Ícones" in filter_text or "Icones" in filter_text:
            type_filter = "icon_or_graphic"

        self.images_list = get_all_images(
            db_path=self.db_path,
            status=status_filter,
            person_label=self.filter_person,
            query_text=query if query else None,
            image_type=type_filter
        )

        if filter_text == "Com Pessoas Identificadas":
            filtered = []
            for img in self.images_list:
                pp = img.get("people_present") or "[]"
                try:
                    p_list = json.loads(pp)
                    if isinstance(p_list, list) and len(p_list) > 0:
                        filtered.append(img)
                except Exception:
                    if pp and pp != "[]":
                        filtered.append(img)
            self.images_list = filtered

        elif filter_text == "Sem Pessoas":
            filtered = []
            for img in self.images_list:
                pp = img.get("people_present") or "[]"
                try:
                    p_list = json.loads(pp)
                    if not isinstance(p_list, list) or len(p_list) == 0:
                        filtered.append(img)
                except Exception:
                    if not pp or pp == "[]":
                        filtered.append(img)
            self.images_list = filtered

        # Posiciona no ID inicial se solicitado, ou preserva o índice atual no acervo
        total = len(self.images_list)
        if total == 0:
            self.current_idx = 0
        else:
            found_idx = None
            if target_id is not None:
                for idx, img in enumerate(self.images_list):
                    if img["id"] == target_id:
                        found_idx = idx
                        break
            if found_idx is not None:
                self.current_idx = found_idx
            else:
                self.current_idx = max(0, min(prev_idx, total - 1))

        self._show_current_image()

    def _on_filter_changed(self):
        self._load_images()

    def _show_current_image(self):
        total = len(self.images_list)
        if total == 0:
            self.lbl_counter.config(text="Nenhuma foto encontrada")
            self.canvas_photo.delete("all")
            self.canvas_photo.create_text(
                200, 150,
                text="Nenhuma imagem corresponde aos filtros aplicados.",
                fill="#AAAAAA",
                font=("Segoe UI", 11, "italic")
            )
            self._render_empty_faces_panel()
            self.lbl_meta_filename.config(text="Arquivo: -")
            self.lbl_meta_status.config(text="Status: -")
            self.txt_description.delete("1.0", self.tk.END)
            self.btn_prev.config(state=self.tk.DISABLED)
            self.btn_next.config(state=self.tk.DISABLED)
            return

        self.current_idx = max(0, min(self.current_idx, total - 1))
        img_data = self.images_list[self.current_idx]

        self.lbl_counter.config(text=f"Foto {self.current_idx + 1} de {total}")
        self.btn_prev.config(state=self.tk.NORMAL if self.current_idx > 0 else self.tk.DISABLED)
        self.btn_next.config(state=self.tk.NORMAL if self.current_idx < total - 1 else self.tk.DISABLED)

        # Atualiza Metadados
        file_path = img_data.get("file_path", "")
        file_name = img_data.get("file_name", os.path.basename(file_path))
        status = img_data.get("status", "pending")
        raw_type = img_data.get("image_type", "unclassified")
        type_badges = {
            "photo": "📷 Foto Real",
            "screenshot": "📱 Print de Tela",
            "icon_or_graphic": "🎨 Ícone / Asset",
            "other": "📄 Outro / Doc",
            "unclassified": "⏳ Não Classificada"
        }
        type_str = type_badges.get(raw_type, raw_type or "⏳ Não Classificada")
        size_kb = (img_data.get("file_size") or (os.path.getsize(file_path) if os.path.isfile(file_path) else 0)) / 1024

        dim_str = "N/D"
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as im:
                    dim_str = f"{im.width}x{im.height}"
            except Exception:
                pass

        self.lbl_meta_filename.config(text=f"Arquivo: {file_name}")
        self.lbl_meta_status.config(text=f"Tipo: {type_str} | Status: {status.upper()} | Tamanho: {size_kb:.1f} KB | Resolução: {dim_str}")

        self.txt_description.delete("1.0", self.tk.END)
        self.txt_description.insert(self.tk.END, img_data.get("description") or "(Sem descrição)")

        # Carrega rostos e renderiza imagem
        self._load_and_render_photo_with_faces(img_data)

    def _render_current_photo_deferred(self):
        if self.images_list and 0 <= self.current_idx < len(self.images_list):
            img_data = self.images_list[self.current_idx]
            self._render_photo_to_canvas(img_data)

    def _load_and_render_photo_with_faces(self, img_data: Dict[str, Any]):
        file_path = img_data.get("file_path", "")
        if not os.path.isfile(file_path):
            self.canvas_photo.delete("all")
            self.canvas_photo.create_text(
                200, 150,
                text=f"Arquivo de imagem não encontrado no disco:\n{file_path}",
                fill="#FF6666",
                font=("Segoe UI", 10)
            )
            self._render_empty_faces_panel("Arquivo de imagem ausente.")
            return

        # Extrai ou compõe rostos presentes
        raw_pp = img_data.get("people_present") or "[]"
        try:
            people_names = json.loads(raw_pp)
            if not isinstance(people_names, list):
                people_names = [str(raw_pp)] if raw_pp else []
        except Exception:
            people_names = [raw_pp] if raw_pp else []

        # Tenta extrair recortes faciais em memória para coordenadas precisas
        try:
            crops = extract_face_crops_in_memory(file_path)
        except Exception:
            crops = []

        # Atribui os rótulos de pessoas conhecidas aos recortes detectados
        for idx, crop in enumerate(crops):
            if idx < len(people_names):
                crop["person_label"] = people_names[idx]
            else:
                crop["person_label"] = f"Rosto {idx + 1}"

        self.current_face_crops = crops
        self._render_photo_to_canvas(img_data)
        self._render_faces_sidebar(people_names, crops, img_data)

    def _render_photo_to_canvas(self, img_data: Dict[str, Any]):
        file_path = img_data.get("file_path", "")
        if not os.path.isfile(file_path):
            return

        try:
            with Image.open(file_path) as pil_img:
                orig_w, orig_h = pil_img.size
                annotated = pil_img.convert("RGB")
                if self.current_face_crops:
                    annotated = draw_face_boxes_on_image(annotated, self.current_face_crops)

                cw = max(100, self.canvas_photo.winfo_width())
                ch = max(100, self.canvas_photo.winfo_height())

                # Redimensiona mantendo proporção
                img_copy = annotated.copy()
                img_copy.thumbnail((cw, ch), Image.Resampling.LANCZOS)
                disp_w, disp_h = img_copy.size

                self.main_photo_tk = self.ImageTk.PhotoImage(img_copy, master=self.canvas_photo)
                self.canvas_photo.delete("all")
                cx = cw // 2
                cy = ch // 2
                self.canvas_photo.create_image(cx, cy, image=self.main_photo_tk, anchor=self.tk.CENTER)

                # Salva dimensões geométricas na tela para conversão de coordenadas do mouse
                img_x0 = cx - (disp_w // 2)
                img_y0 = cy - (disp_h // 2)
                img_x1 = img_x0 + disp_w
                img_y1 = img_y0 + disp_h
                self._rendered_img_rect = (img_x0, img_y0, img_x1, img_y1)
                self._rendered_orig_size = (orig_w, orig_h)
        except Exception as e:
            self.canvas_photo.delete("all")
            self.canvas_photo.create_text(
                150, 100,
                text=f"Erro ao renderizar imagem: {e}",
                fill="#FF6666",
                font=("Segoe UI", 9)
            )

    def _toggle_manual_crop_mode(self):
        """Ativa ou desativa o modo de seleção manual de rosto na foto."""
        self.manual_crop_active = not self.manual_crop_active
        if self.manual_crop_active:
            self.canvas_photo.config(cursor="crosshair")
            self.lbl_status_msg.config(
                text="✂️ Modo Seleção Manual Ativo: Clique e arraste o mouse sobre o rosto na foto para recortar."
            )
            self.lbl_crop_hint.config(
                foreground="#00AAEE",
                text="✂️ MODO SELEÇÃO ATIVO: Arraste o mouse sobre o rosto na foto para criar o recorte."
            )
        else:
            self.canvas_photo.config(cursor="")
            self.lbl_status_msg.config(
                text="Atalhos: [← / →]: Navegar | [R]: Rotacionar 90° | [L]: Rotacionar -90° | [F5]: Redetectar Rostos"
            )
            self.lbl_crop_hint.config(
                foreground="#888888",
                text="💡 Dica: Arraste o mouse sobre qualquer rosto na foto para selecioná-lo e nomeá-lo manualmente."
            )

    def _on_canvas_press(self, event):
        self._drag_start = (event.x, event.y)
        if self._drag_rect_id:
            try:
                self.canvas_photo.delete(self._drag_rect_id)
            except Exception:
                pass
            self._drag_rect_id = None
        self._drag_rect_id = self.canvas_photo.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#00E5FF", width=2, dash=(4, 2)
        )

    def _on_canvas_drag(self, event):
        if not self._drag_start or not self._drag_rect_id:
            return
        self.canvas_photo.coords(
            self._drag_rect_id,
            self._drag_start[0], self._drag_start[1],
            event.x, event.y
        )

    def _on_canvas_release(self, event):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self._drag_start = None

        if self._drag_rect_id:
            try:
                self.canvas_photo.delete(self._drag_rect_id)
            except Exception:
                pass
            self._drag_rect_id = None

        canv_x0, canv_x1 = min(x0, x1), max(x0, x1)
        canv_y0, canv_y1 = min(y0, y1), max(y0, y1)

        # Se o arraste for muito pequeno (< 8px), ignora como clique acidental
        if (canv_x1 - canv_x0 < 8) or (canv_y1 - canv_y0 < 8):
            return

        self._process_manual_crop_selection(canv_x0, canv_y0, canv_x1, canv_y1)

    def _rotate_current_image(self, angle: int = 90):
        """Rotaciona o arquivo de imagem atual em disco e atualiza a interface."""
        if not self.images_list or not (0 <= self.current_idx < len(self.images_list)):
            return
        img_data = self.images_list[self.current_idx]
        image_id = img_data["id"]
        file_path = img_data.get("file_path", "")

        if not os.path.isfile(file_path):
            self.messagebox.showerror("Erro", f"Arquivo não encontrado no disco:\n{file_path}")
            return

        ok = rotate_image_file(file_path, angle)
        if not ok:
            self.messagebox.showerror("Erro", f"Falha ao rotacionar imagem:\n{file_path}")
            return

        # Atualiza tamanho do arquivo no banco
        try:
            new_size = os.path.getsize(file_path)
            with get_connection(self.db_path) as conn:
                conn.execute("UPDATE images SET file_size = ? WHERE id = ?", (new_size, image_id))
                conn.commit()
            img_data["file_size"] = new_size
        except Exception:
            pass

        # Limpa cache de detecção anterior (pois coordenadas giraram)
        self.current_face_crops = []

        # Recarrega e renderiza
        self._show_current_image()
        direction_str = f"+{angle}°" if angle > 0 else f"{angle}°"
        self.lbl_status_msg.config(text=f"✓ Foto rotacionada {direction_str} com sucesso.")

    def _process_manual_crop_selection(self, cx0: int, cy0: int, cx1: int, cy1: int):
        """Converte seleção em coordenadas da tela para pixels da imagem e abre diálogo de nomeação."""
        if not self.images_list or not (0 <= self.current_idx < len(self.images_list)):
            return
        img_data = self.images_list[self.current_idx]
        file_path = img_data.get("file_path", "")
        if not os.path.isfile(file_path):
            return

        ix0, iy0, ix1, iy1 = getattr(self, "_rendered_img_rect", (0, 0, 0, 0))
        orig_w, orig_h = getattr(self, "_rendered_orig_size", (0, 0))
        disp_w = ix1 - ix0
        disp_h = iy1 - iy0

        if disp_w <= 0 or disp_h <= 0 or orig_w <= 0 or orig_h <= 0:
            return

        # Limita seleção à área da imagem
        clamped_x0 = max(ix0, min(ix1, cx0))
        clamped_x1 = max(ix0, min(ix1, cx1))
        clamped_y0 = max(iy0, min(iy1, cy0))
        clamped_y1 = max(iy0, min(iy1, cy1))

        if (clamped_x1 - clamped_x0 < 4) or (clamped_y1 - clamped_y0 < 4):
            return

        scale_x = orig_w / disp_w
        scale_y = orig_h / disp_h

        orig_x0 = max(0, min(orig_w - 1, int(round((clamped_x0 - ix0) * scale_x))))
        orig_y0 = max(0, min(orig_h - 1, int(round((clamped_y0 - iy0) * scale_y))))
        orig_x1 = max(1, min(orig_w, int(round((clamped_x1 - ix0) * scale_x))))
        orig_y1 = max(1, min(orig_h, int(round((clamped_y1 - iy0) * scale_y))))

        if orig_x1 <= orig_x0 or orig_y1 <= orig_y0:
            return

        box = (orig_x0, orig_y0, orig_x1, orig_y1)
        crop_pil, crop_bytes, box_2d_norm = create_manual_face_crop(file_path, box)

        if not crop_pil or not crop_bytes:
            return

        self._show_manual_crop_dialog(
            img_data=img_data,
            crop_pil=crop_pil,
            crop_bytes=crop_bytes,
            box_2d_norm=box_2d_norm,
            orig_box=box
        )

    def _show_manual_crop_dialog(
        self,
        img_data: Dict[str, Any],
        crop_pil: Image.Image,
        crop_bytes: bytes,
        box_2d_norm: List[int],
        orig_box: Tuple[int, int, int, int]
    ):
        """Exibe modal para nomeação e cadastro do recorte facial selecionado manualmente."""
        dialog = self.tk.Toplevel(self.root)
        dialog.title("Nomear Recorte Manual de Pessoa")
        dialog.geometry("460x260")
        dialog.transient(self.root)
        dialog.grab_set()

        main_f = self.ttk.Frame(dialog, padding="10")
        main_f.pack(fill=self.tk.BOTH, expand=True)

        top_row = self.ttk.Frame(main_f)
        top_row.pack(fill=self.tk.BOTH, expand=True, pady=(0, 10))

        # Preview do recorte
        lbl_preview = self.tk.Label(top_row, background="#222222", width=90, height=90)
        lbl_preview.pack(side=self.tk.LEFT, padx=(0, 12))

        thumb = crop_pil.copy()
        thumb.thumbnail((90, 90), Image.Resampling.LANCZOS)
        tk_thumb = self.ImageTk.PhotoImage(thumb, master=lbl_preview)
        lbl_preview.config(image=tk_thumb, width=thumb.width, height=thumb.height)
        lbl_preview.image = tk_thumb

        # Formulário
        form_f = self.ttk.Frame(top_row)
        form_f.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)

        self.ttk.Label(
            form_f,
            text="Recorte selecionado com sucesso!\nDigite ou selecione o nome da pessoa:",
            font=("Segoe UI", 9, "bold")
        ).pack(anchor=self.tk.W, pady=(0, 6))

        known_people = get_known_people(self.db_path)
        known_labels = [p["person_label"] for p in known_people]

        raw_pp = img_data.get("people_present") or "[]"
        try:
            cur_pp = json.loads(raw_pp)
            if not isinstance(cur_pp, list):
                cur_pp = []
        except Exception:
            cur_pp = []
        default_name = f"Pessoa {len(cur_pp) + 1}"

        combo_name = self.ttk.Combobox(form_f, values=known_labels, font=("Segoe UI", 10))
        combo_name.set(default_name)
        combo_name.pack(fill=self.tk.X, pady=(0, 6))
        combo_name.focus_set()
        combo_name.select_range(0, self.tk.END)

        lbl_info = self.ttk.Label(
            form_f,
            text=f"Resolução do recorte: {orig_box[2]-orig_box[0]}x{orig_box[3]-orig_box[1]} px",
            font=("Segoe UI", 8),
            foreground="#666666"
        )
        lbl_info.pack(anchor=self.tk.W)

        def do_save():
            chosen_name = combo_name.get().strip()
            if not chosen_name:
                self.messagebox.showwarning("Aviso", "Por favor, digite um nome para a pessoa.", parent=dialog)
                return

            dialog.destroy()
            crop_data_dict = {
                "image_bytes": crop_bytes,
                "box_2d_norm": box_2d_norm,
                "original_face_bbox": orig_box,
                "pil_image": crop_pil,
                "person_label": chosen_name
            }

            self._add_manual_crop_to_image_and_catalog(
                image_id=img_data["id"],
                person_name=chosen_name,
                crop_data=crop_data_dict
            )

        btn_row = self.ttk.Frame(main_f)
        btn_row.pack(fill=self.tk.X, pady=(5, 0))

        btn_ok = self.ttk.Button(btn_row, text="💾 Salvar Pessoa e Recorte", command=do_save)
        btn_ok.pack(side=self.tk.RIGHT, padx=(5, 0))

        btn_cancel = self.ttk.Button(btn_row, text="Cancelar", command=dialog.destroy)
        btn_cancel.pack(side=self.tk.RIGHT)

        combo_name.bind("<Return>", lambda e: do_save())

    def _add_manual_crop_to_image_and_catalog(
        self,
        image_id: int,
        person_name: str,
        crop_data: Dict[str, Any]
    ):
        """Salva o recorte manual, registra a pessoa no catálogo e adiciona à imagem atual."""
        clean_name = person_name.strip()
        if not clean_name:
            return

        img = get_image_by_id(self.db_path, image_id)
        if not img:
            return

        raw_pp = img.get("people_present") or "[]"
        try:
            p_list = json.loads(raw_pp)
            if not isinstance(p_list, list):
                p_list = []
        except Exception:
            p_list = []

        if clean_name not in p_list:
            p_list.append(clean_name)
            update_image_people_present(self.db_path, image_id, p_list)

        crops_dir = Path(self.db_path).parent / "face_crops" if self.db_path else Path(DEFAULT_CROPS_DIR)
        crops_dir.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(c if c.isalnum() else "_" for c in clean_name.lower())
        crop_file_path = str((crops_dir / f"crop_{safe_label}.jpg").resolve())

        try:
            if crop_data.get("image_bytes"):
                with open(crop_file_path, "wb") as f_out:
                    f_out.write(crop_data["image_bytes"])
            elif crop_data.get("pil_image"):
                crop_data["pil_image"].save(crop_file_path, format="JPEG", quality=95)
        except Exception as e:
            logger.warning(f"Não foi possível salvar recorte facial manual: {e}")
            crop_file_path = None

        bbox_json = json.dumps(crop_data.get("box_2d_norm")) if crop_data.get("box_2d_norm") else None

        known = [p["person_label"] for p in get_known_people(self.db_path)]
        if clean_name not in known:
            register_known_person(
                self.db_path,
                person_label=clean_name,
                description="Identificado por recorte manual.",
                first_seen_image_id=image_id,
                face_crop_path=crop_file_path,
                face_bbox=bbox_json
            )
        else:
            if crop_file_path:
                update_person_face_crop(self.db_path, clean_name, crop_file_path, face_bbox=bbox_json)

        self.lbl_status_msg.config(
            text=f"✓ Pessoa '{clean_name}' adicionada com sucesso a partir do recorte manual!"
        )

        self._load_images(initial_image_id=image_id)

    def _render_empty_faces_panel(self, message: str = "Nenhum rosto identificado nesta foto."):
        self._keep_refs.clear()
        for w in self.faces_inner_frame.winfo_children():
            w.destroy()

        lbl = self.ttk.Label(
            self.faces_inner_frame,
            text=message,
            font=("Segoe UI", 9, "italic"),
            foreground="#666666",
            wraplength=240,
            justify=self.tk.CENTER
        )
        lbl.pack(pady=25, padx=10)

    def _render_faces_sidebar(
        self,
        people_names: List[str],
        crops: List[Dict[str, Any]],
        img_data: Dict[str, Any]
    ):
        self._keep_refs.clear()
        for w in self.faces_inner_frame.winfo_children():
            w.destroy()

        known_people = get_known_people(self.db_path)
        known_labels = [p["person_label"] for p in known_people]

        image_id = img_data["id"]

        # Se houver recortes detectados
        if crops:
            for idx, crop in enumerate(crops):
                current_label = crop.get("person_label") or f"Pessoa {idx + 1}"
                card = self.ttk.LabelFrame(
                    self.faces_inner_frame,
                    text=f" Rosto #{idx + 1} — {current_label} ",
                    padding="6"
                )
                card.pack(fill=self.tk.X, expand=True, pady=4, padx=4)

                card_top = self.ttk.Frame(card)
                card_top.pack(fill=self.tk.X)

                # Thumbnail do rosto
                lbl_thumb = self.tk.Label(card_top, background="#EEEEEE", width=70, height=70)
                lbl_thumb.pack(side=self.tk.LEFT, padx=(0, 8))

                pil_crop = crop.get("pil_image")
                if pil_crop:
                    try:
                        crop_thumb = pil_crop.copy()
                        crop_thumb.thumbnail((70, 70), Image.Resampling.LANCZOS)
                        tk_crop = self.ImageTk.PhotoImage(crop_thumb, master=lbl_thumb)
                        self._keep_refs.append(tk_crop)
                        lbl_thumb.config(image=tk_crop, width=crop_thumb.width, height=crop_thumb.height)
                    except Exception:
                        lbl_thumb.config(text="Rosto")
                else:
                    lbl_thumb.config(text="Rosto")

                # Controles de edição / renomeação do rosto
                edit_frame = self.ttk.Frame(card_top)
                edit_frame.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)

                self.ttk.Label(edit_frame, text="Atribuir Identidade:", font=("Segoe UI", 8, "bold")).pack(anchor=self.tk.W)

                combo_name = self.ttk.Combobox(
                    edit_frame,
                    values=known_labels,
                    font=("Segoe UI", 9)
                )
                combo_name.set(current_label)
                combo_name.pack(fill=self.tk.X, pady=(2, 4))

                btn_row = self.ttk.Frame(edit_frame)
                btn_row.pack(fill=self.tk.X)

                btn_save = self.ttk.Button(
                    btn_row,
                    text="💾 Salvar Nome",
                    command=lambda c_idx=idx, cb=combo_name, old_l=current_label, cr=crop: self._assign_face_name(
                        image_id=image_id,
                        face_idx=c_idx,
                        new_name=cb.get().strip(),
                        old_name=old_l,
                        crop_data=cr
                    )
                )
                btn_save.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=(0, 2))

                btn_del = self.ttk.Button(
                    btn_row,
                    text="❌ Remover",
                    width=9,
                    command=lambda old_l=current_label: self._remove_person_from_image(image_id, old_l)
                )
                btn_del.pack(side=self.tk.RIGHT)

        elif people_names:
            # Pessoas registradas no banco mas sem recortes extraídos
            for idx, p_name in enumerate(people_names):
                card = self.ttk.LabelFrame(
                    self.faces_inner_frame,
                    text=f" Pessoa #{idx + 1} — {p_name} ",
                    padding="6"
                )
                card.pack(fill=self.tk.X, expand=True, pady=4, padx=4)

                row = self.ttk.Frame(card)
                row.pack(fill=self.tk.X)

                combo_name = self.ttk.Combobox(row, values=known_labels, font=("Segoe UI", 9))
                combo_name.set(p_name)
                combo_name.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=(0, 4))

                btn_save = self.ttk.Button(
                    row,
                    text="💾 Salvar",
                    width=8,
                    command=lambda old_l=p_name, cb=combo_name: self._assign_face_name(
                        image_id=image_id,
                        face_idx=None,
                        new_name=cb.get().strip(),
                        old_name=old_l,
                        crop_data=None
                    )
                )
                btn_save.pack(side=self.tk.LEFT, padx=(0, 2))

                btn_del = self.ttk.Button(
                    row,
                    text="❌",
                    width=3,
                    command=lambda old_l=p_name: self._remove_person_from_image(image_id, old_l)
                )
                btn_del.pack(side=self.tk.LEFT)
        else:
            self._render_empty_faces_panel("Nenhum rosto registrado nesta foto.\nClique em 'Detectar Rostos' abaixo.")

    def _assign_face_name(
        self,
        image_id: int,
        face_idx: Optional[int],
        new_name: str,
        old_name: str,
        crop_data: Optional[Dict[str, Any]]
    ):
        """Atribui ou renomeia a identidade de uma pessoa nesta imagem e no catálogo global."""
        new_clean = new_name.strip()
        if not new_clean:
            self.messagebox.showwarning("Aviso", "Por favor, digite um nome válido para a pessoa.", parent=self.root)
            return

        img = get_image_by_id(self.db_path, image_id)
        if not img:
            return

        raw_pp = img.get("people_present") or "[]"
        try:
            people_list = json.loads(raw_pp)
            if not isinstance(people_list, list):
                people_list = []
        except Exception:
            people_list = []

        # Pergunta ao usuário se deseja renomear globalmente ou apenas nesta foto se o nome anterior for específico
        rename_globally = False
        if old_name and old_name != new_clean and old_name in [p["person_label"] for p in get_known_people(self.db_path)]:
            ask = self.messagebox.askyesnocancel(
                "Renomear Pessoa",
                f"Deseja renomear '{old_name}' para '{new_clean}' em TODAS as fotos do banco?\n\n"
                f"• Sim: Renomeia '{old_name}' globalmente em todo o catálogo.\n"
                f"• Não: Altera apenas nesta foto específica.\n"
                f"• Cancelar: Não realiza alterações.",
                parent=self.root
            )
            if ask is None:
                return
            rename_globally = bool(ask)

        if rename_globally:
            res = rename_or_merge_known_person(self.db_path, old_name, new_clean)
            if self.filter_person == old_name:
                self.filter_person = new_clean
                self.root.title(f"ImageSorter — Visualizador de Fotos (Filtrando por: {new_clean})")
            self.lbl_status_msg.config(
                text=f"✓ '{old_name}' renomeado globalmente para '{new_clean}' ({res['affected_images']} fotos atualizadas)."
            )
        else:
            # Atualiza apenas a lista desta imagem
            if old_name in people_list:
                people_list = [new_clean if p == old_name else p for p in people_list]
            elif new_clean not in people_list:
                people_list.append(new_clean)

            # Garante que não haja duplicatas
            dedup_people = []
            for p in people_list:
                if p not in dedup_people:
                    dedup_people.append(p)

            update_image_people_present(self.db_path, image_id, dedup_people)

            # Registra no catálogo de pessoas se ainda não constar
            known = [p["person_label"] for p in get_known_people(self.db_path)]
            if new_clean not in known:
                crop_path = None
                bbox_json = None
                if crop_data and crop_data.get("image_bytes"):
                    crops_dir = Path(self.db_path).parent / "face_crops" if self.db_path else Path(DEFAULT_CROPS_DIR)
                    crops_dir.mkdir(parents=True, exist_ok=True)
                    safe_label = "".join(c if c.isalnum() else "_" for c in new_clean.lower())
                    crop_file_path = str((crops_dir / f"crop_{safe_label}.jpg").resolve())
                    try:
                        with open(crop_file_path, "wb") as f_out:
                            f_out.write(crop_data["image_bytes"])
                        crop_path = crop_file_path
                    except Exception:
                        crop_path = None
                    bbox_json = json.dumps(crop_data.get("box_2d_norm"))

                register_known_person(
                    self.db_path,
                    person_label=new_clean,
                    description=f"Identificado manualmente.",
                    first_seen_image_id=image_id,
                    face_crop_path=crop_path,
                    face_bbox=bbox_json
                )

            self.lbl_status_msg.config(
                text=f"✓ Pessoa '{new_clean}' atribuída à imagem com sucesso!"
            )

        # Recarrega a imagem atual
        self._load_images(initial_image_id=image_id)

    def _remove_person_from_image(self, image_id: int, person_label: str):
        """Remove a pessoa da lista de presentes nesta imagem."""
        img = get_image_by_id(self.db_path, image_id)
        if not img:
            return

        raw_pp = img.get("people_present") or "[]"
        try:
            people_list = json.loads(raw_pp)
            if not isinstance(people_list, list):
                people_list = []
        except Exception:
            people_list = []

        if person_label in people_list:
            people_list = [p for p in people_list if p != person_label]
            update_image_people_present(self.db_path, image_id, people_list)
            self.lbl_status_msg.config(text=f"✓ '{person_label}' removido desta imagem.")
            self._load_images(initial_image_id=image_id)

    def _detect_faces_on_current_image(self):
        """Executa a detecção facial e reconhecimento biométrico local na imagem atual."""
        if not self.images_list:
            return
        img_data = self.images_list[self.current_idx]
        image_id = img_data["id"]
        file_path = img_data["file_path"]

        if not os.path.isfile(file_path):
            self.messagebox.showerror("Erro", f"Arquivo de imagem não encontrado:\n{file_path}")
            return

        try:
            self.lbl_status_msg.config(text="Detectando rostos com algoritmo de visão computacional...")
            self.root.update_idletasks()

            crops = extract_face_crops_in_memory(file_path)
            if not crops:
                self.messagebox.showinfo("Detec��ão Facial", "Nenhum rosto foi detectado nesta foto pelo algoritmo local.")
                return

            known_people = get_known_people(self.db_path)
            matcher = FaceIdentityMatcher()

            identified_labels = []
            for crop in crops:
                pil_crop = crop.get("pil_image")
                match_res = matcher.match(
                    crop=pil_crop,
                    known_people=known_people,
                    excluded_labels=list(identified_labels)
                )
                if match_res.get("status") == "matched" and match_res.get("person_label"):
                    p_label = match_res["person_label"]
                else:
                    # Rótulo temporário incremental
                    idx_num = len(known_people) + len(identified_labels) + 1
                    p_label = f"Pessoa {idx_num}"

                identified_labels.append(p_label)
                crop["person_label"] = p_label

            update_image_people_present(self.db_path, image_id, identified_labels)
            self._load_images(initial_image_id=image_id)
            self.messagebox.showinfo(
                "Detecção Concluída",
                f"Detectado(s) {len(crops)} rosto(s) com sucesso!\n\nVocê pode renomeá-los agora no painel lateral."
            )
        except Exception as ex:
            self.messagebox.showerror("Erro na Detecção", f"Falha ao executar detecção facial: {ex}")

    def _add_person_manually_dialog(self):
        """Abre uma janela de diálogo para adicionar manualmente o nome de uma pessoa à foto."""
        if not self.images_list:
            return
        img_data = self.images_list[self.current_idx]
        image_id = img_data["id"]

        dialog = self.tk.Toplevel(self.root)
        dialog.title("Adicionar Pessoa Manualmente")
        dialog.geometry("380x160")
        dialog.transient(self.root)
        dialog.grab_set()

        self.ttk.Label(dialog, text="Selecione ou digite o nome da pessoa:", font=("Segoe UI", 9, "bold")).pack(pady=(15, 5), padx=15, anchor=self.tk.W)

        known_labels = [p["person_label"] for p in get_known_people(self.db_path)]
        combo = self.ttk.Combobox(dialog, values=known_labels, font=("Segoe UI", 10))
        combo.pack(fill=self.tk.X, padx=15, pady=5)
        combo.focus_set()

        def do_add():
            name = combo.get().strip()
            if not name:
                return
            dialog.destroy()
            self._assign_face_name(
                image_id=image_id,
                face_idx=None,
                new_name=name,
                old_name="",
                crop_data=None
            )

        btn_box = self.ttk.Frame(dialog)
        btn_box.pack(fill=self.tk.X, padx=15, pady=10)
        self.ttk.Button(btn_box, text="Adicionar", command=do_add).pack(side=self.tk.RIGHT, padx=5)
        self.ttk.Button(btn_box, text="Cancelar", command=dialog.destroy).pack(side=self.tk.RIGHT)
        combo.bind("<Return>", lambda e: do_add())

    def _save_description_edit(self):
        """Salva a edição manual do texto de descrição da foto."""
        if not self.images_list:
            return
        img_data = self.images_list[self.current_idx]
        image_id = img_data["id"]
        new_desc = self.txt_description.get("1.0", self.tk.END).strip()

        with get_connection(self.db_path) as conn:
            conn.execute("UPDATE images SET description = ? WHERE id = ?", (new_desc, image_id))
            conn.commit()

        self.lbl_status_msg.config(text="✓ Descrição da imagem atualizada no banco de dados com sucesso.")
        self.messagebox.showinfo("Sucesso", "Descrição atualizada com sucesso!")

    def _open_current_in_system(self):
        if not self.images_list:
            return
        file_path = self.images_list[self.current_idx].get("file_path", "")
        open_file_in_system(file_path)

    def _prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._show_current_image()

    def _next_image(self):
        if self.current_idx < len(self.images_list) - 1:
            self.current_idx += 1
            self._show_current_image()

    def start(self):
        self.root.mainloop()


def run_cli_photo_viewer(db_path: str = DEFAULT_DB_PATH, filter_person: Optional[str] = None):
    """Visualizador e gerenciador interativo de fotos pelo terminal (CLI)."""
    images = get_all_images(db_path, person_label=filter_person)
    if not images:
        print("\nNenhuma imagem encontrada no banco de dados.")
        return

    print("\n" + "=" * 65)
    print("VISUALIZADOR INTERATIVO DE FOTOS E NOMEAÇÃO (CLI)")
    print("=" * 65)
    print(f"Total de imagens carregadas: {len(images)}")

    idx = 0
    while 0 <= idx < len(images):
        img = images[idx]
        image_id = img["id"]
        file_path = img.get("file_path", "")
        file_name = img.get("file_name", "")
        people_raw = img.get("people_present") or "[]"
        try:
            p_list = json.loads(people_raw)
            p_str = ", ".join(p_list) if isinstance(p_list, list) else str(people_raw)
        except Exception:
            p_str = str(people_raw)

        raw_type = img.get("image_type", "unclassified")
        type_badges = {
            "photo": "Foto Real",
            "screenshot": "Print de Tela",
            "icon_or_graphic": "Ícone / Asset",
            "other": "Outro / Documento",
            "unclassified": "Não Classificada"
        }
        type_str = type_badges.get(raw_type, raw_type or "Não Classificada")

        print("-" * 65)
        print(f"Foto [{idx + 1}/{len(images)}] (ID: {image_id}) — {file_name}")
        print(f" Tipo: {type_str}")
        print(f" Caminho: {file_path}")
        print(f" Pessoas presentes: {p_str if p_str else '(Nenhuma pessoa registrada)'}")
        print(f" Descrição: {img.get('description') or '(Sem descrição)'}")

        print("\nOpções:")
        print(" [o] Abrir foto no visualizador padrão do sistema operacional")
        print(" [r] Rotacionar foto (90° horário / anti-horário / 180°)")
        print(" [c] Criar recorte/crop manual de pessoa")
        print(" [n] Nomear / Adicionar pessoa nesta foto")
        print(" [d] Detectar rostos com algoritmo de visão computacional")
        print(" [ENTER / p] Próxima foto")
        print(" [a] Foto anterior")
        print(" [q] Sair do visualizador")

        try:
            choice = input("\nEscolha uma opção: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == 'q':
            break
        elif choice == 'o':
            open_file_in_system(file_path)
        elif choice == 'r':
            print("\nOpções de Rotação:")
            print(" [1] 90° Horário (+90°)")
            print(" [2] 90° Anti-horário (-90°)")
            print(" [3] 180°")
            try:
                r_opt = input("Escolha [1/2/3, Padrão: 1]: ").strip()
            except (EOFError, KeyboardInterrupt):
                r_opt = "1"
            ang = 180 if r_opt == '3' else (-90 if r_opt == '2' else 90)
            if rotate_image_file(file_path, ang):
                print(f"[OK] Foto rotacionada {ang}° com sucesso!")
                images = get_all_images(db_path, person_label=filter_person)
            else:
                print(f"[!] Falha ao rotacionar imagem {file_path}")
        elif choice == 'c':
            if not os.path.isfile(file_path):
                print("[!] Arquivo de imagem não encontrado no disco.")
            else:
                try:
                    with Image.open(file_path) as im:
                        w, h = im.size
                    print(f"\nDimensões da imagem original: {w}x{h} px")
                    print("Informe as coordenadas em pixels (ou pressione ENTER para recorte central):")
                    coords_in = input("Formato 'left, top, right, bottom' (ex: 50, 50, 200, 200): ").strip()
                    if coords_in and "," in coords_in:
                        parts = [int(p.strip()) for p in coords_in.split(",")]
                        box = (parts[0], parts[1], parts[2], parts[3])
                    else:
                        margin_x = int(w * 0.25)
                        margin_y = int(h * 0.15)
                        box = (margin_x, margin_y, w - margin_x, h - margin_y)

                    p_name = input("Digite o nome da pessoa para este recorte: ").strip()
                    if not p_name:
                        p_name = f"Pessoa {len(images) + 1}"

                    crops_dir = Path(db_path).parent / "face_crops" if db_path else Path(DEFAULT_CROPS_DIR)
                    crops_dir.mkdir(parents=True, exist_ok=True)
                    safe_label = "".join(c if c.isalnum() else "_" for c in p_name.lower())
                    out_crop = str((crops_dir / f"crop_{safe_label}.jpg").resolve())

                    crop_pil, crop_bytes, box_norm = create_manual_face_crop(file_path, box, output_crop_path=out_crop)
                    if crop_pil:
                        cur_list = []
                        try:
                            cur_list = json.loads(img.get("people_present") or "[]")
                            if not isinstance(cur_list, list):
                                cur_list = []
                        except Exception:
                            cur_list = []
                        if p_name not in cur_list:
                            cur_list.append(p_name)
                            update_image_people_present(db_path, image_id, cur_list)

                        bbox_json = json.dumps(box_norm) if box_norm else None
                        known = [p["person_label"] for p in get_known_people(db_path)]
                        if p_name not in known:
                            register_known_person(
                                db_path,
                                person_label=p_name,
                                description="Identificado por recorte manual.",
                                first_seen_image_id=image_id,
                                face_crop_path=out_crop,
                                face_bbox=bbox_json
                            )
                        else:
                            update_person_face_crop(db_path, p_name, out_crop, face_bbox=bbox_json)
                        print(f"[OK] Recorte manual criado e '{p_name}' salvo com sucesso!")
                        images = get_all_images(db_path, person_label=filter_person)
                except Exception as ex:
                    print(f"[!] Erro ao criar recorte manual: {ex}")
        elif choice == 'n':
            new_p = input("Digite o nome da pessoa a adicionar/atribuir nesta foto: ").strip()
            if new_p:
                try:
                    cur_list = json.loads(img.get("people_present") or "[]")
                    if not isinstance(cur_list, list):
                        cur_list = []
                except Exception:
                    cur_list = []
                if new_p not in cur_list:
                    cur_list.append(new_p)
                update_image_people_present(db_path, image_id, cur_list)
                print(f"[OK] '{new_p}' adicionado à foto {image_id}.")
                images = get_all_images(db_path, person_label=filter_person)
        elif choice == 'd':
            if os.path.isfile(file_path):
                crops = extract_face_crops_in_memory(file_path)
                print(f"[OK] Detectado(s) {len(crops)} rosto(s) na foto.")
                if crops:
                    p_names = [f"Pessoa {i + 1}" for i in range(len(crops))]
                    update_image_people_present(db_path, image_id, p_names)
                    images = get_all_images(db_path, person_label=filter_person)
            else:
                print("[!] Arquivo de imagem não encontrado no disco.")
        elif choice == 'a':
            if idx > 0:
                idx -= 1
            else:
                print("Você já está na primeira foto.")
        else:
            idx += 1


def start_photo_viewer_gui(
    db_path: str = DEFAULT_DB_PATH,
    initial_image_id: Optional[int] = None,
    filter_person: Optional[str] = None,
    parent_root: Optional[Any] = None
) -> Optional[PhotoViewerGUI]:
    """Inicia a interface gráfica do visualizador de fotos."""
    try:
        app = PhotoViewerGUI(
            db_path=db_path,
            initial_image_id=initial_image_id,
            filter_person=filter_person,
            parent_root=parent_root
        )
        if not app.is_toplevel:
            app.start()
        return app
    except Exception as e:
        logger.warning(f"Não foi possível iniciar o visualizador gráfico ({e}). Iniciando modo CLI...")
        run_cli_photo_viewer(db_path=db_path, filter_person=filter_person)
        return None
