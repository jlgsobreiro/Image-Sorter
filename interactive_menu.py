"""
Módulo de menus e painéis interativos para o ImageSorter.
Fornece interface interativa completa via Terminal (CLI) e Painel Gráfico (GUI Tkinter)
para todas as funcionalidades do projeto (Scan, Describe, Identificação de Pessoas, Busca e Diagnósticos).
"""
import os
import sys
import io
import json
import logging
import subprocess
import threading
import queue
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from PIL import Image, ImageTk, ImageDraw, ImageFont

from db import (
    DEFAULT_DB_PATH,
    get_statistics,
    get_known_people,
    get_people_summary,
    get_person_images,
    search_images,
    init_db
)
from scan_images import scan_and_save_images, interactive_scan_prompt, DEFAULT_IMAGE_EXTENSIONS
from describe_images import (
    process_images,
    interactive_describe_prompt,
    get_available_ollama_models,
    reset_errors_to_pending,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_OLLAMA_URL
)
from interactive_namer import start_interactive_namer, InteractiveNamerGUI, DEFAULT_CROPS_DIR
from duplicate_finder import (
    start_duplicate_review_gui,
    run_interactive_duplicates_cli,
    export_unique_images,
    calculate_and_store_hashes,
    select_directory_via_explorer
)
from photo_viewer import start_photo_viewer_gui, run_cli_photo_viewer
from image_classifier import (
    classify_images_batch,
    export_real_photos,
    run_interactive_classification_cli,
    get_image_type_statistics
)

logger = logging.getLogger("ImageSorter.Interactive")


def open_file_in_system_viewer(file_path: str) -> bool:
    """Abre um arquivo de imagem no visualizador padrão do sistema operacional."""
    if not os.path.exists(file_path):
        print(f"[!] Arquivo não encontrado: {file_path}")
        return False
    try:
        if sys.platform.startswith("win"):
            os.startfile(file_path)
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", file_path], check=False)
        else:
            subprocess.run(["xdg-open", file_path], check=False)
        return True
    except Exception as e:
        logger.error(f"Erro ao abrir arquivo {file_path}: {e}")
        return False


# =====================================================================
# EXPLORADOR E BUSCA INTERATIVA (CLI)
# =====================================================================

def run_interactive_explorer(db_path: str = DEFAULT_DB_PATH):
    """Submenu interativo para explorar e buscar imagens e pessoas catalogadas."""
    while True:
        print("\n" + "=" * 60)
        print(" [*] EXPLORADOR DE IMAGENS E PESSOAS")
        print("=" * 60)
        print(" [1] Listar todas as pessoas catalogadas e quantidade de fotos")
        print(" [2] Buscar e listar fotos de uma pessoa específica")
        print(" [3] Buscar imagens por texto na descrição ou nome de arquivo")
        print(" [4] Listar imagens por status (pendentes / processadas / erros)")
        print(" [0] Voltar ao Menu Principal")
        print("-" * 60)

        try:
            choice = input("Escolha uma opção [0-4]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "0":
            break
        elif choice == "1":
            summary = get_people_summary(db_path)
            if not summary:
                print("\n[i] Nenhuma pessoa catalogada no banco ainda.")
            else:
                print(f"\n[+] Pessoas Catalogadas ({len(summary)}):")
                print("-" * 60)
                for i, p in enumerate(summary, start=1):
                    has_crop = "Rosto recortado" if p.get("face_crop_path") and os.path.exists(p["face_crop_path"]) else "Sem recorte"
                    print(f" [{i}] {p['person_label']} — {p['photo_count']} foto(s) [{has_crop}]")
                    print(f"     Descrição: {p['description']}")
                print("-" * 60)

        elif choice == "2":
            summary = get_people_summary(db_path)
            if not summary:
                print("\n[i] Nenhuma pessoa catalogada no banco de dados.")
                continue

            print("\nSelecione a pessoa para ver as fotos:")
            for i, p in enumerate(summary, start=1):
                print(f" [{i}] {p['person_label']} ({p['photo_count']} fotos)")

            try:
                p_choice = input(f"Digite o número ou nome da pessoa [1-{len(summary)}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue

            target_label = None
            if p_choice.isdigit():
                idx = int(p_choice)
                if 1 <= idx <= len(summary):
                    target_label = summary[idx - 1]["person_label"]
            else:
                target_label = p_choice

            if not target_label:
                continue

            photos = get_person_images(db_path, target_label)
            print(f"\n[+] {len(photos)} foto(s) encontrada(s) para '{target_label}':")
            for i, ph in enumerate(photos, start=1):
                exists_tag = "[OK]" if os.path.exists(ph["file_path"]) else "[!] (Arquivo nao encontrado)"
                print(f" [{i}] ID: {ph['id']} | {ph['file_name']} {exists_tag}")
                print(f"     Caminho: {ph['file_path']}")
                if ph["description"]:
                    desc_prev = ph["description"].replace("\n", " ")[:100] + "..."
                    print(f"     Descrição: {desc_prev}")

            if photos:
                try:
                    open_choice = input("\nDigite o número da foto para abrir no visualizador padrão (ou ENTER para voltar): ").strip()
                    if open_choice.isdigit():
                        p_idx = int(open_choice)
                        if 1 <= p_idx <= len(photos):
                            selected_photo = photos[p_idx - 1]
                            print(f"Abrindo: {selected_photo['file_path']}...")
                            open_file_in_system_viewer(selected_photo["file_path"])
                except (EOFError, KeyboardInterrupt):
                    pass

        elif choice == "3":
            try:
                term = input("Digite o termo de busca (descrição ou nome): ").strip()
            except (EOFError, KeyboardInterrupt):
                continue

            if not term:
                continue

            results = search_images(db_path, query_text=term, limit=30)
            print(f"\n[+] {len(results)} resultado(s) para '{term}':")
            for i, r in enumerate(results, start=1):
                print(f" [{i}] ID: {r['id']} | {r['file_name']} (Status: {r['status']})")
                print(f"     Caminho: {r['file_path']}")
                if r["description"]:
                    print(f"     Descrição: {r['description'][:120]}...")

            if results:
                try:
                    open_choice = input("\nDigite o número para abrir a imagem (ou ENTER para voltar): ").strip()
                    if open_choice.isdigit():
                        idx = int(open_choice)
                        if 1 <= idx <= len(results):
                            open_file_in_system_viewer(results[idx - 1]["file_path"])
                except (EOFError, KeyboardInterrupt):
                    pass

        elif choice == "4":
            print("\nFiltrar por status:")
            print(" [1] Imagens pendentes (pending)")
            print(" [2] Imagens processadas com sucesso (processed)")
            print(" [3] Imagens com erro (error)")
            try:
                s_opt = input("Escolha [1-3]: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue

            status_map = {"1": "pending", "2": "processed", "3": "error"}
            chosen_status = status_map.get(s_opt)
            if not chosen_status:
                continue

            results = search_images(db_path, status=chosen_status, limit=25)
            print(f"\n[+] {len(results)} imagem(ns) com status '{chosen_status}':")
            for i, r in enumerate(results, start=1):
                print(f" [{i}] ID: {r['id']} | {r['file_name']}")
                print(f"     Caminho: {r['file_path']}")
                if r["error_message"]:
                    print(f"     Erro: {r['error_message']}")


# =====================================================================
# MENU PRINCIPAL INTERATIVO (CLI)
# =====================================================================

def run_interactive_cli(db_path: str = DEFAULT_DB_PATH):
    """
    Menu principal interativo via console para todas as operações do ImageSorter.
    """
    while True:
        stats = get_statistics(db_path)
        print("\n" + "=" * 65)
        print(" [*] IMAGESORTER - PAINEL INTERATIVO DE GERENCIAMENTO")
        print("=" * 65)
        print(f" Banco de dados: {os.path.abspath(db_path)}")
        print(f" Imagens: {stats['total']} total | {stats['pending']} pendentes | {stats['processed']} processadas | {stats['errors']} erros")
        print(f" Pessoas Únicas Catalogadas: {stats.get('known_people', 0)}")
        print(f" Duplicatas: {stats.get('duplicate_groups', 0)} grupo(s) ({stats.get('duplicate_images_total', 0)} fotos repetidas)")
        print(f" Tipos: {stats.get('photos_count', 0)} Fotos Reais | {stats.get('screenshots_count', 0)} Prints | {stats.get('icons_count', 0)} Ícones | {stats.get('unclassified_count', 0)} Não Classificadas")
        print("=" * 65)
        print(" [1] Escanear Disco ou Diretório (Scan de Imagens)")
        print(" [2] Avaliar & Descrever Fotos (Biometria / Ollama)")
        print(" [3] Nomear & Unificar Pessoas (Catálogo de Rostos)")
        print(" [4] Visualizador Interativo de Fotos (Ver & Nomear Pessoas na Foto)")
        print(" [5] Identificar & Gerenciar Imagens Duplicadas (Ciclar & Comparar)")
        print(" [6] Exportar Imagens Únicas para Pasta (Seletor Explorer)")
        print(" [7] Classificar Tipos de Imagem com IA (Fotos vs Prints vs Ícones)")
        print(" [8] Exportar Apenas Fotos Reais para Pasta (Seletor Explorer)")
        print(" [9] Explorar e Buscar Imagens / Pessoas")
        print(" [10] Executar Pipeline Completo (Scan -> Describe -> Namer)")
        print(" [11] Ver Estatísticas Detalhadas do Banco")
        print(" [12] Abrir Painel Gráfico Geral (Dashboard GUI)")
        print(" [0] Sair")
        print("-" * 65)

        try:
            choice = input("Escolha uma opção [0-12]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando ImageSorter. Até logo!")
            break

        if choice == "0":
            print("\nEncerrando ImageSorter. Até logo!")
            break

        elif choice == "1":
            interactive_scan_prompt(db_path=db_path)

        elif choice == "2":
            print("\nEscolha a interface para avaliação:")
            print(" [1] Janela Gráfica Interativa (Mostra Imagem, Rostos e Texto IA em tempo real) [Recomendado]")
            print(" [2] Modo Terminal (CLI com prompt interativo)")
            try:
                sub_c = input("Opção [1/2, Padrão: 1]: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if sub_c == "2":
                interactive_describe_prompt(db_path=db_path)
            else:
                start_live_evaluator_gui(db_path=db_path)

        elif choice == "3":
            print("\nEscolha a interface para gerenciamento e nomeação de pessoas:")
            print(" [1] Janela Gráfica Geral (GUI Tkinter com visualização do rosto) [Recomendado]")
            print(" [2] Revisão Facial Biométrica de Pessoas Semelhantes (Sugestões de Merge)")
            print(" [3] Modo Terminal (CLI)")
            try:
                sub_c = input("Opção [1/2/3, Padrão: 1]: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if sub_c == "2":
                from face_identity_review import run_cli_similar_people_review
                from interactive_namer import SimilarPeopleReviewGUI
                try:
                    import tkinter as tk
                    root = tk.Tk()
                    root.withdraw()
                    gui = SimilarPeopleReviewGUI(db_path=db_path, parent_root=root)
                    gui.root.protocol("WM_DELETE_WINDOW", root.destroy)
                    root.mainloop()
                except Exception:
                    run_cli_similar_people_review(db_path=db_path)
            elif sub_c == "3":
                start_interactive_namer(db_path=db_path, cli=True)
            else:
                start_interactive_namer(db_path=db_path, cli=False)

        elif choice == "4":
            print("\nEscolha o modo do visualizador de fotos:")
            print(" [1] Visualizador Gráfico Completo (com marcação de rostos e edição manual) [Recomendado]")
            print(" [2] Modo Terminal (CLI)")
            try:
                sub_v = input("Opção [1/2, Padrão: 1]: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            if sub_v == "2":
                run_cli_photo_viewer(db_path=db_path)
            else:
                start_photo_viewer_gui(db_path=db_path)

        elif choice == "5":
            run_interactive_duplicates_cli(db_path=db_path)

        elif choice == "6":
            print("\nIniciando exportação de imagens únicas...")
            export_unique_images(db_path=db_path, open_explorer_on_complete=True)

        elif choice == "7":
            run_interactive_classification_cli(db_path=db_path)

        elif choice == "8":
            unique_in = input("\nExportar apenas fotos únicas (descartar duplicatas)? [S/n]: ").strip().lower()
            only_unique = unique_in not in ("n", "nao", "não", "no")
            export_real_photos(db_path=db_path, only_unique=only_unique, open_explorer_on_complete=True)

        elif choice == "9":
            run_interactive_explorer(db_path=db_path)

        elif choice == "10":
            print("\n" + "=" * 60)
            print(" [*] PIPELINE COMPLETO GUIADO")
            print("=" * 60)
            print("Etapa 1: Varredura de Imagens")
            scan_res = interactive_scan_prompt(db_path=db_path)
            if scan_res is not None:
                print("\nEtapa 2: Avaliação Visual")
                interactive_describe_prompt(db_path=db_path)
                print("\nEtapa 3: Identificação de Pessoas")
                try:
                    open_namer = input("Deseja abrir agora a janela de nomeação de pessoas? [S/n]: ").strip().lower()
                    if open_namer not in ("n", "nao", "não"):
                        start_interactive_namer(db_path=db_path, cli=False)
                except (EOFError, KeyboardInterrupt):
                    pass

        elif choice == "11":
            from main import show_status
            show_status(db_path)

        elif choice == "12":
            start_interactive_dashboard_gui(db_path=db_path)

        else:
            print("Opção inválida. Por favor, escolha um número de 0 a 12.")


# =====================================================================
# AVALIADOR VISUAL INTERATIVO AO VIVO (GUI TKINTER - 3 QUADROS)
# =====================================================================

def draw_face_boxes_on_image(pil_img: Image.Image, face_crops: List[Dict[str, Any]]) -> Image.Image:
    """
    Desenha caixas delimitadoras coloridas e rótulos de identificação sobre os rostos detectados.
    """
    if not face_crops:
        return pil_img
    img_copy = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size
    line_w = max(2, int(min(w, h) * 0.006))

    for i, crop in enumerate(face_crops, start=1):
        box = crop.get("original_face_bbox") or crop.get("bbox")
        if not box or len(box) != 4:
            continue
        l, t, r, b = box
        label = crop.get("person_label") or f"Rosto {i}"

        # Bounding box em verde vivo
        draw.rectangle([l, t, r, b], outline="#00FF66", width=line_w)

        # Fundo do texto da etiqueta
        tag_text = f" {label} "
        box_tag_h = max(18, int(min(w, h) * 0.04))
        tag_top = max(0, t - box_tag_h)
        tag_bottom = max(box_tag_h, t)
        tag_right = min(w, l + len(tag_text) * 9 + 8)
        draw.rectangle([l, tag_top, tag_right, tag_bottom], fill="#004411")
        draw.text((l + 3, tag_top + 2), tag_text, fill="#FFFFFF")

    return img_copy


class InteractiveLiveEvaluatorGUI:
    """
    Interface gráfica interativa para avaliação visual com IA em tempo real.
    Apresenta 3 quadros sincronizados:
    1. Quadro Esquerdo: Imagem a ser avaliada (com marcação das faces detectadas)
    2. Quadro Direito: Galeria com recortes dos rostos detectados e perfis fisionômicos
    3. Quadro Inferior: Painel de texto da IA com pessoas identificadas e descrição contextual
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, parent_root: Optional[Any] = None):
        import tkinter as tk
        from tkinter import ttk, messagebox, scrolledtext

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.scrolledtext = scrolledtext
        self.db_path = db_path

        self.is_toplevel = parent_root is not None
        if self.is_toplevel:
            self.root = tk.Toplevel(parent_root)
        else:
            self.root = tk.Tk()

        self.root.title("ImageSorter — Avaliador Visual Interativo com IA (Ao Vivo)")
        self.root.geometry("1060x820")
        self.root.minsize(880, 640)

        # Controle de concorrência e estado
        self.events_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.pause_event = threading.Event()
        self.pause_event.set()  # Começa não-pausado
        self.stop_event = threading.Event()
        self.is_running = False
        self.step_by_step_mode = tk.BooleanVar(value=False)

        # Dados da imagem atual
        self.current_pil_image: Optional[Image.Image] = None
        self.current_face_crops: List[Dict[str, Any]] = []
        self._keep_refs: List[Any] = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._schedule_queue_check()

    def _build_ui(self):
        main_frame = self.ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=self.tk.BOTH, expand=True)

        # -------------------------------------------------------------
        # BARRA SUPERIOR: CONTROLES E CONFIGURAÇÕES
        # -------------------------------------------------------------
        ctrl_frame = self.ttk.LabelFrame(main_frame, text=" Configurações & Controles da IA ", padding="8")
        ctrl_frame.pack(fill=self.tk.X, pady=(0, 8))

        # Linha 1 de controles
        row1 = self.ttk.Frame(ctrl_frame)
        row1.pack(fill=self.tk.X, pady=(0, 5))

        self.ttk.Label(row1, text="Modelo Ollama:", font=("Segoe UI", 9, "bold")).pack(side=self.tk.LEFT, padx=(0, 6))
        self.model_var = self.tk.StringVar(value=DEFAULT_MODEL)
        self.combo_models = self.ttk.Combobox(row1, textvariable=self.model_var, width=22)
        self.combo_models.pack(side=self.tk.LEFT, padx=(0, 6))

        btn_refresh = self.ttk.Button(row1, text="🔄 Detectar", command=self._refresh_models)
        btn_refresh.pack(side=self.tk.LEFT, padx=(0, 15))

        self.ttk.Label(row1, text="Limite:").pack(side=self.tk.LEFT, padx=(0, 4))
        self.limit_var = self.tk.StringVar(value="")
        self.entry_limit = self.ttk.Entry(row1, textvariable=self.limit_var, width=6)
        self.entry_limit.pack(side=self.tk.LEFT, padx=(0, 15))

        self.retry_errors_var = self.tk.BooleanVar(value=False)
        self.ttk.Checkbutton(row1, text="Retentar Erros", variable=self.retry_errors_var).pack(side=self.tk.LEFT, padx=(0, 15))

        self.use_face_det_var = self.tk.BooleanVar(value=True)
        self.ttk.Checkbutton(row1, text="Detecção Facial Rápida (CPU)", variable=self.use_face_det_var).pack(side=self.tk.LEFT, padx=(0, 15))

        self.describe_ai_var = self.tk.BooleanVar(value=True)
        self.ttk.Checkbutton(row1, text="Descrever Foto com IA (Ollama)", variable=self.describe_ai_var).pack(side=self.tk.LEFT)

        # Linha 2 de controles: Modo e Botões de Ação
        row2 = self.ttk.Frame(ctrl_frame)
        row2.pack(fill=self.tk.X, pady=(5, 0))

        self.ttk.Label(row2, text="Fluxo:").pack(side=self.tk.LEFT, padx=(0, 6))
        self.ttk.Radiobutton(row2, text="🔄 Modo Contínuo", variable=self.step_by_step_mode, value=False).pack(side=self.tk.LEFT, padx=(0, 8))
        self.ttk.Radiobutton(row2, text="⏸️ Passo a Passo", variable=self.step_by_step_mode, value=True).pack(side=self.tk.LEFT, padx=(0, 20))

        self.btn_start = self.ttk.Button(row2, text="▶️ Iniciar Avaliação", command=self.start_evaluation)
        self.btn_start.pack(side=self.tk.LEFT, padx=(0, 6))

        self.btn_pause = self.ttk.Button(row2, text="⏸️ Pausar", command=self.toggle_pause, state=self.tk.DISABLED)
        self.btn_pause.pack(side=self.tk.LEFT, padx=(0, 6))

        self.btn_next = self.ttk.Button(row2, text="⏭️ Próxima Imagem", command=self.step_next, state=self.tk.DISABLED)
        self.btn_next.pack(side=self.tk.LEFT, padx=(0, 6))

        self.btn_stop = self.ttk.Button(row2, text="⏹️ Parar", command=self.stop_evaluation, state=self.tk.DISABLED)
        self.btn_stop.pack(side=self.tk.LEFT, padx=(0, 10))

        # Carrega lista inicial de modelos Ollama em segundo plano
        threading.Thread(target=self._async_load_models, daemon=True).start()

        # -------------------------------------------------------------
        # ÁREA CENTRAL: QUADROS 1, 2 E 3
        # -------------------------------------------------------------
        # PanedWindow vertical para dividir parte superior (Imagens/Rostos) e inferior (Texto IA)
        self.v_pane = self.ttk.PanedWindow(main_frame, orient=self.tk.VERTICAL)
        self.v_pane.pack(fill=self.tk.BOTH, expand=True, pady=(0, 6))

        # Painel Superior (dividido horizontalmente em Quadro 1 e Quadro 2)
        top_pane = self.ttk.PanedWindow(self.v_pane, orient=self.tk.HORIZONTAL)
        self.v_pane.add(top_pane, weight=3)

        # -------------------------------------------------------------
        # QUADRO 1: IMAGEM A SER AVALIADA
        # -------------------------------------------------------------
        self.frame_img = self.ttk.LabelFrame(top_pane, text=" 🖼️ Imagem em Avaliação ", padding="8")
        top_pane.add(self.frame_img, weight=3)

        self.lbl_img_info = self.ttk.Label(
            self.frame_img,
            text="Aguardando início do processamento...",
            font=("Segoe UI", 9, "bold"),
            foreground="#333333"
        )
        self.lbl_img_info.pack(anchor=self.tk.W, pady=(0, 4))

        self.lbl_img_display = self.ttk.Label(
            self.frame_img,
            text="[ Nenhuma imagem carregada ]\n\nClique em '▶️ Iniciar Avalia��ão' acima para começar.",
            anchor=self.tk.CENTER,
            justify=self.tk.CENTER,
            background="#1e1e1e",
            foreground="#aaaaaa",
            font=("Segoe UI", 10)
        )
        self.lbl_img_display.pack(fill=self.tk.BOTH, expand=True, pady=4)

        self.lbl_img_status = self.ttk.Label(
            self.frame_img,
            text="Status: Ocioso",
            font=("Segoe UI", 9, "italic"),
            foreground="#555555"
        )
        self.lbl_img_status.pack(anchor=self.tk.W, pady=(4, 0))

        # -------------------------------------------------------------
        # QUADRO 2: ROSTOS DETECTADOS
        # -------------------------------------------------------------
        self.frame_faces = self.ttk.LabelFrame(top_pane, text=" 👤 Rostos Detectados na Imagem ", padding="8")
        top_pane.add(self.frame_faces, weight=2)

        self.lbl_faces_count = self.ttk.Label(
            self.frame_faces,
            text="Rostos detectados: 0",
            font=("Segoe UI", 9, "bold"),
            foreground="#333333"
        )
        self.lbl_faces_count.pack(anchor=self.tk.W, pady=(0, 4))

        # Canvas com Scrollbar para os cartões de rostos
        faces_canvas_frame = self.ttk.Frame(self.frame_faces)
        faces_canvas_frame.pack(fill=self.tk.BOTH, expand=True)

        self.faces_canvas = self.tk.Canvas(faces_canvas_frame, background="#f9f9f9", highlightthickness=0)
        self.faces_scrollbar = self.ttk.Scrollbar(faces_canvas_frame, orient=self.tk.VERTICAL, command=self.faces_canvas.yview)
        self.faces_container = self.ttk.Frame(self.faces_canvas)

        self.faces_container.bind(
            "<Configure>",
            lambda e: self.faces_canvas.configure(scrollregion=self.faces_canvas.bbox("all"))
        )
        self.faces_canvas_window = self.faces_canvas.create_window((0, 0), window=self.faces_container, anchor="nw")
        self.faces_canvas.configure(yscrollcommand=self.faces_scrollbar.set)

        self.faces_canvas.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)
        self.faces_scrollbar.pack(side=self.tk.RIGHT, fill=self.tk.Y)

        self._show_faces_placeholder("Nenhum rosto detectado ainda.")

        # -------------------------------------------------------------
        # QUADRO 3: TEXTO DA IA & DESCRIÇÃO
        # -------------------------------------------------------------
        self.frame_ai = self.ttk.LabelFrame(self.v_pane, text=" 🤖 Texto & Diagnóstico da IA (Ollama Vision) ", padding="8")
        self.v_pane.add(self.frame_ai, weight=2)

        # Cabeçalho de pessoas identificadas
        self.lbl_people_tags = self.ttk.Label(
            self.frame_ai,
            text="Pessoas Presentes: [ Aguardando análise da IA ]",
            font=("Segoe UI", 10, "bold"),
            foreground="#0055aa"
        )
        self.lbl_people_tags.pack(anchor=self.tk.W, pady=(0, 4))

        # Caixa de texto estruturada
        self.txt_ai_output = self.scrolledtext.ScrolledText(
            self.frame_ai,
            wrap=self.tk.WORD,
            height=6,
            font=("Consolas", 10),
            background="#fafafa",
            foreground="#222222"
        )
        self.txt_ai_output.pack(fill=self.tk.BOTH, expand=True, pady=4)
        self.txt_ai_output.insert(self.tk.END, "As descrições da cena e características das pessoas serão exibidas aqui em tempo real durante o processamento.")
        self.txt_ai_output.configure(state=self.tk.DISABLED)

        # -------------------------------------------------------------
        # BARRA DE STATUS INFERIOR
        # -------------------------------------------------------------
        status_bar = self.ttk.Frame(main_frame)
        status_bar.pack(fill=self.tk.X, pady=(4, 0))

        self.progressbar = self.ttk.Progressbar(status_bar, orient=self.tk.HORIZONTAL, mode="determinate")
        self.progressbar.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=(0, 15))

        self.lbl_status_stats = self.ttk.Label(
            status_bar,
            text="Pronto para iniciar.",
            font=("Segoe UI", 9)
        )
        self.lbl_status_stats.pack(side=self.tk.RIGHT)

    def _async_load_models(self):
        models = get_available_ollama_models(DEFAULT_OLLAMA_URL)
        if models:
            self.root.after(0, lambda: self._apply_model_list(models))

    def _apply_model_list(self, models: List[str]):
        self.combo_models["values"] = models
        if self.model_var.get() not in models:
            self.model_var.set(models[0])

    def _refresh_models(self):
        models = get_available_ollama_models(DEFAULT_OLLAMA_URL)
        if models:
            self._apply_model_list(models)
            self.messagebox.showinfo("Ollama", f"{len(models)} modelo(s) detectado(s) com sucesso!")
        else:
            self.messagebox.showwarning("Ollama", "Nenhum modelo detectado ou servidor inacessível.")

    def _show_faces_placeholder(self, message: str):
        for widget in self.faces_container.winfo_children():
            widget.destroy()
        lbl = self.ttk.Label(
            self.faces_container,
            text=message,
            font=("Segoe UI", 9, "italic"),
            foreground="#777777",
            wraplength=260,
            justify=self.tk.CENTER
        )
        lbl.pack(pady=30, padx=15)

    def _render_image_to_quadro1(self, pil_img: Image.Image, face_crops: Optional[List[Dict[str, Any]]] = None):
        """Redimensiona e renderiza a imagem no Quadro 1 com destaque visual nas faces."""
        try:
            if face_crops:
                annotated = draw_face_boxes_on_image(pil_img, face_crops)
            else:
                annotated = pil_img.copy()

            # Calcula dimensões proporcionais
            max_w = max(380, self.lbl_img_display.winfo_width() - 20)
            max_h = max(280, self.lbl_img_display.winfo_height() - 20)
            if max_w < 100:
                max_w = 460
            if max_h < 100:
                max_h = 340

            annotated.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(annotated)
            self._keep_refs.append(photo)

            self.lbl_img_display.configure(image=photo, text="")
        except Exception as e:
            logger.debug(f"Erro ao renderizar imagem no Quadro 1: {e}")

    def _set_ai_text(self, text: str, people_list: Optional[List[str]] = None):
        """Atualiza a caixa de texto do Quadro 3."""
        if people_list is not None:
            if people_list:
                self.lbl_people_tags.configure(
                    text=f"Pessoas Presentes: {', '.join(people_list)}",
                    foreground="#007722"
                )
            else:
                self.lbl_people_tags.configure(
                    text="Pessoas Presentes: [ Nenhuma pessoa identificada na foto ]",
                    foreground="#666666"
                )

        self.txt_ai_output.configure(state=self.tk.NORMAL)
        self.txt_ai_output.delete("1.0", self.tk.END)
        self.txt_ai_output.insert(self.tk.END, text)
        self.txt_ai_output.configure(state=self.tk.DISABLED)

    def _schedule_queue_check(self):
        """Verifica a fila de eventos periódicos da thread de IA de forma thread-safe."""
        try:
            while not self.events_queue.empty():
                event_data = self.events_queue.get_nowait()
                self._handle_worker_event(event_data)
        except Exception as e:
            logger.debug(f"Erro no processamento da fila GUI: {e}")

        if not self.stop_event.is_set():
            self.root.after(40, self._schedule_queue_check)

    def _handle_worker_event(self, data: Dict[str, Any]):
        """Trata cada evento emitido pelo `process_images` atualizando os 3 quadros em tempo real."""
        event = data.get("event")

        if event == "start_image":
            img_id = data["image_id"]
            file_path = data["file_path"]
            idx = data["index"]
            total = data["total"]

            pct = int(((idx - 1) / total) * 100) if total > 0 else 0
            self.progressbar.configure(maximum=total, value=idx - 1)
            self.lbl_status_stats.configure(text=f"Processando imagem {idx} de {total} ({pct}%) [ID {img_id}]...")

            # Carrega imagem principal
            try:
                pil_img = Image.open(file_path).convert("RGB")
                self.current_pil_image = pil_img
                self.current_face_crops = []
                self._keep_refs.clear()

                w, h = pil_img.size
                file_name = os.path.basename(file_path)
                self.lbl_img_info.configure(text=f"ID {img_id}: {file_name} ({w}x{h} px)")
                self.lbl_img_status.configure(text="Status: 🔍 Detectando rostos com visão computacional...")
                self._render_image_to_quadro1(pil_img)
            except Exception as ie:
                self.lbl_img_info.configure(text=f"ID {img_id}: Erro ao carregar arquivo")
                self.lbl_img_status.configure(text=f"Falha: {ie}")

            self._show_faces_placeholder("Detectando rostos na imagem...")
            self._set_ai_text("Iniciando avaliação visual...", people_list=[])

        elif event == "faces_detected":
            face_crops = data.get("face_crops", [])
            count = len(face_crops)
            self.current_face_crops = face_crops
            self.lbl_faces_count.configure(text=f"Rostos detectados: {count}")

            if self.current_pil_image:
                self._render_image_to_quadro1(self.current_pil_image, face_crops)

            if count > 0:
                self.lbl_img_status.configure(text=f"Status: 👤 {count} rosto(s) detectado(s). Enviando recortes para a IA...")
                # Cria cartões vazios para cada face
                for widget in self.faces_container.winfo_children():
                    widget.destroy()

                for f_idx, crop in enumerate(face_crops, start=1):
                    card = self.ttk.Frame(self.faces_container, padding="6", relief="solid")
                    card.pack(fill=self.tk.X, expand=True, pady=4, padx=4)

                    # Thumbnail
                    thumb = crop.get("pil_image")
                    if thumb:
                        thumb_copy = thumb.copy()
                        thumb_copy.thumbnail((75, 75), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(thumb_copy)
                        self._keep_refs.append(photo)
                        lbl_th = self.ttk.Label(card, image=photo)
                        lbl_th.pack(side=self.tk.LEFT, padx=(0, 8))

                    info_frame = self.ttk.Frame(card)
                    info_frame.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)

                    lbl_title = self.ttk.Label(
                        info_frame,
                        text=f"Rosto {f_idx} de {count}",
                        font=("Segoe UI", 9, "bold")
                    )
                    lbl_title.pack(anchor=self.tk.W)

                    lbl_state = self.ttk.Label(
                        info_frame,
                        text="Aguardando reconhecimento pela IA...",
                        font=("Segoe UI", 8, "italic"),
                        foreground="#555555",
                        wraplength=200
                    )
                    lbl_state.pack(anchor=self.tk.W, pady=(2, 0))

                    crop["_gui_card"] = card
                    crop["_gui_title"] = lbl_title
                    crop["_gui_state"] = lbl_state
            else:
                self.lbl_img_status.configure(text="Status: ⏭️ Nenhum rosto detectado. Pulando para a próxima imagem...")
                self._show_faces_placeholder("Nenhum rosto detectado na imagem.\nAvaliação com IA dispensada.")

        elif event == "image_skipped_no_faces":
            self.lbl_img_status.configure(text="Status: ⏭️ Imagem sem rostos processada (avaliação com IA dispensada).")

        elif event == "face_recognized":
            f_idx = data["face_idx"]
            p_label = data["person_label"]
            desc = data["description"]
            is_new = data["is_new"]

            # Atualiza o cartão correspondente
            if self.current_face_crops and f_idx <= len(self.current_face_crops):
                crop = self.current_face_crops[f_idx - 1]
                crop["person_label"] = p_label
                if "_gui_title" in crop:
                    badge = " [⭐ Novo]" if is_new else " [✔️ Conhecido]"
                    crop["_gui_title"].configure(text=f"{p_label}{badge}", foreground="#007722" if is_new else "#0055aa")
                if "_gui_state" in crop:
                    crop["_gui_state"].configure(text=desc[:110] + ("..." if len(desc) > 110 else ""), font=("Segoe UI", 8), foreground="#333333")

            # Re-renderiza imagem do Quadro 1 com os nomes atualizados
            if self.current_pil_image:
                self._render_image_to_quadro1(self.current_pil_image, self.current_face_crops)

            self.lbl_img_status.configure(text=f"Status: 👤 Rosto {f_idx} reconhecido como '{p_label}'.")

        elif event == "ai_thinking":
            stage = data.get("stage", "")
            if stage == "face_recognition":
                f_idx = data.get("face_idx", 1)
                tot = data.get("total_faces", 1)
                self.lbl_img_status.configure(text=f"Status: 🧠 Analisando fisionomia do rosto {f_idx}/{tot} com Ollama...")
            elif stage == "scene_description":
                self.lbl_img_status.configure(text="Status: 🧠 Gerando descrição detalhada do cenário e contexto...")
            elif stage == "full_scene":
                self.lbl_img_status.configure(text="Status: 🧠 Analisando imagem completa com Ollama Vision...")

        elif event == "image_completed":
            desc = data["description"]
            pp_str = data.get("people_present", "[]")
            duration = data.get("duration", 0.0)
            idx = data.get("processed_count", 0) + data.get("error_count", 0)
            total = data.get("total", 1)
            pct = int((idx / total) * 100) if total > 0 else 0

            try:
                people_list = json.loads(pp_str)
            except Exception:
                people_list = [pp_str] if pp_str else []

            # Quadro 3 atualizado com o texto completo da IA
            ai_text_full = (
                f"=== DESCRIÇÃO GERADA PELA IA ({duration:.2f}s) ===\n\n"
                f"{desc}\n\n"
                f"--------------------------------------------------\n"
                f"Pessoas Catalogadas: {', '.join(people_list) if people_list else 'Nenhuma'}"
            )
            self._set_ai_text(ai_text_full, people_list=people_list)

            self.lbl_img_status.configure(text=f"Status: ✅ Imagem avaliada com sucesso em {duration:.2f}s.")
            self.progressbar.configure(value=idx)
            self.lbl_status_stats.configure(
                text=f"Progresso: {idx}/{total} ({pct}%) | Concluídas: {data.get('processed_count', 0)} | Erros: {data.get('error_count', 0)}"
            )

            # Se estiver no modo passo a passo, pausa automaticamente para o usuário examinar
            if self.step_by_step_mode.get() and idx < total:
                self.pause_event.clear()
                self.btn_pause.configure(text="▶️ Retomar", state=self.tk.NORMAL)
                self.btn_next.configure(state=self.tk.NORMAL)
                self.lbl_img_status.configure(text="Status: ⏸️ Pausado (Passo a Passo). Clique em 'Próxima Imagem' para continuar.")

        elif event == "image_error":
            err_msg = data.get("error_message", "Erro desconhecido")
            self.lbl_img_status.configure(text=f"Status: ❌ Erro ao processar imagem.")
            self._set_ai_text(f"[ERRO NA AVALIAÇÃO]\n{err_msg}", people_list=[])

        elif event == "finished":
            proc = data.get("processed", 0)
            errs = data.get("errors", 0)
            tot = data.get("total", 0)
            elapsed = data.get("elapsed_seconds", 0)

            self.is_running = False
            self.btn_start.configure(state=self.tk.NORMAL)
            self.btn_pause.configure(state=self.tk.DISABLED, text="⏸️ Pausar")
            self.btn_next.configure(state=self.tk.DISABLED)
            self.btn_stop.configure(state=self.tk.DISABLED)

            self.progressbar.configure(value=tot)
            status_text = "Interrompido pelo usuário." if self.stop_event.is_set() else f"Finalizado! {proc} processada(s), {errs} erro(s) em {elapsed}s."
            self.lbl_status_stats.configure(text=status_text)
            self.lbl_img_status.configure(text=f"Status: {status_text}")

    def start_evaluation(self):
        """Inicia a thread de processamento visual com IA."""
        if self.is_running:
            return

        model = self.model_var.get().strip()
        limit_str = self.limit_var.get().strip()
        limit = int(limit_str) if limit_str.isdigit() and int(limit_str) > 0 else None
        use_face_det = self.use_face_det_var.get()
        describe_ai = self.describe_ai_var.get()

        if self.retry_errors_var.get():
            reset_errors_to_pending(self.db_path)

        self.stop_event.clear()
        self.pause_event.set()
        self.is_running = True

        self.btn_start.configure(state=self.tk.DISABLED)
        self.btn_pause.configure(state=self.tk.NORMAL, text="⏸️ Pausar")
        self.btn_next.configure(state=self.tk.DISABLED)
        self.btn_stop.configure(state=self.tk.NORMAL)

        def worker():
            try:
                process_images(
                    db_path=self.db_path,
                    model=model,
                    limit=limit,
                    use_face_detection=use_face_det,
                    describe_ai=describe_ai,
                    progress_callback=lambda evt: self.events_queue.put(evt),
                    pause_event=self.pause_event,
                    stop_event=self.stop_event
                )
            except Exception as e:
                logger.error(f"Erro na thread de avaliação: {e}")
                self.events_queue.put({"event": "image_error", "error_message": str(e)})
                self.events_queue.put({"event": "finished", "processed": 0, "errors": 1, "total": 0, "elapsed_seconds": 0})

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def toggle_pause(self):
        """Pausa ou retoma a execução contínua."""
        if not self.is_running:
            return
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.btn_pause.configure(text="▶️ Retomar")
            self.btn_next.configure(state=self.tk.NORMAL)
            self.lbl_img_status.configure(text="Status: ⏸️ Processamento pausado.")
        else:
            self.pause_event.set()
            self.btn_pause.configure(text="⏸️ Pausar")
            self.btn_next.configure(state=self.tk.DISABLED)
            self.lbl_img_status.configure(text="Status: ▶️ Retomando processamento...")

    def step_next(self):
        """Avança exatamente uma imagem quando pausado ou no modo passo a passo."""
        if not self.is_running:
            return
        self.pause_event.set()
        self.btn_next.configure(state=self.tk.DISABLED)

    def stop_evaluation(self):
        """Interrompe a execução."""
        self.stop_event.set()
        self.pause_event.set()
        self.is_running = False
        self.lbl_img_status.configure(text="Status: ⏹️ Interrompendo...")

    def _on_close(self):
        self.stop_event.set()
        self.pause_event.set()
        self.root.destroy()

    def start(self):
        self.root.mainloop()


def start_live_evaluator_gui(db_path: str = DEFAULT_DB_PATH, parent_root: Optional[Any] = None) -> InteractiveLiveEvaluatorGUI:
    """Inicia a interface interativa de avaliação visual com IA (Quadro 1: Imagem, Quadro 2: Rostos, Quadro 3: Texto IA)."""
    app = InteractiveLiveEvaluatorGUI(db_path=db_path, parent_root=parent_root)
    if parent_root is None:
        app.start()
    return app


# =====================================================================
# PAINEL GRÁFICO GERAL (GUI TKINTER)
# =====================================================================

class InteractiveDashboardGUI:
    """
    Painel gráfico integrado Tkinter para controle total do ImageSorter:
    Varredura, Processamento IA, Identificação de Pessoas e Busca.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.filedialog = filedialog
        self.db_path = db_path

        self.root = tk.Tk()
        self.root.title("ImageSorter — Painel de Controle Integrado")
        self.root.geometry("860x720")
        self.root.minsize(740, 600)

        # Controle de tarefas concorrentes e interrupção
        self.events_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_busy = False
        self.current_worker: Optional[threading.Thread] = None

        self._build_ui()
        self.refresh_stats()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._schedule_dashboard_queue_check()

    def _build_ui(self):
        # Frame Principal
        main_frame = self.ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=self.tk.BOTH, expand=True)

        # Cabeçalho
        title_label = self.ttk.Label(
            main_frame,
            text="ImageSorter — Sistema Inteligente de Imagens",
            font=("Segoe UI", 16, "bold")
        )
        title_label.pack(anchor=self.tk.W, pady=(0, 5))

        db_label = self.ttk.Label(
            main_frame,
            text=f"Banco SQLite: {os.path.abspath(self.db_path)}",
            font=("Segoe UI", 9),
            foreground="#666666"
        )
        db_label.pack(anchor=self.tk.W, pady=(0, 10))

        # Card de Estatísticas
        self.stats_frame = self.ttk.LabelFrame(main_frame, text=" Status Geral do Sistema ", padding="10")
        self.stats_frame.pack(fill=self.tk.X, pady=(0, 10))

        self.lbl_stats_total = self.ttk.Label(self.stats_frame, text="Total: -", font=("Segoe UI", 10, "bold"))
        self.lbl_stats_total.grid(row=0, column=0, padx=12, pady=5, sticky=self.tk.W)

        self.lbl_stats_pending = self.ttk.Label(self.stats_frame, text="Pendentes: -", font=("Segoe UI", 10))
        self.lbl_stats_pending.grid(row=0, column=1, padx=12, pady=5, sticky=self.tk.W)

        self.lbl_stats_processed = self.ttk.Label(self.stats_frame, text="Processadas: -", font=("Segoe UI", 10))
        self.lbl_stats_processed.grid(row=0, column=2, padx=12, pady=5, sticky=self.tk.W)

        self.lbl_stats_errors = self.ttk.Label(self.stats_frame, text="Erros: -", font=("Segoe UI", 10))
        self.lbl_stats_errors.grid(row=0, column=3, padx=12, pady=5, sticky=self.tk.W)

        self.lbl_stats_people = self.ttk.Label(self.stats_frame, text="Pessoas: -", font=("Segoe UI", 10, "bold"))
        self.lbl_stats_people.grid(row=0, column=4, padx=12, pady=5, sticky=self.tk.W)

        self.lbl_stats_duplicates = self.ttk.Label(self.stats_frame, text="Duplicatas: -", font=("Segoe UI", 10))
        self.lbl_stats_duplicates.grid(row=0, column=5, padx=12, pady=5, sticky=self.tk.W)

        self.lbl_stats_photos = self.ttk.Label(self.stats_frame, text="Fotos Reais: -", font=("Segoe UI", 10, "bold"), foreground="#007722")
        self.lbl_stats_photos.grid(row=0, column=6, padx=12, pady=5, sticky=self.tk.W)

        # Notebook / Abas
        self.notebook = self.ttk.Notebook(main_frame)
        self.notebook.pack(fill=self.tk.BOTH, expand=True)

        self._build_scan_tab()
        self._build_describe_tab()
        self._build_people_tab()
        self._build_duplicates_tab()
        self._build_classification_tab()
        self._build_search_tab()

        # -------------------------------------------------------------
        # BARRA DE PROGRESSO & CONTROLE DE INTERRUPÇÃO INFERIOR
        # -------------------------------------------------------------
        self.progress_frame = self.ttk.LabelFrame(main_frame, text=" 📊 Progresso & Controle de Operações ", padding="10")
        self.progress_frame.pack(fill=self.tk.X, pady=(10, 0))

        prog_top_row = self.ttk.Frame(self.progress_frame)
        prog_top_row.pack(fill=self.tk.X, pady=(0, 4))

        self.lbl_progress_status = self.ttk.Label(
            prog_top_row,
            text="Sistema pronto para operações.",
            font=("Segoe UI", 9)
        )
        self.lbl_progress_status.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True)

        self.lbl_progress_pct = self.ttk.Label(
            prog_top_row,
            text="",
            font=("Segoe UI", 9, "bold"),
            foreground="#0055aa"
        )
        self.lbl_progress_pct.pack(side=self.tk.RIGHT)

        prog_bar_row = self.ttk.Frame(self.progress_frame)
        prog_bar_row.pack(fill=self.tk.X)

        self.progressbar = self.ttk.Progressbar(prog_bar_row, orient=self.tk.HORIZONTAL, mode="determinate")
        self.progressbar.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=(0, 12))

        self.btn_stop_op = self.ttk.Button(
            prog_bar_row,
            text="⏹️ Interromper Operação",
            command=self.stop_current_operation,
            state=self.tk.DISABLED
        )
        self.btn_stop_op.pack(side=self.tk.RIGHT)

    def _schedule_dashboard_queue_check(self):
        try:
            while not self.events_queue.empty():
                evt = self.events_queue.get_nowait()
                self._handle_dashboard_event(evt)
        except Exception as e:
            logger.debug(f"Erro na fila de eventos do dashboard: {e}")
        self.root.after(50, self._schedule_dashboard_queue_check)

    def _handle_dashboard_event(self, evt: Dict[str, Any]):
        ev_type = evt.get("event")

        if ev_type == "scan_start":
            self.lbl_progress_status.configure(text=f"Escaneando diretório: {evt.get('root_path', '')}...")
            self.lbl_progress_pct.configure(text="Em andamento...")

        elif ev_type == "scan_progress":
            scanned = evt.get("scanned_files", 0)
            found = evt.get("images_found", 0)
            inserted = evt.get("new_inserted", 0)
            cur_file = evt.get("current_file", "")
            self.lbl_progress_status.configure(
                text=f"Varredura: {scanned} arquivos analisados | {found} imagens encontradas ({inserted} novas inseridas)... [{cur_file[:30]}]"
            )

        elif ev_type == "scan_completed":
            self.is_busy = False
            self.progressbar.stop()
            self.progressbar.configure(mode="determinate", maximum=100, value=100)
            self.btn_stop_op.configure(state=self.tk.DISABLED)
            self.refresh_stats()

            interrupted = evt.get("interrupted", False)
            found = evt.get("images_found", 0)
            inserted = evt.get("new_inserted", 0)
            elapsed = evt.get("elapsed_seconds", 0)

            if interrupted:
                status_msg = f"Varredura interrompida! {found} imagens localizadas, {inserted} novas cadastradas ({elapsed}s)."
                self.lbl_progress_status.configure(text=status_msg)
                self.lbl_progress_pct.configure(text="[Interrompido]", foreground="#cc3300")
            else:
                status_msg = f"Varredura concluída com sucesso! {found} imagens encontradas, {inserted} novas inseridas ({elapsed}s)."
                self.lbl_progress_status.configure(text=status_msg)
                self.lbl_progress_pct.configure(text="100%", foreground="#007722")
                self.messagebox.showinfo("Varredura Concluída", f"Imagens encontradas: {found}\nNovas inseridas: {inserted}\nTempo: {elapsed}s")

        elif ev_type == "hash_start":
            tot = max(1, evt.get("total", 1))
            self.progressbar.configure(mode="determinate", maximum=tot, value=0)
            self.lbl_progress_status.configure(text=f"Calculando hashes para {tot} imagem(ns)...")
            self.lbl_progress_pct.configure(text="0%", foreground="#0055aa")

        elif ev_type == "hash_progress":
            c = evt.get("current", 0)
            t = max(1, evt.get("total", 1))
            pct = int((c / t) * 100)
            self.progressbar.configure(value=c)
            self.lbl_progress_status.configure(text=f"Calculando hashes: {c} de {t} imagens...")
            self.lbl_progress_pct.configure(text=f"{pct}%", foreground="#0055aa")

        elif ev_type == "hash_completed":
            self.is_busy = False
            self.btn_stop_op.configure(state=self.tk.DISABLED)
            self.refresh_stats()
            interrupted = evt.get("interrupted", False)
            p = evt.get("processed", 0)
            if interrupted:
                self.lbl_progress_status.configure(text=f"Cálculo de hashes interrompido ({p} processadas).")
                self.lbl_progress_pct.configure(text="[Interrompido]", foreground="#cc3300")
            else:
                self.progressbar.configure(value=self.progressbar["maximum"])
                self.lbl_progress_status.configure(text=f"Hashes atualizados! {p} imagens processadas.")
                self.lbl_progress_pct.configure(text="100%", foreground="#007722")
                self.messagebox.showinfo("Sucesso", f"Hashes atualizados para {p} imagens!")

        elif ev_type == "export_start":
            tot = max(1, evt.get("total_unique", 1))
            self.progressbar.configure(mode="determinate", maximum=tot, value=0)
            self.lbl_progress_status.configure(text=f"Iniciando exportação de {tot} imagens únicas...")
            self.lbl_progress_pct.configure(text="0%", foreground="#0055aa")

        elif ev_type == "export_progress":
            c = evt.get("current", 0)
            t = max(1, evt.get("total", 1))
            pct = int((c / t) * 100)
            self.progressbar.configure(value=c)
            f_name = evt.get("file_name", "")
            self.lbl_progress_status.configure(text=f"Exportando: {c}/{t} ({f_name[:30]})...")
            self.lbl_progress_pct.configure(text=f"{pct}%", foreground="#0055aa")

        elif ev_type == "export_completed":
            self.is_busy = False
            self.btn_stop_op.configure(state=self.tk.DISABLED)
            interrupted = evt.get("interrupted", False)
            exp = evt.get("exported_count", 0)
            avoided = evt.get("duplicates_avoided", 0)
            dest = evt.get("destination_dir", "")

            if interrupted:
                self.lbl_progress_status.configure(text=f"Exportação interrompida pelo usuário ({exp} copiadas).")
                self.lbl_progress_pct.configure(text="[Interrompido]", foreground="#cc3300")
            else:
                self.progressbar.configure(value=self.progressbar["maximum"])
                self.lbl_progress_status.configure(text=f"Exportação concluída! {exp} imagens únicas salvas.")
                self.lbl_progress_pct.configure(text="100%", foreground="#007722")
                self.messagebox.showinfo(
                    "Exportação Concluída",
                    f"Foram exportadas {exp} imagens únicas para:\n{dest}\n\n"
                    f"Economia de {avoided} fotos repetidas."
                )

        elif ev_type == "classification_start":
            tot = max(1, evt.get("total", 1))
            self.progressbar.configure(mode="determinate", maximum=tot, value=0)
            self.lbl_progress_status.configure(text=f"Classificando com IA ({evt.get('model', '')}): 0/{tot}...")
            self.lbl_progress_pct.configure(text="0%", foreground="#0055aa")

        elif ev_type == "image_classification_complete":
            c = evt.get("current", 0)
            t = max(1, evt.get("total", 1))
            pct = evt.get("percent", 0)
            self.progressbar.configure(value=c)
            f_name = evt.get("file_name", "")
            lbl = evt.get("label", "")
            self.lbl_progress_status.configure(text=f"Classificação [{c}/{t}]: {f_name[:25]} -> {lbl}")
            self.lbl_progress_pct.configure(text=f"{pct}%", foreground="#0055aa")

        elif ev_type == "classification_finished":
            self.is_busy = False
            self.btn_stop_op.configure(state=self.tk.DISABLED)
            self.refresh_stats()
            summ = evt.get("summary", {})
            interrupted = summ.get("interrupted", False)
            class_count = summ.get("classified_count", 0)
            photos = summ.get("photos_found", 0)
            screens = summ.get("screenshots_found", 0)
            icons = summ.get("icons_found", 0)

            if interrupted:
                self.lbl_progress_status.configure(text=f"Classificação interrompida pelo usuário ({class_count} analisadas).")
                self.lbl_progress_pct.configure(text="[Interrompido]", foreground="#cc3300")
            else:
                self.progressbar.configure(value=self.progressbar["maximum"])
                self.lbl_progress_status.configure(text=f"Classificação concluída! {class_count} imagens categorizadas.")
                self.lbl_progress_pct.configure(text="100%", foreground="#007722")
                self.messagebox.showinfo(
                    "Classificação Concluída",
                    f"Processamento com IA finalizado!\n\n"
                    f"• 📷 Fotos Reais detectadas: {photos}\n"
                    f"• 📱 Prints de Tela: {screens}\n"
                    f"• 🎨 Ícones / Assets: {icons}\n"
                    f"• 📄 Outros: {summ.get('others_found', 0)}"
                )

        elif ev_type == "export_photos_start":
            tot = max(1, evt.get("total_photos", 1))
            self.progressbar.configure(mode="determinate", maximum=tot, value=0)
            self.lbl_progress_status.configure(text=f"Iniciando exportação de {tot} fotos reais...")
            self.lbl_progress_pct.configure(text="0%", foreground="#0055aa")

        elif ev_type == "photo_exported":
            c = evt.get("current", 0)
            t = max(1, evt.get("total", 1))
            pct = evt.get("percent", 0)
            self.progressbar.configure(value=c)
            dest_f = os.path.basename(evt.get("dest_file", ""))
            self.lbl_progress_status.configure(text=f"Exportando foto real [{c}/{t}]: {dest_f}...")
            self.lbl_progress_pct.configure(text=f"{pct}%", foreground="#0055aa")

        elif ev_type == "export_photos_complete":
            self.is_busy = False
            self.btn_stop_op.configure(state=self.tk.DISABLED)
            summ = evt.get("summary", {})
            interrupted = summ.get("interrupted", False)
            exp = summ.get("exported_count", 0)
            dest = summ.get("destination_dir", "")

            if interrupted:
                self.lbl_progress_status.configure(text=f"Exportação de fotos interrompida ({exp} copiadas).")
                self.lbl_progress_pct.configure(text="[Interrompido]", foreground="#cc3300")
            else:
                self.progressbar.configure(value=self.progressbar["maximum"])
                self.lbl_progress_status.configure(text=f"Exportação concluída! {exp} fotos reais salvas.")
                self.lbl_progress_pct.configure(text="100%", foreground="#007722")
                self.messagebox.showinfo(
                    "Exportação de Fotos Concluída",
                    f"Foram exportadas com sucesso {exp} fotos reais para:\n{dest}\n\n"
                    f"Volume total: {summ.get('total_mb', 0)} MB."
                )

        elif ev_type == "op_error":
            self.is_busy = False
            self.progressbar.stop()
            self.btn_stop_op.configure(state=self.tk.DISABLED)
            msg = evt.get("message", "Erro desconhecido")
            self.lbl_progress_status.configure(text=f"Erro: {msg}")
            self.lbl_progress_pct.configure(text="[Erro]", foreground="#cc0000")
            self.messagebox.showerror("Erro na Operação", msg)

    def stop_current_operation(self):
        """Interrompe a operação em segundo plano em execução."""
        if self.is_busy:
            self.stop_event.set()
            self.lbl_progress_status.configure(text="⏹️ Solicitando interrupção da operação...")
            self.btn_stop_op.configure(state=self.tk.DISABLED)

    def _on_close(self):
        self.stop_event.set()
        self.root.destroy()

    def _build_scan_tab(self):
        tab = self.ttk.Frame(self.notebook, padding="15")
        self.notebook.add(tab, text=" 📂 Varredura (Scan) ")

        lbl_desc = self.ttk.Label(
            tab,
            text="Escanear diretórios ou unidades inteiras de disco e indexar caminhos de imagens:",
            font=("Segoe UI", 10)
        )
        lbl_desc.pack(anchor=self.tk.W, pady=(0, 10))

        # Seletor de diretório
        dir_frame = self.ttk.Frame(tab)
        dir_frame.pack(fill=self.tk.X, pady=5)

        self.ttk.Label(dir_frame, text="Diretório / Disco:").pack(side=self.tk.LEFT, padx=(0, 10))
        self.scan_path_var = self.tk.StringVar(value=os.path.abspath("."))
        self.entry_scan_path = self.ttk.Entry(dir_frame, textvariable=self.scan_path_var, width=50)
        self.entry_scan_path.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=(0, 10))

        btn_browse = self.ttk.Button(dir_frame, text="Procurar...", command=self._browse_directory)
        btn_browse.pack(side=self.tk.LEFT)

        # Configurações adicionais
        opt_frame = self.ttk.Frame(tab)
        opt_frame.pack(fill=self.tk.X, pady=10)

        self.ttk.Label(opt_frame, text="Tamanho do lote:").pack(side=self.tk.LEFT, padx=(0, 5))
        self.scan_batch_var = self.tk.StringVar(value="1000")
        self.ttk.Entry(opt_frame, textvariable=self.scan_batch_var, width=8).pack(side=self.tk.LEFT, padx=(0, 20))

        btn_run_scan = self.ttk.Button(
            tab,
            text="🚀 Iniciar Varredura de Imagens",
            command=self._execute_scan
        )
        btn_run_scan.pack(pady=15, ipady=5)

    def _build_describe_tab(self):
        tab = self.ttk.Frame(self.notebook, padding="15")
        self.notebook.add(tab, text=" 🤖 Avaliação IA (Ollama) ")

        lbl_desc = self.ttk.Label(
            tab,
            text="Analisar imagens pendentes com modelos de visão e identificar pessoas:",
            font=("Segoe UI", 10)
        )
        lbl_desc.pack(anchor=self.tk.W, pady=(0, 10))

        # Modelo Ollama
        model_frame = self.ttk.Frame(tab)
        model_frame.pack(fill=self.tk.X, pady=5)

        self.ttk.Label(model_frame, text="Modelo Ollama:").pack(side=self.tk.LEFT, padx=(0, 10))
        self.desc_model_var = self.tk.StringVar(value=DEFAULT_MODEL)
        self.combo_models = self.ttk.Combobox(model_frame, textvariable=self.desc_model_var, width=30)
        self.combo_models.pack(side=self.tk.LEFT, padx=(0, 10))

        btn_refresh_models = self.ttk.Button(model_frame, text="Detectar Modelos", command=self._refresh_ollama_models)
        btn_refresh_models.pack(side=self.tk.LEFT)

        # Limite e Erros
        limit_frame = self.ttk.Frame(tab)
        limit_frame.pack(fill=self.tk.X, pady=5)

        self.ttk.Label(limit_frame, text="Limite de imagens:").pack(side=self.tk.LEFT, padx=(0, 10))
        self.desc_limit_var = self.tk.StringVar(value="")
        self.ttk.Entry(limit_frame, textvariable=self.desc_limit_var, width=8).pack(side=self.tk.LEFT, padx=(0, 20))
        self.ttk.Label(limit_frame, text="(Vazio para processar todas as pendentes)", font=("Segoe UI", 8, "italic")).pack(side=self.tk.LEFT)

        self.desc_retry_var = self.tk.BooleanVar(value=False)
        self.ttk.Checkbutton(tab, text="Retentar imagens marcadas com erro anterior", variable=self.desc_retry_var).pack(anchor=self.tk.W, pady=5)

        self.desc_ai_var = self.tk.BooleanVar(value=True)
        self.ttk.Checkbutton(tab, text="Descrever cenário e ambiente com IA (Ollama)", variable=self.desc_ai_var).pack(anchor=self.tk.W, pady=5)

        btn_run_desc = self.ttk.Button(
            tab,
            text="⚡ Iniciar Processamento com IA (Avaliador Visual)",
            command=self._execute_describe
        )
        btn_run_desc.pack(pady=15, ipady=5)

    def _build_people_tab(self):
        tab = self.ttk.Frame(self.notebook, padding="15")
        self.notebook.add(tab, text=" 👤 Nomeação & Rostos ")

        lbl_desc = self.ttk.Label(
            tab,
            text="Visualizar o recorte facial de cada pessoa catalogada para renomear ou unificar identidades,\nou abrir o visualizador de fotos para inspecionar e nomear pessoas diretamente nas imagens:",
            font=("Segoe UI", 10)
        )
        lbl_desc.pack(anchor=self.tk.W, pady=(0, 15))

        btn_open_viewer = self.ttk.Button(
            tab,
            text="🖼️ Abrir Visualizador Completo de Fotos (Ver e Nomear Rostos nas Fotos)",
            command=self._open_photo_viewer
        )
        btn_open_viewer.pack(fill=self.tk.X, pady=(0, 10), ipady=8)

        btn_open_namer = self.ttk.Button(
            tab,
            text="👤 Abrir Janela de Nomeação e Catálogo de Rostos (Recortes)",
            command=self._open_face_namer
        )
        btn_open_namer.pack(fill=self.tk.X, pady=(0, 10), ipady=6)

    def _build_duplicates_tab(self):
        tab = self.ttk.Frame(self.notebook, padding="15")
        self.notebook.add(tab, text=" 👯 Duplicatas & Exportação ")

        lbl_desc = self.ttk.Label(
            tab,
            text="Identificar fotos repetidas no banco, ciclar entre as duplicatas para decidir qual manter\ne exportar um diretório contendo exclusivamente as imagens únicas:",
            font=("Segoe UI", 10)
        )
        lbl_desc.pack(anchor=self.tk.W, pady=(0, 15))

        actions_frame = self.ttk.Frame(tab)
        actions_frame.pack(fill=self.tk.X, pady=10)

        btn_open_dup_gui = self.ttk.Button(
            actions_frame,
            text="🔍 Abrir Comparador de Duplicatas (Visualizador de Quadros)",
            command=self._open_duplicates_gui
        )
        btn_open_dup_gui.pack(fill=self.tk.X, pady=6, ipady=6)

        btn_export_unique = self.ttk.Button(
            actions_frame,
            text="📁 Exportar Imagens Únicas para Pasta (Seletor Windows Explorer)",
            command=self._export_unique_images
        )
        btn_export_unique.pack(fill=self.tk.X, pady=6, ipady=6)

        btn_recalc_hashes = self.ttk.Button(
            actions_frame,
            text="⚡ Recalcular Hashes das Imagens",
            command=self._recalculate_hashes
        )
        btn_recalc_hashes.pack(fill=self.tk.X, pady=6, ipady=4)

    def _build_classification_tab(self):
        tab = self.ttk.Frame(self.notebook, padding="15")
        self.notebook.add(tab, text=" 🖼️ Classificação & Fotos Reais ")

        # Seção 1: Classificação com IA
        sec1 = self.ttk.LabelFrame(tab, text=" 🤖 Classificação de Tipos com IA (Fotos vs Prints vs Ícones) ", padding="10")
        sec1.pack(fill=self.tk.X, pady=(0, 15))

        lbl_info = self.ttk.Label(
            sec1,
            text="Classifique automaticamente todo o acervo de imagens em:\n"
                 "• 📷 Fotos Reais (pessoas, viagens, paisagens, eventos, objetos no mundo real)\n"
                 "• 📱 Prints de Tela (conversas de WhatsApp, janelas de software, capturas de celular/PC)\n"
                 "• 🎨 Ícones / Assets (botões de interface, gráficos de programas, logotipos)\n"
                 "• 📄 Outros / Documentos (texturas, textos e documentos escaneados)",
            font=("Segoe UI", 9)
        )
        lbl_info.pack(anchor=self.tk.W, pady=(0, 10))

        row_cls = self.ttk.Frame(sec1)
        row_cls.pack(fill=self.tk.X, pady=4)

        self.ttk.Label(row_cls, text="Motor de IA:").pack(side=self.tk.LEFT, padx=(0, 5))
        self.cls_engine_var = self.tk.StringVar(value="⚡ IA Local Integrada (Ultrarrápida, ~1ms)")
        self.combo_cls_engine = self.ttk.Combobox(
            row_cls,
            textvariable=self.cls_engine_var,
            values=[
                "⚡ IA Local Integrada (Ultrarrápida, ~1ms)",
                "🦙 Ollama VLM (LLaVA / Visão Generativa)"
            ],
            width=36,
            state="readonly"
        )
        self.combo_cls_engine.pack(side=self.tk.LEFT, padx=(0, 12))

        self.ttk.Label(row_cls, text="Limite:").pack(side=self.tk.LEFT, padx=(0, 5))
        self.cls_limit_var = self.tk.StringVar(value="")
        self.ttk.Entry(row_cls, textvariable=self.cls_limit_var, width=6).pack(side=self.tk.LEFT, padx=(0, 12))

        self.cls_force_var = self.tk.BooleanVar(value=False)
        self.ttk.Checkbutton(row_cls, text="Forçar reclassificação", variable=self.cls_force_var).pack(side=self.tk.LEFT, padx=(0, 10))

        btn_run_cls = self.ttk.Button(
            row_cls,
            text="🚀 Iniciar Classificação IA",
            command=self._execute_classification
        )
        btn_run_cls.pack(side=self.tk.RIGHT)

        # Seção 2: Exportação de Fotos Reais
        sec2 = self.ttk.LabelFrame(tab, text=" 📸 Exportação Exclusiva de Fotos Reais ", padding="10")
        sec2.pack(fill=self.tk.X, pady=(0, 15))

        lbl_exp_info = self.ttk.Label(
            sec2,
            text="Exporte exclusivamente as imagens classificadas como 'Foto Real' para uma pasta dedicada,\n"
                 "descartando prints de tela, ícones e arquivos repetidos:",
            font=("Segoe UI", 9)
        )
        lbl_exp_info.pack(anchor=self.tk.W, pady=(0, 8))

        row_dest = self.ttk.Frame(sec2)
        row_dest.pack(fill=self.tk.X, pady=4)

        self.ttk.Label(row_dest, text="Pasta Destino:").pack(side=self.tk.LEFT, padx=(0, 5))
        self.export_photos_dest_var = self.tk.StringVar(value="")
        self.entry_exp_dest = self.ttk.Entry(row_dest, textvariable=self.export_photos_dest_var, width=40)
        self.entry_exp_dest.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=(0, 5))

        btn_browse_exp = self.ttk.Button(row_dest, text="📁 Escolher Pasta...", command=self._browse_export_photos_dest)
        btn_browse_exp.pack(side=self.tk.LEFT)

        row_exp_opts = self.ttk.Frame(sec2)
        row_exp_opts.pack(fill=self.tk.X, pady=8)

        self.exp_photos_unique_var = self.tk.BooleanVar(value=True)
        self.ttk.Checkbutton(
            row_exp_opts,
            text="Apenas fotos únicas (eliminar duplicatas confirmadas)",
            variable=self.exp_photos_unique_var
        ).pack(side=self.tk.LEFT)

        btn_run_exp_photos = self.ttk.Button(
            row_exp_opts,
            text="💾 Exportar Apenas Fotos Reais",
            command=self._execute_export_photos
        )
        btn_run_exp_photos.pack(side=self.tk.RIGHT)

        # Seção 3: Resumo Atual dos Tipos
        sec3 = self.ttk.LabelFrame(tab, text=" 📊 Detalhamento de Tipos Cadastrados ", padding="10")
        sec3.pack(fill=self.tk.BOTH, expand=True)

        self.lbl_types_summary = self.ttk.Label(
            sec3,
            text="Carregando estatísticas...",
            font=("Segoe UI", 9)
        )
        self.lbl_types_summary.pack(anchor=self.tk.W, pady=5)

    def _browse_export_photos_dest(self):
        chosen = select_directory_via_explorer(
            title="Selecione a pasta onde salvar exclusivamente as Fotos Reais"
        )
        if chosen:
            self.export_photos_dest_var.set(os.path.abspath(chosen))

    def _execute_classification(self):
        if self.is_busy:
            self.messagebox.showwarning("Ocupado", "Já existe uma operação em andamento. Aguarde ou clique em Interromper.")
            return

        engine_choice = self.cls_engine_var.get().strip()
        is_local = "Local" in engine_choice
        engine = "fast_local" if is_local else "ollama"
        model = "local-fast-v1" if is_local else DEFAULT_MODEL
        limit_str = self.cls_limit_var.get().strip()
        limit = int(limit_str) if limit_str.isdigit() else None
        force = self.cls_force_var.get()

        self.stop_event.clear()
        self.is_busy = True
        self.btn_stop_op.configure(state=self.tk.NORMAL)
        self.progressbar.configure(mode="indeterminate")
        self.progressbar.start(10)
        desc = "IA Local Integrada (~1ms/img)" if is_local else f"Ollama ({model})"
        self.lbl_progress_status.configure(text=f"Iniciando classificação de imagens com {desc}...")
        self.lbl_progress_pct.configure(text="")

        def worker():
            try:
                classify_images_batch(
                    db_path=self.db_path,
                    engine=engine,
                    model=model,
                    limit=limit,
                    force=force,
                    progress_callback=lambda d: self.events_queue.put(d),
                    stop_event=self.stop_event
                )
            except Exception as ex:
                self.events_queue.put({"event": "op_error", "message": f"Erro na classificação: {ex}"})

        self.current_worker = threading.Thread(target=worker, daemon=True)
        self.current_worker.start()

    def _execute_export_photos(self):
        if self.is_busy:
            self.messagebox.showwarning("Ocupado", "Já existe uma operação em andamento. Aguarde ou clique em Interromper.")
            return

        dest = self.export_photos_dest_var.get().strip()
        if not dest:
            dest = select_directory_via_explorer(
                title="Selecione a pasta onde salvar exclusivamente as Fotos Reais"
            )
            if dest:
                self.export_photos_dest_var.set(os.path.abspath(dest))
            else:
                return

        only_unique = self.exp_photos_unique_var.get()

        self.stop_event.clear()
        self.is_busy = True
        self.btn_stop_op.configure(state=self.tk.NORMAL)
        self.progressbar.configure(mode="indeterminate")
        self.progressbar.start(10)
        self.lbl_progress_status.configure(text=f"Iniciando exportação de fotos reais para: {dest}...")
        self.lbl_progress_pct.configure(text="")

        def worker():
            try:
                export_real_photos(
                    db_path=self.db_path,
                    destination_dir=dest,
                    only_unique=only_unique,
                    open_explorer_on_complete=True,
                    progress_callback=lambda d: self.events_queue.put(d),
                    stop_event=self.stop_event
                )
            except Exception as ex:
                self.events_queue.put({"event": "op_error", "message": f"Erro na exportação de fotos: {ex}"})

        self.current_worker = threading.Thread(target=worker, daemon=True)
        self.current_worker.start()

    def _build_search_tab(self):
        tab = self.ttk.Frame(self.notebook, padding="15")
        self.notebook.add(tab, text=" 🔍 Galeria & Busca ")

        search_bar = self.ttk.Frame(tab)
        search_bar.pack(fill=self.tk.X, pady=(0, 10))

        self.ttk.Label(search_bar, text="Buscar:").pack(side=self.tk.LEFT, padx=(0, 5))
        self.search_query_var = self.tk.StringVar()
        entry_q = self.ttk.Entry(search_bar, textvariable=self.search_query_var, width=35)
        entry_q.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=(0, 10))
        entry_q.bind("<Return>", lambda e: self._execute_search())

        btn_search = self.ttk.Button(search_bar, text="Buscar", command=self._execute_search)
        btn_search.pack(side=self.tk.LEFT, padx=(0, 5))

        btn_all = self.ttk.Button(search_bar, text="Ver Todas", command=self._show_all_images)
        btn_all.pack(side=self.tk.LEFT)

        # Treeview de Resultados
        columns = ("id", "filename", "status", "people", "path")
        self.tree_results = self.ttk.Treeview(tab, columns=columns, show="headings", height=10)
        self.tree_results.heading("id", text="ID")
        self.tree_results.heading("filename", text="Arquivo")
        self.tree_results.heading("status", text="Status")
        self.tree_results.heading("people", text="Pessoas")
        self.tree_results.heading("path", text="Caminho Completo")

        self.tree_results.column("id", width=45, anchor=self.tk.CENTER)
        self.tree_results.column("filename", width=140)
        self.tree_results.column("status", width=80, anchor=self.tk.CENTER)
        self.tree_results.column("people", width=140)
        self.tree_results.column("path", width=300)

        self.tree_results.pack(fill=self.tk.BOTH, expand=True, pady=5)
        self.tree_results.bind("<Double-1>", lambda e: self._open_selected_in_photo_viewer())

        btn_bar = self.ttk.Frame(tab)
        btn_bar.pack(fill=self.tk.X, pady=5)

        btn_viewer = self.ttk.Button(
            btn_bar,
            text="🖼️ Abrir no Visualizador de Fotos (Ver & Nomear)",
            command=self._open_selected_in_photo_viewer
        )
        btn_viewer.pack(side=self.tk.RIGHT, padx=(5, 0))

        btn_open = self.ttk.Button(
            btn_bar,
            text="👁️ Abrir no Sistema",
            command=self._open_selected_image
        )
        btn_open.pack(side=self.tk.RIGHT)

    def _browse_directory(self):
        chosen = self.filedialog.askdirectory(initialdir=self.scan_path_var.get())
        if chosen:
            self.scan_path_var.set(os.path.abspath(chosen))

    def _refresh_ollama_models(self):
        models = get_available_ollama_models(DEFAULT_OLLAMA_URL)
        if models:
            self.combo_models["values"] = models
            if self.desc_model_var.get() not in models:
                self.desc_model_var.set(models[0])
            self.messagebox.showinfo("Ollama", f"{len(models)} modelo(s) detectado(s) com sucesso!")
        else:
            self.messagebox.showwarning("Ollama", "Nenhum modelo detectado ou servidor inacessível.")

    def refresh_stats(self):
        stats = get_statistics(self.db_path)
        self.lbl_stats_total.config(text=f"Total: {stats['total']}")
        self.lbl_stats_pending.config(text=f"Pendentes: {stats['pending']}")
        self.lbl_stats_processed.config(text=f"Processadas: {stats['processed']}")
        self.lbl_stats_errors.config(text=f"Erros: {stats['errors']}")
        self.lbl_stats_people.config(text=f"Pessoas: {stats.get('known_people', 0)}")
        self.lbl_stats_duplicates.config(text=f"Duplicatas: {stats.get('duplicate_groups', 0)} ({stats.get('duplicate_images_total', 0)})")
        self.lbl_stats_photos.config(text=f"Fotos Reais: {stats.get('photos_count', 0)}")

        if hasattr(self, "lbl_types_summary"):
            self.lbl_types_summary.config(
                text=f"• 📷 Fotos Reais: {stats.get('photos_count', 0)} foto(s)\n"
                     f"• 📱 Prints de Tela: {stats.get('screenshots_count', 0)} print(s)\n"
                     f"• 🎨 Ícones / Assets de UI: {stats.get('icons_count', 0)} asset(s)\n"
                     f"• 📄 Outros / Documentos: {stats.get('other_types_count', 0)} arquivo(s)\n"
                     f"• ⏳ Pendentes de Classificação: {stats.get('unclassified_count', 0)} imagem(ns)"
            )

    def _open_duplicates_gui(self):
        start_duplicate_review_gui(db_path=self.db_path)
        self.refresh_stats()

    def _export_unique_images(self):
        if self.is_busy:
            self.messagebox.showwarning("Ocupado", "Já existe uma operação em andamento.")
            return

        dest = select_directory_via_explorer(
            title="Escolha a pasta de destino para as imagens únicas"
        )
        if dest:
            self.stop_event.clear()
            self.is_busy = True
            self.btn_stop_op.configure(state=self.tk.NORMAL)
            self.progressbar.configure(mode="determinate", value=0, maximum=100)

            def worker():
                try:
                    export_unique_images(
                        db_path=self.db_path,
                        destination_dir=dest,
                        open_explorer_on_complete=True,
                        progress_callback=lambda d: self.events_queue.put(d),
                        stop_event=self.stop_event
                    )
                except Exception as ex:
                    self.events_queue.put({"event": "op_error", "message": f"Erro na exportação: {ex}"})

            self.current_worker = threading.Thread(target=worker, daemon=True)
            self.current_worker.start()

    def _recalculate_hashes(self):
        if self.is_busy:
            self.messagebox.showwarning("Ocupado", "Já existe uma operação em andamento.")
            return

        if self.messagebox.askyesno("Recalcular Hashes", "Deseja recalcular os hashes de todas as imagens do banco?"):
            self.stop_event.clear()
            self.is_busy = True
            self.btn_stop_op.configure(state=self.tk.NORMAL)
            self.progressbar.configure(mode="determinate", value=0, maximum=100)

            def worker():
                try:
                    calculate_and_store_hashes(
                        db_path=self.db_path,
                        force=True,
                        progress_callback=lambda d: self.events_queue.put(d),
                        stop_event=self.stop_event
                    )
                except Exception as ex:
                    self.events_queue.put({"event": "op_error", "message": f"Erro no cálculo de hashes: {ex}"})

            self.current_worker = threading.Thread(target=worker, daemon=True)
            self.current_worker.start()

    def _execute_scan(self):
        if self.is_busy:
            self.messagebox.showwarning("Ocupado", "Já existe uma operação em andamento. Aguarde ou clique em Interromper.")
            return

        path = self.scan_path_var.get().strip()
        if not path or not os.path.exists(path):
            self.messagebox.showerror("Erro", f"Caminho não existe: {path}")
            return
        batch = int(self.scan_batch_var.get()) if self.scan_batch_var.get().isdigit() else 1000

        self.stop_event.clear()
        self.is_busy = True
        self.btn_stop_op.configure(state=self.tk.NORMAL)
        self.progressbar.configure(mode="indeterminate")
        self.progressbar.start(10)
        self.lbl_progress_status.configure(text=f"Iniciando varredura em: {path}...")
        self.lbl_progress_pct.configure(text="")

        def worker():
            try:
                scan_and_save_images(
                    root_path=path,
                    db_path=self.db_path,
                    batch_size=batch,
                    progress_callback=lambda d: self.events_queue.put(d),
                    stop_event=self.stop_event
                )
            except Exception as ex:
                self.events_queue.put({"event": "op_error", "message": f"Erro na varredura: {ex}"})

        self.current_worker = threading.Thread(target=worker, daemon=True)
        self.current_worker.start()

    def _execute_describe(self):
        self._open_live_evaluator()

    def _open_live_evaluator(self):
        evaluator = start_live_evaluator_gui(db_path=self.db_path, parent_root=self.root)
        evaluator.model_var.set(self.desc_model_var.get())
        evaluator.limit_var.set(self.desc_limit_var.get())
        evaluator.retry_errors_var.set(self.desc_retry_var.get())
        evaluator.describe_ai_var.set(self.desc_ai_var.get())

    def _open_face_namer(self):
        start_interactive_namer(db_path=self.db_path, cli=False, parent_root=self.root)
        self.refresh_stats()

    def _open_photo_viewer(self, image_id: Optional[int] = None):
        start_photo_viewer_gui(db_path=self.db_path, initial_image_id=image_id, parent_root=self.root)
        self.refresh_stats()

    def _open_selected_in_photo_viewer(self):
        selected = self.tree_results.selection()
        if not selected:
            self._open_photo_viewer()
            return
        item_vals = self.tree_results.item(selected[0], "values")
        img_id = int(item_vals[0]) if item_vals and str(item_vals[0]).isdigit() else None
        self._open_photo_viewer(image_id=img_id)

    def _execute_search(self):
        query = self.search_query_var.get().strip()
        results = search_images(self.db_path, query_text=query, limit=50)
        self._populate_tree(results)

    def _show_all_images(self):
        results = search_images(self.db_path, limit=100)
        self._populate_tree(results)

    def _populate_tree(self, items: List[Dict[str, Any]]):
        for item in self.tree_results.get_children():
            self.tree_results.delete(item)
        for r in items:
            pp = r.get("people_present") or "[]"
            try:
                pp_list = json.loads(pp)
                people_str = ", ".join(pp_list) if isinstance(pp_list, list) else str(pp)
            except Exception:
                people_str = str(pp)
            self.tree_results.insert(
                "",
                self.tk.END,
                values=(r["id"], r["file_name"], r["status"], people_str, r["file_path"])
            )

    def _open_selected_image(self):
        selected = self.tree_results.selection()
        if not selected:
            return
        item_vals = self.tree_results.item(selected[0], "values")
        if item_vals and len(item_vals) >= 5:
            file_path = item_vals[4]
            open_file_in_system_viewer(file_path)

    def start(self):
        self.root.mainloop()


def start_interactive_dashboard_gui(db_path: str = DEFAULT_DB_PATH):
    """Inicia o painel gráfico geral."""
    app = InteractiveDashboardGUI(db_path=db_path)
    app.start()


if __name__ == "__main__":
    run_interactive_cli()
