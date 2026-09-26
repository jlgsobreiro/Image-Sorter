"""
Módulo de nomeação e unificação interativa de pessoas para o ImageSorter.
Permite visualizar apenas o rosto recortado de cada pessoa catalogada e renomeá-la
ou unificá-la com outra pessoa existente (quando a mesma pessoa foi identificada sob outro rótulo).
"""
import os
import sys
import logging
import queue
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path
from PIL import Image

logger = logging.getLogger("ImageSorter.InteractiveNamer")

try:
    from db import (
        get_known_people,
        rename_or_merge_known_person,
        get_person_images,
        DEFAULT_DB_PATH
    )
    from face_cropper import ensure_face_crop_for_person, DEFAULT_CROPS_DIR
    from face_identity_review import find_similar_people, run_cli_similar_people_review
except ImportError:
    from .db import (
        get_known_people,
        rename_or_merge_known_person,
        get_person_images,
        DEFAULT_DB_PATH
    )
    from .face_cropper import ensure_face_crop_for_person, DEFAULT_CROPS_DIR
    from .face_identity_review import find_similar_people, run_cli_similar_people_review


class InteractiveNamerGUI:
    """Interface gráfica Tkinter para navegação, visualização de rostos e nomeação/unificação."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        crops_dir: str = DEFAULT_CROPS_DIR,
        parent_root: Optional[Any] = None
    ):
        import tkinter as tk
        from tkinter import ttk, messagebox
        from PIL import ImageTk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.ImageTk = ImageTk

        self.db_path = db_path
        self.crops_dir = crops_dir
        self.people: List[Dict[str, Any]] = []
        self.current_idx = 0
        self.photo: Optional[Any] = None

        self.is_toplevel = parent_root is not None
        if self.is_toplevel:
            self.root = self.tk.Toplevel(parent_root)
        else:
            self.root = self.tk.Tk()

        self.root.title("ImageSorter - Nomeação e Identificação de Pessoas")
        self.root.geometry("680x780")
        self.root.minsize(580, 640)

        self._load_people_catalog()
        self._build_ui()
        self._show_current_person()

    def _load_people_catalog(self):
        self.people = get_known_people(self.db_path)

    def _build_ui(self):
        # Frame Principal
        main_frame = self.ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=self.tk.BOTH, expand=True)

        # Cabeçalho
        self.header_label = self.ttk.Label(
            main_frame,
            text="Identificação e Nomeação de Pessoas",
            font=("Segoe UI", 16, "bold")
        )
        self.header_label.pack(pady=(0, 5))

        self.sub_label = self.ttk.Label(
            main_frame,
            text="Exibindo apenas o rosto para confirmar identidade ou corrigir duplicatas.",
            font=("Segoe UI", 10)
        )
        self.sub_label.pack(pady=(0, 10))

        # Indicador de Progresso (ex: "Pessoa 1 de 4")
        self.counter_label = self.ttk.Label(
            main_frame,
            text="",
            font=("Segoe UI", 11, "italic"),
            foreground="#555555"
        )
        self.counter_label.pack(pady=(0, 10))

        # Container da Imagem do Rosto
        img_container = self.ttk.LabelFrame(main_frame, text=" Rosto Recortado ", padding="10")
        img_container.pack(fill=self.tk.BOTH, expand=True, pady=5)

        self.img_label = self.tk.Label(img_container, bg="#EFEFEF")
        self.img_label.pack(fill=self.tk.BOTH, expand=True, padx=5, pady=5)

        # Informações da Pessoa
        info_frame = self.ttk.Frame(main_frame, padding="5")
        info_frame.pack(fill=self.tk.X, pady=5)

        self.lbl_current_label = self.ttk.Label(
            info_frame,
            text="Identificador Atual: ",
            font=("Segoe UI", 11, "bold")
        )
        self.lbl_current_label.pack(anchor=self.tk.W)

        self.lbl_occurrences = self.ttk.Label(
            info_frame,
            text="Fotos encontradas: 0",
            font=("Segoe UI", 10)
        )
        self.lbl_occurrences.pack(anchor=self.tk.W, pady=(2, 2))

        self.lbl_desc_title = self.ttk.Label(
            info_frame,
            text="Descrição visual fisionômica:",
            font=("Segoe UI", 9, "bold")
        )
        self.lbl_desc_title.pack(anchor=self.tk.W, pady=(5, 0))

        self.txt_description = self.tk.Text(info_frame, height=3, wrap=self.tk.WORD, font=("Segoe UI", 9))
        self.txt_description.pack(fill=self.tk.X, pady=(2, 5))

        # Ações de Nomeação / Unificação
        actions_frame = self.ttk.LabelFrame(main_frame, text=" Atribuir Nome ou Unificar ", padding="10")
        actions_frame.pack(fill=self.tk.X, pady=5)

        # Linha 1: Digitar novo nome
        row1 = self.ttk.Frame(actions_frame)
        row1.pack(fill=self.tk.X, pady=3)

        self.ttk.Label(row1, text="Novo Nome:", width=18).pack(side=self.tk.LEFT)
        self.entry_name = self.ttk.Entry(row1, font=("Segoe UI", 10))
        self.entry_name.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=(0, 5))
        self.entry_name.bind("<Return>", lambda event: self._save_name())

        self.btn_save_name = self.ttk.Button(row1, text="Salvar Nome", command=self._save_name)
        self.btn_save_name.pack(side=self.tk.RIGHT)

        # Linha 2: Unificar com pessoa já existente
        row2 = self.ttk.Frame(actions_frame)
        row2.pack(fill=self.tk.X, pady=3)

        self.ttk.Label(row2, text="Ou é a mesma pessoa que:", width=18).pack(side=self.tk.LEFT)
        self.combo_merge = self.ttk.Combobox(row2, state="readonly", font=("Segoe UI", 9))
        self.combo_merge.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=(0, 5))

        self.btn_merge = self.ttk.Button(row2, text="Unificar / Mesclar", command=self._merge_with_selected)
        self.btn_merge.pack(side=self.tk.RIGHT)

        # Linha 3: Ver fotos desta pessoa no visualizador
        row3 = self.ttk.Frame(actions_frame)
        row3.pack(fill=self.tk.X, pady=(6, 0))

        self.btn_view_photos = self.ttk.Button(
            row3,
            text="🖼️ Ver Fotos desta Pessoa no Visualizador Interativo",
            command=self._open_in_photo_viewer
        )
        self.btn_view_photos.pack(fill=self.tk.X)

        self.btn_review_similar = self.ttk.Button(
            actions_frame,
            text="🔎 Encontrar e revisar pessoas possivelmente iguais",
            command=self._open_similar_people_review
        )
        self.btn_review_similar.pack(fill=self.tk.X, pady=(6, 0))

        # Feedback e Navegação
        self.status_msg = self.ttk.Label(
            main_frame,
            text="",
            font=("Segoe UI", 9, "bold"),
            foreground="#006600"
        )
        self.status_msg.pack(pady=5)

        nav_frame = self.ttk.Frame(main_frame)
        nav_frame.pack(fill=self.tk.X, pady=(5, 0))

        self.btn_prev = self.ttk.Button(nav_frame, text="← Anterior", command=self._prev_person)
        self.btn_prev.pack(side=self.tk.LEFT)

        self.btn_next = self.ttk.Button(nav_frame, text="Próximo →", command=self._next_person)
        self.btn_next.pack(side=self.tk.RIGHT)

    def _show_current_person(self):
        if not self.people:
            self.header_label.config(text="Nenhuma Pessoa no Catálogo")
            self.sub_label.config(text="Processe imagens com o Ollama primeiro para que as pessoas sejam detectadas.")
            self.counter_label.config(text="")
            self.img_label.config(image="", text="Sem fotos de rostos cadastradas no momento.")
            self.lbl_current_label.config(text="")
            self.lbl_occurrences.config(text="")
            self.entry_name.delete(0, self.tk.END)
            self.btn_save_name.config(state=self.tk.DISABLED)
            self.btn_merge.config(state=self.tk.DISABLED)
            self.btn_prev.config(state=self.tk.DISABLED)
            self.btn_next.config(state=self.tk.DISABLED)
            return

        self.current_idx = max(0, min(self.current_idx, len(self.people) - 1))
        person = self.people[self.current_idx]
        current_label = person["person_label"]

        self.counter_label.config(text=f"Pessoa {self.current_idx + 1} de {len(self.people)}")
        self.lbl_current_label.config(text=f"Identificador Atual: {current_label}")

        # Busca quantidade de fotos
        imgs = get_person_images(self.db_path, current_label)
        self.lbl_occurrences.config(text=f"Fotos em que esta pessoa aparece no banco: {len(imgs)}")

        # Descrição
        self.txt_description.delete("1.0", self.tk.END)
        self.txt_description.insert(self.tk.END, person.get("description", ""))

        # Campo de nome
        self.entry_name.delete(0, self.tk.END)
        self.entry_name.insert(0, current_label)
        self.entry_name.focus_set()

        # Atualiza combobox de unificação (todas as outras pessoas exceto a atual)
        other_people = [p["person_label"] for p in self.people if p["person_label"] != current_label]
        self.combo_merge["values"] = other_people
        if other_people:
            self.combo_merge.current(0)
            self.btn_merge.config(state=self.tk.NORMAL)
        else:
            self.combo_merge.set("")
            self.btn_merge.config(state=self.tk.DISABLED)

        # Garante o recorte do rosto
        crop_path = ensure_face_crop_for_person(
            self.db_path,
            person_label=current_label,
            crops_dir=self.crops_dir
        )

        # Carrega e exibe a imagem
        if crop_path and os.path.exists(crop_path):
            try:
                pil_img = Image.open(crop_path)
                pil_img.thumbnail((320, 320), Image.Resampling.LANCZOS)
                self.photo = self.ImageTk.PhotoImage(pil_img, master=self.img_label)
                self.img_label.image = self.photo
                self.img_label.config(image=self.photo, text="")
            except Exception as e:
                self.img_label.image = None
                self.img_label.config(image="", text=f"Erro ao carregar imagem: {e}")
        else:
            self.img_label.image = None
            self.img_label.config(
                image="",
                text="Recorte de rosto não disponível.\nClique abaixo em 'Ver Fotos' para visualizar as imagens desta pessoa."
            )

        # Botões de navegação
        self.btn_prev.config(state=self.tk.NORMAL if self.current_idx > 0 else self.tk.DISABLED)
        self.btn_next.config(state=self.tk.NORMAL if self.current_idx < len(self.people) - 1 else self.tk.DISABLED)

    def _open_in_photo_viewer(self):
        if not self.people:
            return
        person = self.people[self.current_idx]
        current_label = person["person_label"]
        try:
            from photo_viewer import start_photo_viewer_gui
            start_photo_viewer_gui(
                db_path=self.db_path,
                filter_person=current_label,
                parent_root=self.root
            )
        except Exception as e:
            self.messagebox.showerror("Erro", f"Não foi possível abrir o visualizador de fotos: {e}")

    def _open_similar_people_review(self):
        self.similar_people_review = SimilarPeopleReviewGUI(self.db_path, self.root)

    def _save_name(self):
        if not self.people:
            return
        person = self.people[self.current_idx]
        old_label = person["person_label"]
        new_label = self.entry_name.get().strip()
        new_desc = self.txt_description.get("1.0", self.tk.END).strip()

        if not new_label:
            self.messagebox.showwarning("Aviso", "Por favor, digite um nome válido.")
            return

        if new_label == old_label and new_desc == person.get("description", ""):
            self._next_person()
            return

        res = rename_or_merge_known_person(
            self.db_path,
            old_label=old_label,
            new_label=new_label,
            new_description=new_desc
        )

        self._load_people_catalog()
        self.status_msg.config(
            text=f"✓ '{old_label}' renomeado para '{new_label}' ({res['affected_images']} fotos atualizadas)."
        )
        self._show_current_person()

    def _merge_with_selected(self):
        if not self.people:
            return
        person = self.people[self.current_idx]
        old_label = person["person_label"]
        target_label = self.combo_merge.get().strip()

        if not target_label:
            self.messagebox.showwarning("Aviso", "Selecione a pessoa com quem deseja unificar.")
            return

        confirm = self.messagebox.askyesno(
            "Confirmar Unificação",
            f"Deseja realmente confirmar que '{old_label}' é na verdade '{target_label}'?\n\n"
            f"Todas as fotos de '{old_label}' serão migradas para '{target_label}' e '{old_label}' será removido."
        )
        if not confirm:
            return

        res = rename_or_merge_known_person(
            self.db_path,
            old_label=old_label,
            new_label=target_label
        )

        self._load_people_catalog()
        self.status_msg.config(
            text=f"✓ '{old_label}' unificado com sucesso em '{target_label}' ({res['affected_images']} fotos atualizadas)!"
        )
        self._show_current_person()

    def _prev_person(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.status_msg.config(text="")
            self._show_current_person()

    def _next_person(self):
        if self.current_idx < len(self.people) - 1:
            self.current_idx += 1
            self.status_msg.config(text="")
            self._show_current_person()

    def run(self):
        self.root.mainloop()


class SimilarPeopleReviewGUI:
    """Apresenta pares faciais semelhantes e exige confirmação para unificá-los."""

    def __init__(self, db_path: str, parent_root: Any):
        import tkinter as tk
        from tkinter import ttk, messagebox
        from PIL import ImageTk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.ImageTk = ImageTk
        self.db_path = db_path
        self.candidates: List[Dict[str, Any]] = []
        self.preview_images = []
        self.results = queue.Queue()
        self.root = tk.Toplevel(parent_root)
        self.root.title("ImageSorter - Revisão de Pessoas Semelhantes")
        self.root.geometry("900x700")
        self.root.minsize(760, 580)

        self.ttk.Label(
            self.root,
            text="Possíveis pessoas iguais — confirme visualmente antes de mesclar",
            font=("Segoe UI", 13, "bold")
        ).pack(pady=(12, 4))
        self.ttk.Label(
            self.root,
            text="Similaridade cosseno é uma sugestão visual da biometria local. Nenhum cadastro é mesclado sem sua confirmação.",
            wraplength=850
        ).pack(pady=(0, 6))

        # Barra de filtros
        filter_frame = self.ttk.LabelFrame(self.root, text=" Filtros de Busca Biométrica ", padding=8)
        filter_frame.pack(fill=tk.X, padx=10, pady=(0, 6))

        self.ttk.Label(filter_frame, text="Nível de Similaridade:").pack(side=tk.LEFT, padx=(0, 5))
        self.threshold_var = tk.StringVar(value="Todos (≥ 0.35)")
        self.threshold_combo = self.ttk.Combobox(
            filter_frame,
            textvariable=self.threshold_var,
            values=["Todos (≥ 0.35)", "Média e Alta (≥ 0.45)", "Apenas Alta (≥ 0.65)"],
            state="readonly",
            width=22
        )
        self.threshold_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.threshold_combo.bind("<<ComboboxSelected>>", lambda e: self._start_search())

        self.filter_cooccur_var = tk.BooleanVar(value=False)
        self.chk_cooccur = self.ttk.Checkbutton(
            filter_frame,
            text="Ocultar pessoas que aparecem na mesma foto",
            variable=self.filter_cooccur_var,
            command=self._start_search
        )
        self.chk_cooccur.pack(side=tk.LEFT, padx=(0, 15))

        self.btn_refresh = self.ttk.Button(
            filter_frame,
            text="🔄 Atualizar Lista",
            command=self._start_search
        )
        self.btn_refresh.pack(side=tk.RIGHT)

        content = self.ttk.Frame(self.root, padding=10)
        content.pack(fill=tk.BOTH, expand=True)
        self.tree = self.ttk.Treeview(
            content,
            columns=("first", "second", "score", "confidence", "cooccur"),
            show="headings",
            height=7
        )
        self.tree.heading("first", text="Pessoa A")
        self.tree.heading("second", text="Pessoa B")
        self.tree.heading("score", text="Similaridade")
        self.tree.heading("confidence", text="Confiança")
        self.tree.heading("cooccur", text="Mesma Foto?")
        self.tree.column("first", width=190)
        self.tree.column("second", width=190)
        self.tree.column("score", width=100, anchor=tk.CENTER)
        self.tree.column("confidence", width=120, anchor=tk.CENTER)
        self.tree.column("cooccur", width=110, anchor=tk.CENTER)
        self.tree.pack(fill=tk.X)
        self.tree.bind("<<TreeviewSelect>>", self._show_selected)

        # Aviso contextual de co-ocorrência
        self.alert_frame = self.tk.Frame(content, bg="#FFF3CD", bd=1, relief=tk.SOLID)
        self.alert_label = self.tk.Label(
            self.alert_frame,
            text="⚠️ Atenção: Ambas as pessoas aparecem juntas em foto(s). Provavelmente são pessoas distintas.",
            bg="#FFF3CD",
            fg="#856404",
            font=("Segoe UI", 9, "bold")
        )
        self.alert_label.pack(pady=4, padx=8)

        previews = self.ttk.Frame(content)
        previews.pack(fill=tk.BOTH, expand=True, pady=6)
        self.preview_labels = []
        self.detail_labels = []
        for title in ("Pessoa A", "Pessoa B"):
            pane = self.ttk.LabelFrame(previews, text=title, padding=8)
            pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            image_label = tk.Label(pane, text="Selecione um par", bg="#EFEFEF", height=10)
            image_label.pack(fill=tk.BOTH, expand=True)
            detail = self.ttk.Label(pane, text="", wraplength=380, justify=tk.LEFT)
            detail.pack(fill=tk.X, pady=(5, 0))
            self.preview_labels.append(image_label)
            self.detail_labels.append(detail)

        self.status_label = self.ttk.Label(content, text="", anchor=tk.W)
        self.status_label.pack(fill=tk.X, pady=4)
        actions = self.ttk.Frame(content)
        actions.pack(fill=tk.X, pady=(4, 0))
        self.btn_keep_first = self.ttk.Button(
            actions, text="Manter Pessoa A", command=lambda: self._merge_selected(keep_first=True)
        )
        self.btn_keep_first.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.btn_keep_second = self.ttk.Button(
            actions, text="Manter Pessoa B", command=lambda: self._merge_selected(keep_first=False)
        )
        self.btn_keep_second.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.ttk.Button(actions, text="Fechar", command=self.root.destroy).pack(side=tk.RIGHT, padx=(4, 0))
        self.btn_keep_first.configure(state=tk.DISABLED)
        self.btn_keep_second.configure(state=tk.DISABLED)

        self._start_search()

    def _get_selected_min_similarity(self) -> float:
        val = self.threshold_var.get()
        if "0.65" in val:
            return 0.65
        elif "0.45" in val:
            return 0.45
        return 0.35

    def _start_search(self):
        min_sim = self._get_selected_min_similarity()
        filter_cooccur = self.filter_cooccur_var.get()
        self.status_label.configure(text="Comparando recortes faciais com biometria local; aguarde...")
        self.btn_keep_first.configure(state=self.tk.DISABLED)
        self.btn_keep_second.configure(state=self.tk.DISABLED)
        self.alert_frame.pack_forget()
        for item in self.tree.get_children():
            self.tree.delete(item)

        def worker():
            try:
                res = find_similar_people(
                    self.db_path,
                    min_similarity=min_sim,
                    filter_cooccurring=filter_cooccur
                )
                self.results.put(("done", res))
            except Exception as exc:
                self.results.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self._poll_search)

    def _poll_search(self):
        try:
            result, payload = self.results.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_search)
            return

        if result == "error":
            self.status_label.configure(text="Falha ao comparar rostos.")
            self.messagebox.showerror("Revisão de pessoas", payload, parent=self.root)
            return

        self.candidates = payload
        for index, candidate in enumerate(self.candidates):
            first = candidate["first"]["person_label"]
            second = candidate["second"]["person_label"]
            score_str = f"{candidate['score']:.3f}"
            conf_str = candidate.get("confidence_tag", "⚪ Possível")
            cooccur_str = "⚠️ Sim" if candidate.get("co_occurs") else "Não"
            self.tree.insert(
                "",
                self.tk.END,
                iid=str(index),
                values=(first, second, score_str, conf_str, cooccur_str)
            )
        if self.candidates:
            self.status_label.configure(text=f"Encontrados {len(self.candidates)} par(es) para revisão manual.")
            first_item = self.tree.get_children()[0]
            self.tree.selection_set(first_item)
            self.tree.focus(first_item)
            self._show_selected()
        else:
            self.status_label.configure(text="Nenhum par acima do limiar foi encontrado com recortes utilizáveis.")
            self._clear_previews()

    def _show_selected(self, _event=None):
        selection = self.tree.selection()
        if not selection:
            return
        candidate = self.candidates[int(selection[0])]

        # Alerta se houver co-ocorrência
        if candidate.get("co_occurs"):
            self.alert_frame.pack(fill=self.tk.X, pady=(0, 4), before=self.preview_labels[0].master.master)
        else:
            self.alert_frame.pack_forget()

        self.preview_images = []
        for index, person in enumerate((candidate["first"], candidate["second"])):
            label = person["person_label"]
            crop_path = person.get("face_crop_path")
            p_count = candidate.get(f"{'first' if index == 0 else 'second'}_photo_count", 0)
            photo = None
            if crop_path:
                try:
                    with Image.open(crop_path) as source:
                        image = source.convert("RGB")
                        image.thumbnail((360, 240), Image.Resampling.LANCZOS)
                        photo = self.ImageTk.PhotoImage(image, master=self.preview_labels[index])
                except (OSError, ValueError):
                    pass
            self.preview_labels[index].configure(
                image=photo if photo else "",
                text="Recorte facial indisponível" if photo is None else ""
            )
            self.preview_labels[index].image = photo
            self.preview_images.append(photo)
            self.detail_labels[index].configure(
                text=f"{label} ({p_count} foto(s))\n{person.get('description', '')}"
            )

        self.btn_keep_first.configure(
            text=f"Mesclar '{candidate['second']['person_label']}' em '{candidate['first']['person_label']}'",
            state=self.tk.NORMAL
        )
        self.btn_keep_second.configure(
            text=f"Mesclar '{candidate['first']['person_label']}' em '{candidate['second']['person_label']}'",
            state=self.tk.NORMAL
        )

    def _clear_previews(self):
        self.preview_images = []
        self.alert_frame.pack_forget()
        for image_label, detail_label in zip(self.preview_labels, self.detail_labels):
            image_label.configure(image="", text="Nenhum par para exibir")
            image_label.image = None
            detail_label.configure(text="")

    def _merge_selected(self, keep_first: bool):
        selection = self.tree.selection()
        if not selection:
            return
        candidate = self.candidates[int(selection[0])]
        target = candidate["first"] if keep_first else candidate["second"]
        source = candidate["second"] if keep_first else candidate["first"]

        extra_warn = ""
        if candidate.get("co_occurs"):
            extra_warn = "\n\n⚠️ ATENÇÃO: Estas pessoas aparecem juntas em foto(s). Confirme somente se tem absoluta certeza!"

        if not self.messagebox.askyesno(
            "Confirmar unificação",
            f"Confirma mesclar '{source['person_label']}' em '{target['person_label']}'?\n\n"
            f"As referências nas imagens serão atualizadas e o cadastro de '{source['person_label']}' será removido.{extra_warn}",
            parent=self.root
        ):
            return
        result = rename_or_merge_known_person(
            self.db_path,
            source["person_label"],
            target["person_label"]
        )
        self.status_label.configure(
            text=f"Pessoas unificadas; {result['affected_images']} imagem(ns) atualizada(s). Recalculando sugestões..."
        )
        self._start_search()


def run_cli_namer(db_path: str = DEFAULT_DB_PATH, crops_dir: str = DEFAULT_CROPS_DIR):
    """Modo de nomeação interativa pelo terminal (CLI)."""
    people = get_known_people(db_path)
    if not people:
        print("\nNenhuma pessoa catalogada no banco de dados. Execute a avaliação com Ollama primeiro.")
        return

    print("\n" + "=" * 65)
    print("NOMEAÇÃO E UNIFICAÇÃO INTERATIVA DE PESSOAS (CLI)")
    print("=" * 65)
    print(f"Total de pessoas catalogadas: {len(people)}")

    for idx, person in enumerate(people, start=1):
        label = person["person_label"]
        desc = person["description"]
        imgs = get_person_images(db_path, label)

        print("-" * 65)
        print(f"[{idx}/{len(people)}] Pessoa Atual: {label}")
        print(f" - Fotos em que aparece: {len(imgs)}")
        print(f" - Descrição física: {desc}")

        crop_path = ensure_face_crop_for_person(db_path, label, crops_dir=crops_dir)
        if crop_path and os.path.exists(crop_path):
            print(f" - Recorte do rosto salvo em: {crop_path}")
            # Tenta abrir o visualizador de imagens do sistema
            try:
                if sys.platform.startswith("win"):
                    os.startfile(crop_path)
                elif sys.platform.startswith("darwin"):
                    os.system(f"open '{crop_path}'")
                else:
                    os.system(f"xdg-open '{crop_path}'")
            except Exception:
                pass
        else:
            print(" - (Recorte facial não pôde ser gerado)")

        print("\nOpções:")
        print(" [1] Digitar novo nome (ex: João, Maria)")
        print(" [2] Unificar com outra pessoa já existente (corrigir duplicata)")
        print(" [r] Revisar pares de pessoas semelhantes (biometria local)")
        print(" [ENTER / 3] Manter como está e ir para a próxima")
        print(" [q] Sair")

        choice = input("\nEscolha uma opção: ").strip()
        if choice.lower() == 'q':
            print("Encerrando nomeação interativa.")
            break
        elif choice.lower() == 'r':
            run_cli_similar_people_review(db_path)
            people = get_known_people(db_path)
            continue
        elif choice == '1':
            new_name = input(f"Digite o novo nome para '{label}': ").strip()
            if new_name and new_name != label:
                res = rename_or_merge_known_person(db_path, label, new_name)
                print(f"[OK] Atualizado: '{label}' -> '{new_name}' ({res['affected_images']} fotos alteradas).")
        elif choice == '2':
            other_people = [p["person_label"] for p in get_known_people(db_path) if p["person_label"] != label]
            if not other_people:
                print("Nenhuma outra pessoa disponível para unificação.")
                continue
            print("\nPessoas disponíveis para unificar:")
            for p_idx, p_name in enumerate(other_people, start=1):
                print(f"  {p_idx}. {p_name}")
            target_idx = input("Selecione o número da pessoa correta: ").strip()
            try:
                sel_num = int(target_idx)
                if 1 <= sel_num <= len(other_people):
                    target_name = other_people[sel_num - 1]
                    res = rename_or_merge_known_person(db_path, label, target_name)
                    print(f"[OK] Unificado: '{label}' agora é '{target_name}' ({res['affected_images']} fotos atualizadas).")
            except ValueError:
                print("Opção inválida.")

    print("\nProcesso de nomeação interativa concluído!")


def start_interactive_namer(
    db_path: str = DEFAULT_DB_PATH,
    cli: bool = False,
    parent_root: Optional[Any] = None
) -> Optional[InteractiveNamerGUI]:
    """Ponto de entrada unificado para iniciar a ferramenta de nomeação interativa."""
    if cli:
        run_cli_namer(db_path)
        return None
    else:
        try:
            gui = InteractiveNamerGUI(db_path, parent_root=parent_root)
            if not gui.is_toplevel:
                gui.run()
            return gui
        except Exception as e:
            logger.warning(f"Não foi possível iniciar a interface gráfica ({e}). Iniciando modo CLI...")
            run_cli_namer(db_path)
            return None
