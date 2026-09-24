"""
Módulo de nomeação e unificação interativa de pessoas para o ImageSorter.
Permite visualizar apenas o rosto recortado de cada pessoa catalogada e renomeá-la
ou unificá-la com outra pessoa existente (quando a mesma pessoa foi identificada sob outro rótulo).
"""
import os
import sys
import logging
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
except ImportError:
    from .db import (
        get_known_people,
        rename_or_merge_known_person,
        get_person_images,
        DEFAULT_DB_PATH
    )
    from .face_cropper import ensure_face_crop_for_person, DEFAULT_CROPS_DIR


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
        print(" [ENTER / 3] Manter como está e ir para a próxima")
        print(" [q] Sair")

        choice = input("\nEscolha uma opção: ").strip()
        if choice.lower() == 'q':
            print("Encerrando nomeação interativa.")
            break
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
