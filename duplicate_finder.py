"""
Módulo de identificação de imagens duplicadas e exportação de imagens únicas para o ImageSorter.

Funcionalidades:
1. Cálculo de hashes criptográficos (SHA-256) e perceptuais (dHash) para imagens em lote.
2. Identificação e agrupamento de arquivos duplicados no banco SQLite.
3. Interface interativa com visualizador em quadros para ciclar entre fotos repetidas e decidir qual manter.
4. Exportação de imagens únicas para diretório escolhido via Windows Explorer ou CLI.
"""
import os
import sys
import shutil
import hashlib
import logging
import sqlite3
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable

from PIL import Image

from db import (
    DEFAULT_DB_PATH,
    init_db,
    get_connection,
    get_statistics,
    update_image_hash,
    update_image_duplicate_status,
    get_images_for_hashing,
    get_duplicate_groups,
    get_unique_images,
    delete_image_by_id
)

logger = logging.getLogger("ImageSorter.Duplicates")


# =====================================================================
# CÁLCULO DE HASHES
# =====================================================================

def compute_file_hash(file_path: str, chunk_size: int = 65536) -> Optional[str]:
    """
    Calcula o hash SHA-256 exato dos bytes do arquivo.
    Garante máxima eficiência de leitura por blocos em disco.
    """
    if not os.path.isfile(file_path):
        return None
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Erro ao calcular SHA-256 de '{file_path}': {e}")
        return None


def compute_perceptual_hash(file_path: str, hash_size: int = 8) -> Optional[str]:
    """
    Calcula o hash de diferença perceptual (dHash) da imagem usando Pillow.
    Detecta imagens visualmente idênticas mesmo com pequenas variações de compressão ou metadados.
    """
    if not os.path.isfile(file_path):
        return None
    try:
        with Image.open(file_path) as img:
            # Redimensiona para (hash_size + 1, hash_size) e converte para tons de cinza
            resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            if hasattr(resized, "get_flattened_data"):
                pixels = list(resized.get_flattened_data())
            else:
                pixels = list(resized.getdata())

            # Compara pixels adjacentes na horizontal
            diff = []
            width = hash_size + 1
            for row in range(hash_size):
                row_start = row * width
                for col in range(hash_size):
                    left = pixels[row_start + col]
                    right = pixels[row_start + col + 1]
                    diff.append(1 if left > right else 0)

            # Converte lista de bits em string hexadecimal
            decimal_val = 0
            for bit in diff:
                decimal_val = (decimal_val << 1) | bit
            hex_str = f"{decimal_val:0{hash_size * hash_size // 4}x}"
            return hex_str
    except Exception as e:
        logger.debug(f"Falha ao calcular dHash para '{file_path}': {e}")
        # Fallback para SHA-256 caso a imagem não possa ser decodificada pelo Pillow
        return compute_file_hash(file_path)


def compute_image_fingerprint(file_path: str, method: str = "sha256") -> Optional[str]:
    """
    Calcula o fingerprint/hash de uma imagem baseado no método selecionado ('sha256' ou 'dhash').
    """
    if method.lower() in ("dhash", "perceptual"):
        return compute_perceptual_hash(file_path)
    return compute_file_hash(file_path)


def calculate_and_store_hashes(
    db_path: str = DEFAULT_DB_PATH,
    method: str = "sha256",
    force: bool = False,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_event: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Percorre as imagens cadastradas no SQLite, calcula seus hashes e persiste no banco.
    """
    init_db(db_path)
    pending_images = get_images_for_hashing(db_path, force=force)
    total = len(pending_images)

    logger.info(f"Iniciando cálculo de hashes ({method.upper()}) para {total} imagem(ns)...")

    if progress_callback:
        progress_callback({"event": "hash_start", "total": total, "method": method})

    processed = 0
    errors = 0
    interrupted = False

    for idx, row in enumerate(pending_images, start=1):
        if stop_event and stop_event.is_set():
            logger.info("Cálculo de hashes interrompido pelo usuário.")
            interrupted = True
            break

        img_id = row["id"]
        f_path = row["file_path"]

        img_hash = compute_image_fingerprint(f_path, method=method)
        if img_hash:
            update_image_hash(db_path, img_id, img_hash)
            processed += 1
        else:
            errors += 1
            logger.warning(f"[ID {img_id}] Não foi possível calcular hash para: {f_path}")

        if progress_callback and (idx % 25 == 0 or idx == total):
            progress_callback({
                "event": "hash_progress",
                "current": idx,
                "total": total,
                "processed": processed,
                "errors": errors,
                "file_path": f_path
            })

    stats = get_statistics(db_path)
    status_label = "Cálculo de hashes interrompido" if interrupted else "Cálculo de hashes concluído"
    logger.info(
        f"{status_label}: {processed} processados, {errors} erros. "
        f"Duplicatas detectadas: {stats['duplicate_groups']} grupo(s) com {stats['duplicate_images_total']} fotos repetidas."
    )

    if progress_callback:
        progress_callback({
            "event": "hash_completed",
            "total": total,
            "processed": processed,
            "errors": errors,
            "duplicate_groups": stats["duplicate_groups"],
            "duplicate_images_total": stats["duplicate_images_total"],
            "interrupted": interrupted
        })

    return {
        "total": total,
        "processed": processed,
        "errors": errors,
        "duplicate_groups": stats["duplicate_groups"],
        "duplicate_images_total": stats["duplicate_images_total"],
        "interrupted": interrupted
    }


def find_duplicates(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Retorna os grupos de duplicatas identificados no banco.
    """
    return get_duplicate_groups(db_path)


def set_keeper_in_duplicate_group(db_path: str, keeper_id: int, group_images: List[Dict[str, Any]]) -> None:
    """
    Define uma imagem do grupo como principal (is_duplicate = 0) e todas as outras como duplicatas (is_duplicate = 1).
    """
    for img in group_images:
        img_id = img["id"]
        is_dup = 0 if img_id == keeper_id else 1
        update_image_duplicate_status(db_path, img_id, is_dup)
    logger.info(f"Grupo de duplicatas atualizado: imagem ID {keeper_id} marcada como principal.")


# =====================================================================
# EXPORTAÇÃO DE IMAGENS ÚNICAS COM SELETOR DO WINDOWS EXPLORER
# =====================================================================

def select_directory_via_explorer(
    initial_dir: Optional[str] = None,
    title: str = "Selecione a pasta de destino para exportar imagens únicas"
) -> Optional[str]:
    """
    Abre a caixa de diálogo nativa do Windows Explorer / Sistema Operacional
    para o usuário selecionar uma pasta de destino.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_dir = filedialog.askdirectory(
            title=title,
            initialdir=initial_dir or os.path.expanduser("~")
        )
        root.destroy()
        return selected_dir if selected_dir else None
    except Exception as e:
        logger.debug(f"Não foi possível abrir seletor gráfico de pasta ({e}). Solicitando via console.")
        try:
            val = input(f"{title}\nDigite o caminho da pasta de destino: ").strip()
            return val if val else None
        except (EOFError, KeyboardInterrupt):
            return None


def open_folder_in_explorer(folder_path: str) -> bool:
    """Abre o diretório no Windows Explorer."""
    if not os.path.isdir(folder_path):
        return False
    try:
        if sys.platform.startswith("win"):
            os.startfile(folder_path)
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", folder_path], check=False)
        else:
            subprocess.run(["xdg-open", folder_path], check=False)
        return True
    except Exception as e:
        logger.error(f"Erro ao abrir pasta no Explorer: {e}")
        return False


def export_unique_images(
    db_path: str = DEFAULT_DB_PATH,
    destination_dir: Optional[str] = None,
    copy_mode: str = "copy",
    open_explorer_on_complete: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_event: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Exporta apenas as imagens únicas cadastradas no banco para o diretório escolhido.
    Para cada grupo de duplicatas, apenas 1 foto (a principal/original) é exportada,
    evitando qualquer desperdício de espaço e redundância.

    :param db_path: Caminho do SQLite
    :param destination_dir: Diretório de destino (se None, abre o seletor do Explorer)
    :param copy_mode: Modo de exportação ('copy' para cópia segura)
    :param open_explorer_on_complete: Abre o Windows Explorer ao finalizar
    :param progress_callback: Função para feedback em tempo real
    :param stop_event: Objeto threading.Event para interromper a exportação
    :return: Dicionário com o resumo da exportação
    """
    init_db(db_path)

    # 1. Garante que os hashes foram calculados para imagens pendentes
    unhashed_count = len(get_images_for_hashing(db_path, force=False))
    if unhashed_count > 0:
        logger.info(f"Calculando hashes para {unhashed_count} imagem(ns) antes da exportação...")
        calculate_and_store_hashes(db_path=db_path, stop_event=stop_event)
        if stop_event and stop_event.is_set():
            return {"status": "cancelled", "exported_count": 0, "duplicates_avoided": 0, "interrupted": True}

    # 2. Obtém a lista de imagens únicas
    unique_images = get_unique_images(db_path)
    total_unique = len(unique_images)
    total_all = get_statistics(db_path)["total"]
    duplicates_avoided = max(0, total_all - total_unique)

    if total_unique == 0:
        logger.warning("Nenhuma imagem cadastrada no banco para exportar.")
        return {
            "status": "empty",
            "exported_count": 0,
            "duplicates_avoided": 0,
            "total_bytes": 0,
            "destination_dir": None
        }

    # 3. Solicita destino via Windows Explorer se não fornecido
    if not destination_dir:
        destination_dir = select_directory_via_explorer(
            title="Selecione a pasta onde salvar as imagens únicas"
        )
        if not destination_dir:
            logger.info("Exportação cancelada pelo usuário.")
            return {"status": "cancelled", "exported_count": 0, "duplicates_avoided": 0}

    dest_path = Path(destination_dir).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"EXPORTAÇÃO DE IMAGENS ÚNICAS")
    logger.info(f"Destino: {dest_path}")
    logger.info(f"Imagens únicas a exportar: {total_unique} (Economia: {duplicates_avoided} duplicata(s) descartada(s))")
    logger.info("=" * 60)

    if progress_callback:
        progress_callback({
            "event": "export_start",
            "total_unique": total_unique,
            "duplicates_avoided": duplicates_avoided,
            "destination_dir": str(dest_path)
        })

    exported_count = 0
    missing_count = 0
    total_bytes = 0
    used_names: Dict[str, int] = {}
    interrupted = False

    for idx, img in enumerate(unique_images, start=1):
        if stop_event and stop_event.is_set():
            logger.info("Exportação interrompida pelo usuário.")
            interrupted = True
            break

        src_file = img["file_path"]
        if not os.path.isfile(src_file):
            logger.warning(f"Arquivo fonte não encontrado no disco: {src_file}")
            missing_count += 1
            continue

        orig_name = img["file_name"] or os.path.basename(src_file)
        name_stem, name_ext = os.path.splitext(orig_name)

        # Tratamento de colisões de nome de arquivos diferentes
        if orig_name.lower() in used_names:
            used_names[orig_name.lower()] += 1
            target_name = f"{name_stem}_{used_names[orig_name.lower()]}{name_ext}"
        else:
            used_names[orig_name.lower()] = 1
            target_name = orig_name

        target_file = dest_path / target_name

        try:
            shutil.copy2(src_file, target_file)
            file_size = os.path.getsize(target_file)
            total_bytes += file_size
            exported_count += 1
        except Exception as e:
            logger.error(f"Erro ao copiar '{src_file}' para '{target_file}': {e}")
            missing_count += 1

        if progress_callback and (idx % 20 == 0 or idx == total_unique or interrupted):
            progress_callback({
                "event": "export_progress",
                "current": idx,
                "total": total_unique,
                "exported": exported_count,
                "file_name": target_name
            })

    mb_exported = total_bytes / (1024 * 1024)
    status_label = "Exportação interrompida" if interrupted else "Exportação concluída com sucesso!"
    logger.info(
        f"{status_label} {exported_count} imagem(ns) únicas copiadas ({mb_exported:.2f} MB). "
        f"{duplicates_avoided} duplicata(s) ignoradas."
    )

    if progress_callback:
        progress_callback({
            "event": "export_completed",
            "exported_count": exported_count,
            "duplicates_avoided": duplicates_avoided,
            "missing_count": missing_count,
            "total_bytes": total_bytes,
            "destination_dir": str(dest_path),
            "interrupted": interrupted
        })

    if open_explorer_on_complete and not interrupted:
        open_folder_in_explorer(str(dest_path))

    return {
        "status": "cancelled" if interrupted else "success",
        "exported_count": exported_count,
        "duplicates_avoided": duplicates_avoided,
        "missing_count": missing_count,
        "total_bytes": total_bytes,
        "destination_dir": str(dest_path),
        "interrupted": interrupted
    }


# =====================================================================
# INTERFACE GRÁFICA DE REVISÃO E CICLAGEM DE DUPLICATAS (GUI TKINTER)
# =====================================================================

class DuplicateReviewGUI:
    """
    Interface gráfica interativa para comparar visualmente imagens repetidas,
    ciclar entre os grupos e decidir qual arquivo manter como original.
    """
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.groups: List[Dict[str, Any]] = []
        self.current_group_idx = 0
        self.current_image_idx = 0
        self.tk_images: List[Any] = []

        import tkinter as tk
        from tkinter import ttk, messagebox
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox

        self.root = self.tk.Tk()
        self.root.title("ImageSorter - Comparador e Gerenciador de Duplicatas")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)

        self._setup_ui()
        self._load_data()

    def _setup_ui(self):
        # Top Header Bar
        top_frame = self.ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=self.tk.X)

        self.lbl_header = self.ttk.Label(
            top_frame,
            text="Identificação & Gerenciamento de Imagens Repetidas",
            font=("Segoe UI", 13, "bold")
        )
        self.lbl_header.pack(side=self.tk.LEFT)

        btn_rehash = self.ttk.Button(top_frame, text="⚡ Recalcular Hashes", command=self._recalculate_hashes)
        btn_rehash.pack(side=self.tk.RIGHT, padx=5)

        btn_export = self.ttk.Button(top_frame, text="📁 Exportar Imagens Únicas...", command=self._export_unique)
        btn_export.pack(side=self.tk.RIGHT, padx=5)

        # Navigation Bar (Grupos)
        nav_frame = self.ttk.Frame(self.root, padding="6", relief=self.tk.GROOVE)
        nav_frame.pack(fill=self.tk.X, padx=10, pady=5)

        self.btn_prev_group = self.ttk.Button(nav_frame, text="◀ Grupo Anterior", command=self._prev_group)
        self.btn_prev_group.pack(side=self.tk.LEFT, padx=5)

        self.lbl_group_info = self.ttk.Label(
            nav_frame,
            text="Grupo 0 de 0",
            font=("Segoe UI", 11, "bold"),
            anchor=self.tk.CENTER
        )
        self.lbl_group_info.pack(side=self.tk.LEFT, expand=True, fill=self.tk.X)

        self.btn_next_group = self.ttk.Button(nav_frame, text="Próximo Grupo ▶", command=self._next_group)
        self.btn_next_group.pack(side=self.tk.RIGHT, padx=5)

        # Main Content: Container horizontal com os quadros de fotos do grupo
        self.content_frame = self.ttk.Frame(self.root, padding="10")
        self.content_frame.pack(fill=self.tk.BOTH, expand=True, padx=10, pady=5)

        # Bottom Bar com Resumo, Barra de Progresso e Ações Globais
        bottom_frame = self.ttk.Frame(self.root, padding="10")
        bottom_frame.pack(fill=self.tk.X, side=self.tk.BOTTOM)

        self.lbl_status = self.ttk.Label(
            bottom_frame,
            text="Carregando grupos de duplicatas...",
            font=("Segoe UI", 9)
        )
        self.lbl_status.pack(side=self.tk.LEFT, padx=(0, 10))

        self.progressbar = self.ttk.Progressbar(bottom_frame, orient=self.tk.HORIZONTAL, mode="determinate")
        self.progressbar.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=(0, 10))

        self.btn_stop = self.ttk.Button(bottom_frame, text="⏹️ Interromper", command=self._stop_current_action, state=self.tk.DISABLED)
        self.btn_stop.pack(side=self.tk.LEFT, padx=(0, 10))

        btn_close = self.ttk.Button(bottom_frame, text="Fechar", command=self._on_close)
        btn_close.pack(side=self.tk.RIGHT)

        self.stop_event = None
        self.is_busy = False

    def _stop_current_action(self):
        if self.stop_event:
            self.stop_event.set()
            self.lbl_status.configure(text="Interrompendo operação...")

    def _on_close(self):
        if self.stop_event:
            self.stop_event.set()
        self.root.destroy()

    def _load_data(self):
        # Assegura que os hashes existam
        unhashed = len(get_images_for_hashing(self.db_path, force=False))
        if unhashed > 0:
            calculate_and_store_hashes(self.db_path)

        self.groups = find_duplicates(self.db_path)
        self.current_group_idx = 0
        self._render_current_group()

    def _recalculate_hashes(self):
        if self.is_busy:
            return
        if self.messagebox.askyesno("Recalcular Hashes", "Deseja recalcular os hashes de todas as imagens do banco?"):
            import threading
            self.stop_event = threading.Event()
            self.is_busy = True
            self.btn_stop.configure(state=self.tk.NORMAL)
            self.progressbar.configure(value=0, maximum=100)

            def cb(evt):
                if evt.get("event") == "hash_start":
                    self.root.after(0, lambda: self.progressbar.configure(maximum=max(1, evt.get("total", 1)), value=0))
                elif evt.get("event") == "hash_progress":
                    c = evt.get("current", 0)
                    t = evt.get("total", 1)
                    self.root.after(0, lambda: [
                        self.progressbar.configure(value=c),
                        self.lbl_status.configure(text=f"Calculando hash {c}/{t}...")
                    ])
                elif evt.get("event") == "hash_completed":
                    self.root.after(0, lambda: self.progressbar.configure(value=self.progressbar["maximum"]))

            def worker():
                try:
                    res = calculate_and_store_hashes(self.db_path, force=True, progress_callback=cb, stop_event=self.stop_event)
                    self.groups = find_duplicates(self.db_path)
                    self.current_group_idx = 0
                    msg = "Operação interrompida pelo usuário." if res.get("interrupted") else f"Hashes atualizados! {len(self.groups)} grupo(s) de duplicatas encontrados."
                    self.root.after(0, lambda: [
                        self._render_current_group(),
                        self.lbl_status.configure(text=msg),
                        self.btn_stop.configure(state=self.tk.DISABLED),
                        self.messagebox.showinfo("Resultado", msg)
                    ])
                finally:
                    self.is_busy = False

            threading.Thread(target=worker, daemon=True).start()

    def _export_unique(self):
        if self.is_busy:
            return
        dest = select_directory_via_explorer(
            title="Escolha a pasta de destino para as imagens únicas"
        )
        if dest:
            import threading
            self.stop_event = threading.Event()
            self.is_busy = True
            self.btn_stop.configure(state=self.tk.NORMAL)
            self.progressbar.configure(value=0, maximum=100)

            def cb(evt):
                if evt.get("event") == "export_start":
                    self.root.after(0, lambda: self.progressbar.configure(maximum=max(1, evt.get("total_unique", 1)), value=0))
                elif evt.get("event") == "export_progress":
                    c = evt.get("current", 0)
                    t = evt.get("total", 1)
                    self.root.after(0, lambda: [
                        self.progressbar.configure(value=c),
                        self.lbl_status.configure(text=f"Exportando imagem única {c}/{t}...")
                    ])
                elif evt.get("event") == "export_completed":
                    self.root.after(0, lambda: self.progressbar.configure(value=self.progressbar["maximum"]))

            def worker():
                try:
                    res = export_unique_images(
                        db_path=self.db_path,
                        destination_dir=dest,
                        open_explorer_on_complete=True,
                        progress_callback=cb,
                        stop_event=self.stop_event
                    )
                    if res["status"] == "success":
                        self.root.after(0, lambda: [
                            self.lbl_status.configure(text=f"Exportadas {res['exported_count']} imagens únicas."),
                            self.btn_stop.configure(state=self.tk.DISABLED),
                            self.messagebox.showinfo(
                                "Exportação Concluída",
                                f"Foram exportadas {res['exported_count']} imagens únicas para:\n{dest}\n\n"
                                f"Economia de {res['duplicates_avoided']} duplicata(s) repetidas."
                            )
                        ])
                    else:
                        self.root.after(0, lambda: [
                            self.lbl_status.configure(text="Exportação interrompida/cancelada."),
                            self.btn_stop.configure(state=self.tk.DISABLED)
                        ])
                finally:
                    self.is_busy = False

            threading.Thread(target=worker, daemon=True).start()

    def _prev_group(self):
        if self.groups and self.current_group_idx > 0:
            self.current_group_idx -= 1
            self._render_current_group()

    def _next_group(self):
        if self.groups and self.current_group_idx < len(self.groups) - 1:
            self.current_group_idx += 1
            self._render_current_group()

    def _render_current_group(self):
        # Limpa conteúdo anterior
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.tk_images.clear()

        if not self.groups:
            self.lbl_group_info.configure(text="Nenhuma imagem duplicada encontrada no banco de dados!")
            self.btn_prev_group.configure(state=self.tk.DISABLED)
            self.btn_next_group.configure(state=self.tk.DISABLED)

            empty_lbl = self.ttk.Label(
                self.content_frame,
                text="🎉 Excelente! Todas as imagens cadastradas são únicas.\nNenhuma duplicata detectada.",
                font=("Segoe UI", 12),
                justify=self.tk.CENTER
            )
            empty_lbl.pack(expand=True)
            self.lbl_status.configure(text="Status: 0 duplicatas no banco.")
            return

        total_groups = len(self.groups)
        group = self.groups[self.current_group_idx]
        images = group["images"]

        self.lbl_group_info.configure(
            text=f"Grupo {self.current_group_idx + 1} de {total_groups} — ({len(images)} imagens repetidas)"
        )
        self.btn_prev_group.configure(state=self.tk.NORMAL if self.current_group_idx > 0 else self.tk.DISABLED)
        self.btn_next_group.configure(state=self.tk.NORMAL if self.current_group_idx < total_groups - 1 else self.tk.DISABLED)

        # Cria quadros lado a lado para cada imagem do grupo
        cards_container = self.ttk.Frame(self.content_frame)
        cards_container.pack(fill=self.tk.BOTH, expand=True)

        for idx, img_data in enumerate(images, start=1):
            card = self.ttk.LabelFrame(
                cards_container,
                text=f" Foto #{idx} {'⭐ [Principal/Mantida]' if img_data.get('is_duplicate', 0) == 0 else '🗑️ [Duplicata]'} ",
                padding="8"
            )
            card.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True, padx=5)

            # Preview da imagem
            img_path = img_data["file_path"]
            lbl_preview = self.ttk.Label(card, text="Carregando imagem...", anchor=self.tk.CENTER)
            lbl_preview.pack(fill=self.tk.BOTH, expand=True, pady=5)

            pil_img = self._load_thumbnail(img_path, max_size=(360, 320))
            if pil_img:
                from PIL import ImageTk
                tk_img = ImageTk.PhotoImage(pil_img, master=lbl_preview)
                self.tk_images.append(tk_img)
                lbl_preview.configure(image=tk_img, text="")
            else:
                lbl_preview.configure(text="[Arquivo não encontrado]")

            # Metadados da foto
            meta_frame = self.ttk.Frame(card)
            meta_frame.pack(fill=self.tk.X, pady=5)

            f_size_kb = (img_data.get("file_size") or (os.path.getsize(img_path) if os.path.exists(img_path) else 0)) / 1024
            dim_str = self._get_dimensions_str(img_path)

            self.ttk.Label(meta_frame, text=f"Arquivo: {img_data['file_name']}", font=("Segoe UI", 9, "bold")).pack(anchor=self.tk.W)
            self.ttk.Label(meta_frame, text=f"Tamanho: {f_size_kb:.1f} KB | Dimensões: {dim_str}", font=("Segoe UI", 8)).pack(anchor=self.tk.W)
            
            p_present = img_data.get("people_present") or "[]"
            if p_present and p_present != "[]":
                self.ttk.Label(meta_frame, text=f"Pessoas: {p_present}", font=("Segoe UI", 8, "italic")).pack(anchor=self.tk.W)

            self.ttk.Label(meta_frame, text=f"Path: {img_path}", font=("Segoe UI", 7), foreground="#555555").pack(anchor=self.tk.W)

            # Botões de Ação do Card
            btn_box = self.ttk.Frame(card)
            btn_box.pack(fill=self.tk.X, pady=5)

            img_id = img_data["id"]
            btn_keep = self.ttk.Button(
                btn_box,
                text="⭐ Manter esta (Principal)",
                command=lambda k_id=img_id, grp=images: self._set_keeper(k_id, grp)
            )
            btn_keep.pack(side=self.tk.LEFT, fill=self.tk.X, expand=True, padx=2)

            btn_open = self.ttk.Button(
                btn_box,
                text="👁️ Abrir",
                width=8,
                command=lambda p=img_path: self._open_image(p)
            )
            btn_open.pack(side=self.tk.RIGHT, padx=2)

        self.lbl_status.configure(
            text=f"Total de {len(self.groups)} grupo(s) com fotos repetidas. Escolha qual manter como principal em cada grupo."
        )

    def _set_keeper(self, keeper_id: int, group_images: List[Dict[str, Any]]):
        set_keeper_in_duplicate_group(self.db_path, keeper_id, group_images)
        # Recarrega o grupo atual
        self.groups = find_duplicates(self.db_path)
        if self.current_group_idx < len(self.groups):
            self._render_current_group()
        else:
            self.current_group_idx = max(0, len(self.groups) - 1)
            self._render_current_group()

    def _open_image(self, file_path: str):
        if not os.path.exists(file_path):
            self.messagebox.showwarning("Aviso", f"Arquivo não encontrado no disco:\n{file_path}")
            return
        if sys.platform.startswith("win"):
            os.startfile(file_path)
        else:
            subprocess.run(["xdg-open", file_path], check=False)

    def _load_thumbnail(self, file_path: str, max_size=(360, 320)) -> Optional[Image.Image]:
        if not os.path.isfile(file_path):
            return None
        try:
            with Image.open(file_path) as img:
                img_copy = img.convert("RGB")
                img_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
                return img_copy
        except Exception:
            return None

    def _get_dimensions_str(self, file_path: str) -> str:
        if not os.path.isfile(file_path):
            return "N/D"
        try:
            with Image.open(file_path) as img:
                return f"{img.width}x{img.height}"
        except Exception:
            return "N/D"

    def start(self):
        self.root.mainloop()


def start_duplicate_review_gui(db_path: str = DEFAULT_DB_PATH):
    """Inicia a interface gráfica do comparador de duplicatas."""
    app = DuplicateReviewGUI(db_path=db_path)
    app.start()


# =====================================================================
# INTERFACE EM MODO TERMINAL (CLI)
# =====================================================================

def run_interactive_duplicates_cli(db_path: str = DEFAULT_DB_PATH):
    """Submenu interativo no terminal para gerenciar duplicatas e exportar fotos únicas."""
    while True:
        stats = get_statistics(db_path)
        print("\n" + "=" * 65)
        print(" 👯 GERENCIADOR DE IMAGENS DUPLICADAS & EXPORTAÇÃO")
        print("=" * 65)
        print(f" Total de fotos cadastradas:    {stats['total']}")
        print(f" Fotos com hash calculado:      {stats.get('hashed_images', 0)}")
        print(f" Grupos de duplicatas:          {stats.get('duplicate_groups', 0)}")
        print(f" Total de imagens repetidas:    {stats.get('duplicate_images_total', 0)}")
        print("=" * 65)
        print(" [1] Abrir Janela Gráfica Comparadora de Duplicatas (GUI) [Recomendado]")
        print(" [2] Ciclar e Revisar Grupos de Duplicatas no Terminal (CLI)")
        print(" [3] Recalcular/Atualizar Hashes de Todas as Imagens")
        print(" [4] Exportar Imagens Únicas para uma Pasta (Seletor Explorer)")
        print(" [0] Voltar ao Menu Principal")
        print("-" * 65)

        try:
            choice = input("Escolha uma opção [0-4]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "0":
            break

        elif choice == "1":
            try:
                start_duplicate_review_gui(db_path=db_path)
            except Exception as e:
                print(f"[!] Erro ao abrir interface gráfica: {e}")

        elif choice == "2":
            groups = find_duplicates(db_path)
            if not groups:
                print("\n[i] Nenhuma imagem duplicada encontrada no banco de dados!")
                continue

            print(f"\n[+] {len(groups)} grupo(s) de duplicatas encontrados.")
            for g_idx, group in enumerate(groups, start=1):
                imgs = group["images"]
                print("\n" + "-" * 60)
                print(f" Grupo {g_idx} de {len(groups)} (Hash: {group['hash'][:16]}...)")
                print("-" * 60)
                for idx, img in enumerate(imgs, start=1):
                    is_main = "⭐ [Principal]" if img.get("is_duplicate", 0) == 0 else "🗑️ [Duplicata]"
                    print(f"  [{idx}] {is_main} ID: {img['id']} | {img['file_name']}")
                    print(f"      Caminho: {img['file_path']}")
                    print(f"      Tamanho: {(img['file_size'] or 0)/1024:.1f} KB")

                print("\n Opções:")
                print("  [1-N] Escolher qual imagem manter como Principal")
                print("  [p]   Pular para o próximo grupo")
                print("  [s]   Sair da revisão")

                try:
                    c = input("Escolha a opção: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break

                if c == "s":
                    break
                elif c == "p" or not c:
                    continue
                elif c.isdigit() and 1 <= int(c) <= len(imgs):
                    chosen_idx = int(c) - 1
                    chosen_id = imgs[chosen_idx]["id"]
                    set_keeper_in_duplicate_group(db_path, chosen_id, imgs)
                    print(f"[✓] Foto '{imgs[chosen_idx]['file_name']}' definida como principal.")

        elif choice == "3":
            print("\nIniciando cálculo de hashes...")
            calculate_and_store_hashes(db_path=db_path, force=True)
            print("[✓] Hashes atualizados com sucesso!")

        elif choice == "4":
            print("\nSelecione o diretório de destino...")
            export_unique_images(db_path=db_path, open_explorer_on_complete=True)
