"""
Script para encontrar recursivamente todas as imagens em um disco ou diretório
e salvar seus caminhos completos em um banco de dados SQLite.
"""
import os
import sys
import time
import re
import logging
import argparse
from pathlib import Path
from typing import Set, List, Dict, Any, Optional

logger = logging.getLogger("ImageSorter.Scan")

# Permite execução direta ou import como módulo
try:
    from db import insert_images_batch, init_db, get_statistics, DEFAULT_DB_PATH
except ImportError:
    from .db import insert_images_batch, init_db, get_statistics, DEFAULT_DB_PATH


# Extensões padrão de imagens reconhecidas
DEFAULT_IMAGE_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
    ".tiff", ".tif", ".heic", ".heif", ".avif",
    ".raw", ".cr2", ".nef", ".arw", ".dng"
}


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configura o formato padrão de logs se ainda não configurado."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True
    )


def scan_and_save_images(
    root_path: str,
    db_path: str = DEFAULT_DB_PATH,
    extensions: Optional[Set[str]] = None,
    batch_size: int = 1000,
    verbose: bool = True,
    progress_callback: Optional[Any] = None,
    stop_event: Optional[Any] = None
) -> Dict[str, int]:
    """
    Percorre recursivamente o diretório/disco 'root_path', identifica arquivos de imagem
    e salva seus caminhos completos no banco SQLite especificado.

    :param root_path: Diretório raiz ou letra do disco (ex: 'C:\\' ou '/home/user/pictures')
    :param db_path: Caminho do arquivo SQLite
    :param extensions: Conjunto de extensões a considerar (em minúsculas, incluindo o ponto)
    :param batch_size: Quantidade de registros por lote de inserção
    :param verbose: Se True, exibe o progresso no console
    :param progress_callback: Função callback invocada periodicamente para reportar progresso
    :param stop_event: Objeto threading.Event para interromper a varredura
    :return: Dicionário contendo estatísticas da varredura
    """
    setup_logging(logging.INFO if verbose else logging.WARNING)

    valid_extensions = {ext.lower() if ext.startswith('.') else f".{ext.lower()}"
                        for ext in (extensions or DEFAULT_IMAGE_EXTENSIONS)}

    root = Path(root_path).resolve()
    if not root.exists():
        logger.error(f"O caminho especificado não existe: {root_path}")
        raise FileNotFoundError(f"O caminho especificado não existe: {root_path}")

    init_db(db_path)

    total_scanned_files = 0
    total_images_found = 0
    total_new_inserted = 0
    total_errors = 0
    batch: List[Dict[str, Any]] = []

    start_time = time.time()
    logger.info(f"Iniciando varredura em: {root}")
    logger.info(f"Extensões monitoradas ({len(valid_extensions)}): {', '.join(sorted(valid_extensions))}")
    logger.info(f"Banco de dados SQLite de destino: {Path(db_path).resolve()}")
    logger.info(f"Tamanho do lote de inserção: {batch_size}")

    if progress_callback:
        try:
            progress_callback({
                "event": "scan_start",
                "root_path": str(root),
                "batch_size": batch_size
            })
        except Exception:
            pass

    def on_walk_error(err: OSError):
        nonlocal total_errors
        total_errors += 1
        logger.warning(f"Permissão negada ou erro ao acessar pasta: {err.filename} ({err.strerror})")

    interrupted = False
    try:
        for dirpath, dirnames, filenames in os.walk(str(root), onerror=on_walk_error):
            if stop_event and stop_event.is_set():
                logger.info("Varredura interrompida pelo usuário via stop_event.")
                interrupted = True
                break

            logger.debug(f"Verificando diretório: {dirpath} ({len(filenames)} arquivos)")
            for filename in filenames:
                if stop_event and stop_event.is_set():
                    logger.info("Varredura interrompida pelo usuário via stop_event.")
                    interrupted = True
                    break

                total_scanned_files += 1
                ext = os.path.splitext(filename)[1].lower()

                if ext in valid_extensions:
                    total_images_found += 1
                    full_path = os.path.join(dirpath, filename)
                    
                    try:
                        file_size = os.path.getsize(full_path)
                    except (OSError, PermissionError) as pe:
                        logger.debug(f"Não foi possível obter o tamanho do arquivo {full_path}: {pe}")
                        file_size = None

                    logger.debug(f"Imagem encontrada [{total_images_found}]: {full_path}")

                    batch.append({
                        "file_path": os.path.abspath(full_path),
                        "file_name": filename,
                        "file_size": file_size
                    })

                    if len(batch) >= batch_size:
                        logger.info(f"Gravando lote de {len(batch)} imagens no banco de dados...")
                        inserted = insert_images_batch(db_path, batch)
                        total_new_inserted += inserted
                        batch.clear()

                        elapsed = time.time() - start_time
                        logger.info(f"Progresso: {total_images_found} imagens encontradas | "
                                    f"{total_new_inserted} novas cadastradas | "
                                    f"{total_scanned_files} arquivos analisados ({elapsed:.1f}s)")

                if progress_callback and (total_scanned_files % 50 == 0 or total_images_found % 10 == 0):
                    try:
                        progress_callback({
                            "event": "scan_progress",
                            "scanned_files": total_scanned_files,
                            "images_found": total_images_found,
                            "new_inserted": total_new_inserted,
                            "current_dir": dirpath,
                            "current_file": filename
                        })
                    except Exception:
                        pass

            if interrupted:
                break

    except KeyboardInterrupt:
        logger.warning("Varredura interrompida pelo usuário via teclado.")
        interrupted = True
    except Exception as e:
        total_errors += 1
        logger.error(f"Erro inesperado durante a varredura: {e}", exc_info=True)

    # Insere registros restantes se não foi forçado descarte
    if batch:
        logger.info(f"Gravando lote final de {len(batch)} imagens no banco de dados...")
        inserted = insert_images_batch(db_path, batch)
        total_new_inserted += inserted
        batch.clear()

    elapsed = time.time() - start_time
    stats = get_statistics(db_path)
    logger.info("=" * 60)
    status_label = "VARREDURA INTERROMPIDA" if interrupted else "RESUMO DA VARREDURA CONCLUÍDA"
    logger.info(status_label)
    logger.info(f" - Tempo total gasto: {elapsed:.2f} segundos")
    logger.info(f" - Arquivos totais verificados: {total_scanned_files}")
    logger.info(f" - Imagens encontradas: {total_images_found}")
    logger.info(f" - Novas imagens adicionadas ao banco: {total_new_inserted}")
    logger.info(f" - Total acumulado no banco: {stats['total']} (Pendentes: {stats['pending']}, Processadas: {stats['processed']}, Erros: {stats['errors']})")
    logger.info("=" * 60)

    res = {
        "scanned_files": total_scanned_files,
        "images_found": total_images_found,
        "new_inserted": total_new_inserted,
        "elapsed_seconds": int(elapsed),
        "interrupted": interrupted
    }

    if progress_callback:
        try:
            progress_callback({
                "event": "scan_completed",
                **res
            })
        except Exception:
            pass

    return res


def get_available_drives() -> List[str]:
    """Retorna as letras de unidades/discos disponíveis no Windows ou diretório raiz no Unix."""
    drives = []
    if sys.platform.startswith("win"):
        import string
        from ctypes import windll
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1:
                drives.append(f"{letter}:\\")
            bitmask >>= 1
    else:
        drives.append("/")
    return drives


def interactive_scan_prompt(db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Executa a configuração interativa guiada para varredura de imagens.
    Solicita o caminho da pasta/disco, extensões e lote, validando as entradas do usuário.
    """
    print("\n" + "=" * 60)
    print(" [*] VARREDURA INTERATIVA DE IMAGENS")
    print("=" * 60)

    # Sugestões de caminhos
    current_dir = os.path.abspath(".")
    drives = get_available_drives()
    
    print(f"Diretório atual: {current_dir}")
    if drives:
        print(f"Discos detectados: {', '.join(drives)}")

    while True:
        prompt_str = f"Digite o caminho do diretório ou disco a escanear [Padrão: {current_dir}]: "
        try:
            user_path = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nOperação cancelada pelo usuário.")
            return None

        if not user_path:
            chosen_path = current_dir
        else:
            # Remove aspas se o usuário colou com aspas
            chosen_path = user_path.strip('"').strip("'")

        if os.path.exists(chosen_path):
            break
        else:
            print(f"[!] Caminho não encontrado: '{chosen_path}'. Por favor, tente novamente.\n")

    print(f"\n[OK] Caminho selecionado: {os.path.abspath(chosen_path)}")

    # Escolha de extensões
    print("\nOpções de extensões:")
    print(" 1. Todas as extensões suportadas padrão (JPG, PNG, WEBP, GIF, RAW, HEIC, etc.) [Recomendado]")
    print(" 2. Definir extensões personalizadas (ex: jpg, png, webp)")
    try:
        ext_choice = input("Escolha uma opção [1/2, Padrão: 1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nOperação cancelada.")
        return None

    extensions = None
    if ext_choice == "2":
        try:
            custom_ext = input("Digite as extensões separadas por espaço ou vírgula (ex: jpg png webp): ").strip()
            if custom_ext:
                parts = re.split(r'[,\s]+', custom_ext)
                extensions = {p if p.startswith('.') else f".{p}" for p in parts if p}
        except (EOFError, KeyboardInterrupt):
            print("\nOperação cancelada.")
            return None

    # Tamanho do lote
    try:
        batch_input = input("Tamanho do lote de inserção [Padrão: 1000]: ").strip()
        batch_size = int(batch_input) if batch_input.isdigit() and int(batch_input) > 0 else 1000
    except (EOFError, KeyboardInterrupt):
        batch_size = 1000

    print("\nIniciando varredura...")
    return scan_and_save_images(
        root_path=chosen_path,
        db_path=db_path,
        extensions=extensions,
        batch_size=batch_size,
        verbose=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Varre recursivamente um disco ou diretório e salva os caminhos das imagens em SQLite."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Caminho do diretório ou disco para escanear (ex: 'C:\\' ou '/imagens')."
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Executa em modo interativo com perguntas guiadas no terminal"
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Caminho do arquivo SQLite (Padrão: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        help="Lista de extensões personalizadas para buscar (ex: --ext jpg png webp)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Tamanho do lote de inserção no banco (Padrão: 1000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Habilita logs detalhados em nível DEBUG"
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Caminho para salvar os logs em arquivo"
    )

    args = parser.parse_args()

    setup_logging(
        level=logging.DEBUG if args.debug else logging.INFO,
        log_file=args.log_file
    )

    if args.interactive or (args.path is None and len(sys.argv) == 1):
        interactive_scan_prompt(db_path=args.db)
        return

    chosen_path = args.path or "."
    extensions = set(args.ext) if args.ext else None
    scan_and_save_images(
        root_path=chosen_path,
        db_path=args.db,
        extensions=extensions,
        batch_size=args.batch_size,
        verbose=True
    )


if __name__ == "__main__":
    main()
