"""
ImageSorter - Sistema de indexação de imagens em disco e avaliação visual com Ollama.

Permite:
1. Escanear diretórios e discos recursivamente, salvando caminhos em SQLite.
2. Descrever o conteúdo de cada imagem utilizando modelos de visão do Ollama.
"""
import sys
import logging
import argparse
from db import DEFAULT_DB_PATH, get_statistics
from scan_images import scan_and_save_images, interactive_scan_prompt, DEFAULT_IMAGE_EXTENSIONS
from describe_images import (
    process_images,
    interactive_describe_prompt,
    reset_errors_to_pending,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_OLLAMA_URL
)
from interactive_namer import start_interactive_namer, DEFAULT_CROPS_DIR
from duplicate_finder import (
    start_duplicate_review_gui,
    run_interactive_duplicates_cli,
    export_unique_images,
    calculate_and_store_hashes
)
from interactive_menu import (
    run_interactive_cli,
    run_interactive_explorer,
    start_interactive_dashboard_gui,
    start_live_evaluator_gui
)
from photo_viewer import (
    start_photo_viewer_gui,
    run_cli_photo_viewer
)
from image_classifier import (
    classify_images_batch,
    export_real_photos,
    run_interactive_classification_cli
)

logger = logging.getLogger("ImageSorter.Main")


def setup_logging(debug: bool = False, log_file: str = None) -> None:
    """Configura o formato global de logging para a aplicação."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True
    )


def show_status(db_path: str = DEFAULT_DB_PATH):
    """Exibe o status atual do banco de dados de imagens."""
    stats = get_statistics(db_path)
    logger.info("=" * 50)
    logger.info("STATUS DO BANCO DE DADOS DE IMAGENS")
    logger.info("=" * 50)
    logger.info(f"Arquivo SQLite: {db_path}")
    logger.info(f" - Total de imagens cadastradas: {stats['total']}")
    logger.info(f" - Imagens processadas:          {stats['processed']}")
    logger.info(f" - Imagens pendentes:            {stats['pending']}")
    logger.info(f" - Imagens com erro:             {stats['errors']}")
    logger.info(f" - Pessoas únicas no catálogo:   {stats.get('known_people', 0)}")
    logger.info(f" - Imagens com hash calculado:   {stats.get('hashed_images', 0)}")
    logger.info(f" - Grupos de duplicatas:         {stats.get('duplicate_groups', 0)}")
    logger.info(f" - Fotos repetidas detectadas:   {stats.get('duplicate_images_total', 0)}")
    logger.info("--- Tipos de Imagem ---")
    logger.info(f" - 📷 Fotos Reais:               {stats.get('photos_count', 0)}")
    logger.info(f" - 📱 Prints de Tela:            {stats.get('screenshots_count', 0)}")
    logger.info(f" - 🎨 Ícones / Assets:           {stats.get('icons_count', 0)}")
    logger.info(f" - 📄 Outros / Documentos:       {stats.get('other_types_count', 0)}")
    logger.info(f" - ⏳ Não Classificadas:         {stats.get('unclassified_count', 0)}")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="ImageSorter: Indexador, avaliador de imagens com Ollama e identificador interativo de pessoas."
    )
    parser.add_argument("-i", "--interactive", action="store_true", help="Inicia o menu interativo guiado (CLI)")
    parser.add_argument("--gui", action="store_true", help="Inicia o painel gráfico integrado (GUI Tkinter)")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help=f"Caminho do banco SQLite (Padrão: {DEFAULT_DB_PATH})")
    parser.add_argument("--debug", action="store_true", help="Habilita logs detalhados em nível DEBUG")
    parser.add_argument("--log-file", default=None, help="Caminho opcional para salvar logs em arquivo texto")

    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")

    # Subcomando: interactive / menu
    parser_menu = subparsers.add_parser("interactive", aliases=["menu"], help="Iniciar o menu interativo principal no terminal")
    parser_menu.add_argument("--db", default=DEFAULT_DB_PATH, help="Caminho do banco SQLite")

    # Subcomando: gui / dashboard
    parser_gui = subparsers.add_parser("gui", aliases=["dashboard"], help="Iniciar o painel de controle gráfico (Tkinter)")
    parser_gui.add_argument("--db", default=DEFAULT_DB_PATH, help="Caminho do banco SQLite")

    # Subcomando: explore / search
    parser_explore = subparsers.add_parser("explore", aliases=["search", "gallery"], help="Explorar e buscar imagens e pessoas")
    parser_explore.add_argument("--db", default=DEFAULT_DB_PATH, help="Caminho do banco SQLite")

    # Subcomando: scan
    parser_scan = subparsers.add_parser("scan", help="Escanear disco/diretório e cadastrar caminhos no SQLite")
    parser_scan.add_argument("path", nargs="?", default=None, help="Diretório ou disco para varredura (ex: 'C:\\' ou './fotos')")
    parser_scan.add_argument("-i", "--interactive", action="store_true", help="Executar varredura de forma interativa")
    parser_scan.add_argument("--db", default=DEFAULT_DB_PATH, help="Caminho do banco SQLite")
    parser_scan.add_argument("--ext", nargs="*", help="Extensões de imagem permitidas")
    parser_scan.add_argument("--batch-size", type=int, default=1000, help="Tamanho do lote de inserção")

    # Subcomando: describe
    parser_desc = subparsers.add_parser("describe", help="Avaliar imagens pendentes com Ollama")
    parser_desc.add_argument("-i", "--interactive", action="store_true", help="Executar avaliação de forma interativa")
    parser_desc.add_argument("--gui", action="store_true", help="Abrir janela de avaliação visual interativa (3 quadros: Imagem, Rostos e Texto IA)")
    parser_desc.add_argument("--db", default=DEFAULT_DB_PATH, help="Caminho do banco SQLite")
    parser_desc.add_argument("--model", default=DEFAULT_MODEL, help=f"Modelo do Ollama (padrão: {DEFAULT_MODEL})")
    parser_desc.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt de avaliação visual")
    parser_desc.add_argument("--url", default=DEFAULT_OLLAMA_URL, help="URL da API do Ollama")
    parser_desc.add_argument("--limit", type=int, default=None, help="Limite de imagens a processar")
    parser_desc.add_argument("--timeout", type=int, default=120, help="Timeout por requisição (segundos)")
    parser_desc.add_argument("--retry-errors", action="store_true", help="Retentar imagens com erro anterior")
    parser_desc.add_argument("--no-face-detect", action="store_true", help="Desativa a detecção e recorte facial prévia em memória")
    parser_desc.add_argument(
        "--no-ai", "--skip-ai",
        action="store_true",
        dest="no_ai",
        help="Desativa a chamada à IA (Ollama) e realiza apenas a catalogação e reconhecimento facial biométrico local"
    )

    # Subcomando: live / live-eval
    parser_live = subparsers.add_parser("live", aliases=["live-eval", "eval-gui"], help="Abrir janela interativa ao vivo com imagem, rostos detectados e texto da IA")
    parser_live.add_argument("--db", default=DEFAULT_DB_PATH, help="Caminho do banco SQLite")
    parser_live.add_argument("--model", default=DEFAULT_MODEL, help=f"Modelo do Ollama (padrão: {DEFAULT_MODEL})")

    # Subcomando: status
    parser_status = subparsers.add_parser("status", help="Exibir estatísticas do banco de dados")
    parser_status.add_argument("--db", default=DEFAULT_DB_PATH, help="Caminho do banco SQLite")

    # Subcomando: duplicates / dedup
    parser_dup = subparsers.add_parser(
        "duplicates",
        aliases=["dedup", "dup"],
        help="Identificar imagens duplicadas, ciclar entre fotos repetidas e decidir qual manter"
    )
    parser_dup.add_argument("--db", default=DEFAULT_DB_PATH, help="Caminho do banco SQLite")
    parser_dup.add_argument("--cli", action="store_true", help="Executar no modo terminal (CLI) sem abrir GUI")
    parser_dup.add_argument("--method", choices=["sha256", "dhash"], default="sha256", help="Método de hash ('sha256' exato ou 'dhash' perceptual)")
    parser_dup.add_argument("--rehash", action="store_true", help="Forçar recálculo de hashes para todas as imagens")

    # Subcomando: export-unique / export
    parser_export = subparsers.add_parser(
        "export-unique",
        aliases=["export", "exportar"],
        help="Exportar apenas imagens únicas para uma pasta (seletor nativo do Windows Explorer)"
    )
    parser_export.add_argument("dest", nargs="?", default=None, help="Caminho da pasta de destino (abre o Explorer se omitido)")
    parser_export.add_argument("--db", default=DEFAULT_DB_PATH, help="Caminho do banco SQLite")
    parser_export.add_argument("--no-explorer", action="store_true", help="Não abrir a pasta no Explorer após a conclusão")

    # Subcomando: classify / classify-images
    parser_classify = subparsers.add_parser(
        "classify",
        aliases=["classify-images", "classificar"],
        help="Classificar imagens com IA em Fotos Reais, Prints de Tela e Ícones/Assets"
    )
    parser_classify.add_argument("--db", default=DEFAULT_DB_PATH, help="Caminho do banco SQLite")
    parser_classify.add_argument(
        "--engine",
        default="fast_local",
        choices=["fast_local", "fast", "local", "ollama"],
        help="Motor de IA: 'fast_local' (IA Local integrada ultrarrápida ~1ms, padrão) ou 'ollama' (VLM generativo)"
    )
    parser_classify.add_argument("--model", default=None, help=f"Modelo (padrão: 'local-fast-v1' para fast_local ou '{DEFAULT_MODEL}' para ollama)")
    parser_classify.add_argument("--url", default=DEFAULT_OLLAMA_URL, help="URL da API do Ollama (quando engine='ollama')")
    parser_classify.add_argument("--limit", type=int, default=None, help="Limite de imagens a classificar")
    parser_classify.add_argument("--force", action="store_true", help="Forçar re-classificação de imagens já categorizadas")
    parser_classify.add_argument("-i", "--interactive", action="store_true", help="Executar no modo terminal interativo")

    # Subcomando: export-photos / export-real
    parser_export_photos = subparsers.add_parser(
        "export-photos",
        aliases=["export-real", "export-real-photos", "exportar-fotos"],
        help="Exportar exclusivamente Fotos Reais para uma pasta (seletor nativo do Windows Explorer)"
    )
    parser_export_photos.add_argument("dest", nargs="?", default=None, help="Caminho da pasta de destino (abre o Explorer se omitido)")
    parser_export_photos.add_argument("--db", default=DEFAULT_DB_PATH, help="Caminho do banco SQLite")
    parser_export_photos.add_argument(
        "--all-duplicates",
        action="store_true",
        help="Exportar todas as fotos reais (inclusive duplicatas). Por padrão, apenas fotos únicas são exportadas"
    )
    parser_export_photos.add_argument("--no-explorer", action="store_true", help="Não abrir a pasta no Explorer após a conclusão")
    parser_export_photos.add_argument("-i", "--interactive", action="store_true", help="Executar no modo terminal interativo")

    # Subcomando: name-people (ou rename / identify)
    parser_name = subparsers.add_parser(
        "name-people",
        aliases=["rename", "identify"],
        help="Interface interativa para nomear e unificar pessoas vendo apenas o recorte do rosto"
    )
    parser_name.add_argument("--db", default=DEFAULT_DB_PATH, help="Caminho do banco SQLite")
    parser_name.add_argument("--cli", action="store_true", help="Executar no modo terminal/linha de comando (sem GUI)")
    parser_name.add_argument("--crops-dir", default=DEFAULT_CROPS_DIR, help=f"Diretório para salvar recortes faciais (Padrão: {DEFAULT_CROPS_DIR})")

    # Subcomando: view / viewer / photo-viewer
    parser_view = subparsers.add_parser(
        "view",
        aliases=["viewer", "photo-viewer", "foto", "fotos"],
        help="Visualizador interativo de fotos e identificação/nomeação manual de pessoas"
    )
    parser_view.add_argument("--db", default=DEFAULT_DB_PATH, help="Caminho do banco SQLite")
    parser_view.add_argument("--id", "--image-id", type=int, dest="image_id", default=None, help="ID da imagem para abrir diretamente")
    parser_view.add_argument("--person", default=None, help="Filtrar por nome de pessoa específica")
    parser_view.add_argument("--cli", action="store_true", help="Executar no modo terminal/linha de comando (sem GUI)")

    # Subcomando: pipeline (scan + describe + namer)
    parser_pipeline = subparsers.add_parser("pipeline", help="Executar fluxo completo guiado ou automatizado")
    parser_pipeline.add_argument("path", nargs="?", default=None, help="Diretório ou disco para varredura")
    parser_pipeline.add_argument("-i", "--interactive", action="store_true", help="Executar pipeline de forma interativa guiada")
    parser_pipeline.add_argument("--db", default=DEFAULT_DB_PATH, help="Caminho do banco SQLite")
    parser_pipeline.add_argument("--model", default=DEFAULT_MODEL, help="Modelo do Ollama")
    parser_pipeline.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt de avaliação visual")
    parser_pipeline.add_argument("--url", default=DEFAULT_OLLAMA_URL, help="URL da API do Ollama")
    parser_pipeline.add_argument("--limit", type=int, default=None, help="Limite de imagens a avaliar")
    parser_pipeline.add_argument("--no-face-detect", action="store_true", help="Desativa a detecção e recorte facial prévia em memória")
    parser_pipeline.add_argument(
        "--no-ai", "--skip-ai",
        action="store_true",
        dest="no_ai",
        help="Desativa a descrição da imagem com IA (Ollama) e executa apenas o reconhecimento biométrico"
    )

    args = parser.parse_args()

    setup_logging(debug=getattr(args, 'debug', False), log_file=getattr(args, 'log_file', None))

    # Se chamado diretamente sem argumentos ou com flag --gui / -i
    if getattr(args, 'gui', False) or args.command in ("gui", "dashboard"):
        start_interactive_dashboard_gui(db_path=args.db)
        return

    if not args.command or getattr(args, 'interactive', False) or args.command in ("interactive", "menu"):
        run_interactive_cli(db_path=args.db)
        return

    if args.command in ("explore", "search", "gallery"):
        run_interactive_explorer(db_path=args.db)
    elif args.command in ("live", "live-eval", "eval-gui"):
        start_live_evaluator_gui(db_path=args.db)
    elif args.command == "scan":
        if getattr(args, 'interactive', False) or not args.path:
            interactive_scan_prompt(db_path=args.db)
        else:
            scan_and_save_images(
                root_path=args.path,
                db_path=args.db,
                extensions=set(args.ext) if args.ext else None,
                batch_size=args.batch_size
            )
    elif args.command == "describe":
        if getattr(args, 'gui', False):
            start_live_evaluator_gui(db_path=args.db)
        elif getattr(args, 'interactive', False):
            interactive_describe_prompt(db_path=args.db)
        else:
            if args.retry_errors:
                reset_errors_to_pending(args.db)
            process_images(
                db_path=args.db,
                model=args.model,
                prompt=args.prompt,
                ollama_url=args.url,
                limit=args.limit,
                timeout=args.timeout,
                use_face_detection=not getattr(args, 'no_face_detect', False),
                describe_ai=not getattr(args, 'no_ai', False)
            )
    elif args.command == "status":
        show_status(args.db)
    elif args.command in ("duplicates", "dedup", "dup"):
        if getattr(args, 'rehash', False):
            calculate_and_store_hashes(db_path=args.db, method=args.method, force=True)
        if args.cli:
            run_interactive_duplicates_cli(db_path=args.db)
        else:
            start_duplicate_review_gui(db_path=args.db)
    elif args.command in ("export-unique", "export", "exportar"):
        export_unique_images(
            db_path=args.db,
            destination_dir=args.dest,
            open_explorer_on_complete=not getattr(args, 'no_explorer', False)
        )
    elif args.command in ("classify", "classify-images", "classificar"):
        if getattr(args, 'interactive', False):
            run_interactive_classification_cli(db_path=args.db)
        else:
            model_to_use = args.model or ("local-fast-v1" if args.engine in ("fast_local", "fast", "local") else DEFAULT_MODEL)
            classify_images_batch(
                db_path=args.db,
                engine=args.engine,
                model=model_to_use,
                base_url=args.url,
                limit=args.limit,
                force=args.force
            )
    elif args.command in ("export-photos", "export-real", "export-real-photos", "exportar-fotos"):
        if getattr(args, 'interactive', False):
            run_interactive_classification_cli(db_path=args.db)
        else:
            export_real_photos(
                db_path=args.db,
                destination_dir=args.dest,
                only_unique=not getattr(args, 'all_duplicates', False),
                open_explorer_on_complete=not getattr(args, 'no_explorer', False)
            )
    elif args.command in ("name-people", "rename", "identify"):
        start_interactive_namer(db_path=args.db, cli=args.cli)
    elif args.command in ("view", "viewer", "photo-viewer", "foto", "fotos"):
        if args.cli:
            run_cli_photo_viewer(db_path=args.db, filter_person=args.person)
        else:
            start_photo_viewer_gui(
                db_path=args.db,
                initial_image_id=args.image_id,
                filter_person=args.person
            )
    elif args.command == "pipeline":
        if getattr(args, 'interactive', False) or not args.path:
            print("\n" + "=" * 60)
            print(" ⚡ PIPELINE COMPLETO INTERATIVO")
            print("=" * 60)
            scan_res = interactive_scan_prompt(db_path=args.db)
            if scan_res is not None:
                interactive_describe_prompt(db_path=args.db)
                try:
                    open_namer = input("Deseja abrir agora a janela de nomeação de pessoas? [S/n]: ").strip().lower()
                    if open_namer not in ("n", "nao", "não"):
                        start_interactive_namer(db_path=args.db, cli=False)
                except (EOFError, KeyboardInterrupt):
                    pass
        else:
            logger.info("[Etapa 1/2] Iniciando varredura de diretórios e disco...")
            scan_and_save_images(root_path=args.path, db_path=args.db)
            if getattr(args, 'no_ai', False):
                logger.info("[Etapa 2/2] Iniciando catalogação facial biométrica local (sem IA)...")
            else:
                logger.info("[Etapa 2/2] Iniciando avaliação e descrição com Ollama...")
            process_images(
                db_path=args.db,
                model=args.model,
                prompt=args.prompt,
                ollama_url=args.url,
                limit=args.limit,
                use_face_detection=not getattr(args, 'no_face_detect', False),
                describe_ai=not getattr(args, 'no_ai', False)
            )


if __name__ == "__main__":
    main()
