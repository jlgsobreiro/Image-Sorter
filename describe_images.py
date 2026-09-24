"""
Script para avaliar imagens cadastradas no SQLite utilizando modelos de visão do Ollama
(ex: llama3.2-vision, llava, moondream, minicpm-v, bakllava) e salvar a descrição gerada no banco de dados.
"""
import os
import sys
import time
import json
import re
import base64
import logging
import urllib.request
import urllib.error
import argparse
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger("ImageSorter.Describe")

# Permite execução direta ou import como módulo
try:
    from face_recognition import FaceIdentityMatcher
    from db import (
        get_connection,
        init_db,
        get_pending_images,
        update_image_description,
        update_image_error,
        update_image_type,
        get_statistics,
        get_known_people,
        register_known_person,
        update_person_face_crop,
        DEFAULT_DB_PATH
    )
    from face_cropper import (
        ensure_face_crop_for_person,
        crop_face,
        detect_faces_local,
        extract_face_crops_in_memory,
        encode_pil_to_base64,
        DEFAULT_CROPS_DIR
    )
except ImportError:
    from .face_recognition import FaceIdentityMatcher
    from .db import (
        get_connection,
        init_db,
        get_pending_images,
        update_image_description,
        update_image_error,
        update_image_type,
        get_statistics,
        get_known_people,
        register_known_person,
        update_person_face_crop,
        DEFAULT_DB_PATH
    )
    from .face_cropper import (
        ensure_face_crop_for_person,
        crop_face,
        detect_faces_local,
        extract_face_crops_in_memory,
        encode_pil_to_base64,
        DEFAULT_CROPS_DIR
    )


DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "llava"
DEFAULT_PROMPT = (
    "Descreva detalhadamente esta imagem em português, destacando o cenário geral, ambiente, objetos e ações."
)


def build_prompt_with_catalog(base_prompt: str, known_people: List[Dict[str, Any]]) -> str:
    """
    Constrói o prompt enviado ao modelo Ollama Vision, incluindo as diretrizes de identificação
    consistente e o catálogo de pessoas já catalogadas em fotos anteriores.
    """
    prompt_lines = [base_prompt.strip(), ""]

    if known_people:
        next_id_num = len(known_people) + 1
        prompt_lines.append("--- CATÁLOGO DE PESSOAS JÁ REGISTRADAS ---")
        prompt_lines.append(
            "IMPORTANTE: Você deve manter a CONSISTÊNCIA de identidade entre as imagens. "
            "A 'Pessoa 1' deve ser SEMPRE o mesmo indivíduo em todas as imagens onde ela aparecer."
        )
        prompt_lines.append("Perfis de pessoas já cadastradas:")
        for person in known_people:
            prompt_lines.append(f"- {person['person_label']}: {person['description']}")
        prompt_lines.append("")
        prompt_lines.append(
            "INSTRUÇÕES DE IDENTIFICAÇÃO:\n"
            "1. Compare minuciosamente cada pessoa presente na imagem com os perfis do catálogo acima.\n"
            "2. Se uma pessoa na foto for a mesma de algum perfil acima, use RIGOROSAMENTE o mesmo identificador (ex: 'Pessoa 1').\n"
            f"3. Se houver alguma pessoa que NÃO corresponda a nenhum perfil já catalogado, crie o próximo identificador sequencial (iniciando em 'Pessoa {next_id_num}') e descreva detalhadamente seus traços físicos e fisionômicos duradouros (rosto, formato dos olhos/nariz, tom de pele, cabelo, idade aparente, barba/marcas distintivas permanentes).\n"
            "4. Ao final da resposta, inclua OBRIGATORIAMENTE duas seções no formato exato abaixo:\n"
            "PESSOAS_PRESENTES: [\"Pessoa 1\", \"Pessoa 2\"] (ou [] se não houver pessoas)\n"
            "NOVAS_PESSOAS: [{\"id\": \"Pessoa X\", \"descricao\": \"características físicas detalhadas...\"}] (ou [] se nenhuma nova pessoa foi introduzida)"
        )
    else:
        prompt_lines.append(
            "--- IDENTIFICAÇÃO DE PESSOAS (NOVO CATÁLOGO) ---\n"
            "1. Para cada pessoa encontrada na imagem, atribua um identificador sequencial ('Pessoa 1', 'Pessoa 2', etc.).\n"
            "2. Descreva detalhadamente os traços físicos e fisionômicos duradouros de cada pessoa (formato do rosto, cor/tipo de cabelo, tom de pele, idade aparente, traços distintivos permanentes) para que possamos identificá-la com precisão nas próximas fotos.\n"
            "3. Ao final da resposta, inclua OBRIGATORIAMENTE duas seções no formato exato abaixo:\n"
            "PESSOAS_PRESENTES: [\"Pessoa 1\", \"Pessoa 2\"] (ou [] se não houver pessoas)\n"
            "NOVAS_PESSOAS: [{\"id\": \"Pessoa 1\", \"descricao\": \"características fisionômicas duradouras...\"}] (ou [] se não houver pessoas)"
        )

    return "\n".join(prompt_lines)


def extract_new_people_from_response(
    response_text: str,
    known_labels: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Extrai as novas pessoas catalogadas a partir da resposta do modelo de visão.
    Retorna uma lista de dicionários com 'id' (ex: 'Pessoa 1'), 'descricao' e opcionalmente 'bbox'.
    """
    if not response_text:
        return []

    existing_set = set(known_labels or [])
    new_people: List[Dict[str, Any]] = []
    registered_ids = set()

    # 1. Tentativa pelo bloco NOVAS_PESSOAS: [...]
    section_match = re.search(
        r'(?:NOVAS_PESSOAS|NOVA_PESSOA|NEW_PEOPLE)\s*:\s*(\[[^\]]*\])',
        response_text,
        re.IGNORECASE | re.DOTALL
    )
    if section_match:
        raw_json = section_match.group(1).strip()
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        p_id = str(item.get("id", "")).strip()
                        p_desc = str(item.get("descricao") or item.get("description") or "").strip()
                        p_bbox = item.get("box_2d") or item.get("bbox")
                        num_m = re.search(r'\d+', p_id)
                        if num_m:
                            norm_id = f"Pessoa {num_m.group(0)}"
                            if norm_id not in existing_set and norm_id not in registered_ids:
                                new_people.append({
                                    "id": norm_id,
                                    "descricao": p_desc or f"Características registradas na primeira aparição de {norm_id}.",
                                    "bbox": p_bbox
                                })
                                registered_ids.add(norm_id)
        except Exception:
            pass

    # 2. Fallback: procurar definições no texto como "Pessoa X: <descrição>"
    if not new_people:
        matches = re.finditer(
            r'\b(Pessoa|Person|Indiv[ií]duo)\s+(\d+)\s*[:\-–]\s*([^\n\r]+)',
            response_text,
            re.IGNORECASE
        )
        for m in matches:
            norm_id = f"Pessoa {m.group(2)}"
            desc = m.group(3).strip()
            if norm_id not in existing_set and norm_id not in registered_ids:
                new_people.append({
                    "id": norm_id,
                    "descricao": desc,
                    "bbox": None
                })
                registered_ids.add(norm_id)

    return new_people


def extract_people_from_response(response_text: str) -> str:
    """
    Analisa a resposta gerada pelo modelo de visão e extrai a lista de identificadores
    das pessoas presentes na imagem, retornando uma string JSON (ex: '["Pessoa 1", "Pessoa 2"]').
    
    Possui múltiplos mecanismos de fallback para máxima robustez contra variações de formatação do LLM.
    """
    if not response_text:
        return json.dumps([], ensure_ascii=False)

    # 1. Tentativa de casamento com a seção PESSOAS_PRESENTES: [...]
    section_match = re.search(
        r'(?:PESSOAS_PRESENTES|PESSOAS|PEOPLE_PRESENT|PEOPLE)\s*:\s*(\[[^\]]*\])',
        response_text,
        re.IGNORECASE
    )
    if section_match:
        raw_json = section_match.group(1).strip()
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, list):
                cleaned = [str(item).strip() for item in parsed if str(item).strip()]
                return json.dumps(cleaned, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

    # 2. Tentativa de casamento com seção vazia ou explícita de ausência
    none_match = re.search(
        r'(?:PESSOAS_PRESENTES|PESSOAS|PEOPLE_PRESENT|PEOPLE)\s*:\s*(?:nenhuma|nenhum|0|none|\[\s*\]|sem pessoas)',
        response_text,
        re.IGNORECASE
    )
    if none_match:
        return json.dumps([], ensure_ascii=False)

    # 3. Busca por padrões como 'Pessoa 1', 'Pessoa 2', 'Person 1', 'Indivíduo 1' no texto completo
    people_found = re.findall(
        r'\b(?:Pessoa|Person|Indiv[ií]duo)\s+\d+\b',
        response_text,
        re.IGNORECASE
    )
    if people_found:
        # Padroniza como 'Pessoa X' e remove duplicatas preservando a ordem de aparição
        normalized = []
        for p in people_found:
            num = re.search(r'\d+', p)
            if num:
                name = f"Pessoa {num.group(0)}"
                if name not in normalized:
                    normalized.append(name)
        if normalized:
            return json.dumps(normalized, ensure_ascii=False)

    return json.dumps([], ensure_ascii=False)


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


def encode_image_to_base64(image_path: str) -> str:
    """Lê um arquivo de imagem do disco e o converte para string Base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def query_ollama_vision(
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 120,
    response_format: Optional[str] = None
) -> str:
    """
    Envia a imagem e o prompt para a API REST do Ollama (/api/generate).
    Aceita caminho do arquivo no disco (`image_path`) ou string Base64 em memória (`image_base64`).
    
    :param image_path: Caminho completo para a imagem no disco (opcional)
    :param image_base64: String Base64 da imagem em memória (opcional)
    :param model: Nome do modelo de visão no Ollama (ex: llama3.2-vision, llava)
    :param prompt: Instrução/pergunta enviada ao modelo
    :param ollama_url: URL base do servidor Ollama
    :param timeout: Tempo limite da requisição em segundos
    :return: Texto da descrição gerada pelo modelo
    """
    if image_base64:
        encoded_image = image_base64
    elif image_path:
        encoded_image = encode_image_to_base64(image_path)
    else:
        raise ValueError("É necessário fornecer 'image_path' ou 'image_base64'.")

    api_endpoint = f"{ollama_url.rstrip('/')}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encoded_image],
        "stream": False,
        "options": {"temperature": 0, "seed": 42}
    }
    if response_format:
        payload["format"] = response_format
        payload["options"]["num_predict"] = 512

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_endpoint,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result.get("response", "").strip()
    except urllib.error.URLError as e:
        if isinstance(e.reason, ConnectionRefusedError) or "connection refused" in str(e).lower():
            raise ConnectionError(
                f"Não foi possível conectar ao Ollama em '{ollama_url}'. "
                f"Certifique-se de que o serviço do Ollama está rodando ('ollama serve')."
            ) from e
        raise


def next_person_id(known_people: List[Dict[str, Any]]) -> int:
    """Evita colisões após renomeações, exclusões e fusões de pessoas."""
    numbers = [int(match.group(1)) for person in known_people
               if (match := re.fullmatch(r"Pessoa\s+(\d+)", person["person_label"], re.IGNORECASE))]
    return max(numbers, default=0) + 1


def build_face_crop_recognition_prompt(
    known_people: List[Dict[str, Any]],
    next_id_num: int = 1,
    face_idx: int = 1,
    total_faces: int = 1
) -> str:
    """Descreve somente o que é visível; identidade é decidida pelo SFace, não por texto."""
    return (
        f"Descreva em português apenas o rosto central deste recorte (Rosto {face_idx} de {total_faces}). "
        "Informe traços visíveis: formato do rosto, sobrancelhas, nariz, boca, cabelo e marcas distintivas. "
        "Não invente detalhes de regiões desfocadas ou ocultas; explicite limitações de iluminação, "
        "resolução ou oclusão. Não infira identidade, nome, etnia, personalidade ou saúde. "
        "Não compare com outras pessoas. A identificação é feita separadamente por comparação visual. "
        'Responda somente um objeto JSON: {"description": "descrição objetiva e curta do rosto"}.'
    )


def recognize_face_crop_with_ai(
    face_crop_base64: str,
    known_people: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = 60,
    next_id_num: Optional[int] = None,
    face_idx: int = 1,
    total_faces: int = 1
) -> Dict[str, Any]:
    """Descreve um rosto não associado pelo SFace e reserva um rótulo provisório local."""
    next_id_num = max(next_id_num or 1, next_person_id(known_people))

    prompt = build_face_crop_recognition_prompt(
        known_people=known_people,
        next_id_num=next_id_num,
        face_idx=face_idx,
        total_faces=total_faces
    )

    response_text = query_ollama_vision(
        image_base64=face_crop_base64,
        model=model,
        prompt=prompt,
        ollama_url=ollama_url,
        timeout=timeout,
        response_format="json"
    )

    try:
        result = json.loads(response_text)
    except (ValueError, TypeError) as e:
        raise ValueError("Ollama retornou descrição facial fora do formato JSON esperado.") from e
    description = result.get("description") if isinstance(result, dict) else None
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Ollama não retornou uma descrição facial válida.")

    return {
        "person_label": f"Pessoa {next_id_num}",
        "description": description.strip(),
        "is_new": True,
        "raw_response": response_text
    }


def reset_errors_to_pending(db_path: str) -> int:
    """Redefine o status de registros com erro para 'pending' a fim de retentar."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE images SET status = 'pending' WHERE status = 'error'")
        conn.commit()
        reopened = cursor.rowcount
        logger.info(f"{reopened} imagem(ns) com erro redefinida(s) para 'pending'.")
        return reopened


def process_images(
    db_path: str = DEFAULT_DB_PATH,
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    limit: Optional[int] = None,
    timeout: int = 120,
    verbose: bool = True,
    use_face_detection: bool = True,
    describe_ai: bool = True,
    progress_callback: Optional[Any] = None,
    pause_event: Optional[Any] = None,
    stop_event: Optional[Any] = None
) -> Dict[str, int]:
    """
    Processa as imagens pendentes no banco SQLite utilizando biometria facial e opcionalmente Ollama.
    Quando `use_face_detection` estiver ativado, utiliza detecção facial e reconhecimento biométrico local (SFace + YuNet).
    Quando `describe_ai` estiver ativado, utiliza o Ollama para gerar uma descrição contextual detalhada da foto.

    :param db_path: Caminho do banco SQLite
    :param model: Modelo multimodal do Ollama
    :param prompt: Prompt de análise para o modelo
    :param ollama_url: URL do servidor Ollama
    :param limit: Limite máximo de imagens a processar nesta execução
    :param timeout: Tempo limite por requisição
    :param verbose: Se True, imprime logs detalhados
    :param use_face_detection: Se True, ativa detecção facial local rápida e reconhecimento biométrico
    :param describe_ai: Se True, gera descrição de cena detalhada com o modelo de visão Ollama
    :param progress_callback: Função callback invocada a cada etapa do processamento com dados visuais
    :param pause_event: Objeto threading.Event para controle de pausa/retomada
    :param stop_event: Objeto threading.Event para interrupção graciosa
    :return: Estatísticas do processamento
    """
    setup_logging(logging.INFO if verbose else logging.WARNING)
    init_db(db_path)

    pending = get_pending_images(db_path, limit=limit)
    total_to_process = len(pending)

    if total_to_process == 0:
        logger.info("Nenhuma imagem pendente para processar no banco de dados.")
        if progress_callback:
            try:
                progress_callback({"event": "finished", "processed": 0, "errors": 0, "total": 0, "elapsed_seconds": 0})
            except Exception:
                pass
        return {"processed": 0, "errors": 0, "total": 0}

    # Carrega catálogo de pessoas conhecidas previamente
    known_people = get_known_people(db_path)

    logger.info("=" * 60)
    logger.info("INICIANDO PROCESSAMENTO DE IMAGENS")
    logger.info(f" - Banco de dados: {Path(db_path).resolve()}")
    logger.info(f" - Descrição da foto com IA: {'ATIVADA' if describe_ai else 'DESATIVADA'}")
    if describe_ai:
        logger.info(f" - Modelo de visão: {model}")
        logger.info(f" - Servidor Ollama: {ollama_url}")
    logger.info(f" - Detecção facial rápida em memória: {'ATIVADA' if use_face_detection else 'DESATIVADA'}")
    logger.info(f" - Total de imagens a processar: {total_to_process}")
    logger.info(f" - Pessoas já catalogadas no banco: {len(known_people)}")
    if describe_ai:
        logger.info(f" - Prompt base: \"{prompt}\"")
    logger.info("=" * 60)

    processed_count = 0
    error_count = 0
    start_total_time = time.time()

    for idx, row in enumerate(pending, start=1):
        if stop_event and stop_event.is_set():
            logger.info("Processamento interrompido pelo usuário.")
            break

        if pause_event and not pause_event.is_set():
            logger.info("Processamento pausado. Aguardando liberação...")
            pause_event.wait()
            if stop_event and stop_event.is_set():
                break

        image_id = row["id"]
        file_path = row["file_path"]

        logger.info(f"[{idx}/{total_to_process}] Avaliando imagem ID {image_id}: {file_path}")

        if progress_callback:
            try:
                progress_callback({
                    "event": "start_image",
                    "image_id": image_id,
                    "file_path": file_path,
                    "index": idx,
                    "total": total_to_process
                })
            except Exception as pce:
                logger.debug(f"Erro no progress_callback (start_image): {pce}")

        if not os.path.exists(file_path):
            err_msg = f"Arquivo não encontrado no disco: {file_path}"
            logger.error(f"[ID {image_id}] Falha: {err_msg}")
            update_image_error(db_path, image_id, err_msg)
            error_count += 1
            if progress_callback:
                try:
                    progress_callback({
                        "event": "image_error",
                        "image_id": image_id,
                        "file_path": file_path,
                        "error_message": err_msg,
                        "processed_count": processed_count,
                        "error_count": error_count,
                        "total": total_to_process
                    })
                except Exception:
                    pass
            continue

        item_start = time.time()
        try:
            face_crops = []
            if use_face_detection:
                face_crops = extract_face_crops_in_memory(file_path)

            identified_labels: List[str] = []
            provisional_people: List[Dict[str, Any]] = []

            if progress_callback:
                try:
                    progress_callback({
                        "event": "faces_detected",
                        "image_id": image_id,
                        "file_path": file_path,
                        "face_crops": face_crops,
                        "count": len(face_crops)
                    })
                except Exception as pce:
                    logger.debug(f"Erro no progress_callback (faces_detected): {pce}")

            # Se a detecção facial estiver ativa e nenhum rosto foi detectado na imagem,
            # dispensa a avaliação com IA e avança diretamente para a próxima imagem.
            if use_face_detection and not face_crops:
                item_elapsed = time.time() - item_start
                logger.info(f"[ID {image_id}] Nenhum rosto detectado na imagem. Dispensando avaliação com IA e passando para a próxima imagem.")
                desc_no_face = "Nenhum rosto detectado na imagem."
                people_present = "[]"
                update_image_description(
                    db_path=db_path,
                    image_id=image_id,
                    description=desc_no_face,
                    model_used="face_detection_skip",
                    people_present=people_present,
                    status="processed"
                )
                processed_count += 1
                if progress_callback:
                    try:
                        progress_callback({
                            "event": "image_skipped_no_faces",
                            "image_id": image_id,
                            "file_path": file_path,
                            "processed_count": processed_count,
                            "error_count": error_count,
                            "total": total_to_process
                        })
                        progress_callback({
                            "event": "image_completed",
                            "image_id": image_id,
                            "file_path": file_path,
                            "description": "Nenhum rosto detectado na imagem. Avaliação com IA dispensada.",
                            "people_present": "[]",
                            "duration": item_elapsed,
                            "processed_count": processed_count,
                            "error_count": error_count,
                            "total": total_to_process
                        })
                    except Exception as pce:
                        logger.debug(f"Erro no progress_callback (image_skipped_no_faces): {pce}")
                continue

            if face_crops:
                logger.info(f"[ID {image_id}] Detectado(s) {len(face_crops)} rosto(s) com algoritmo de visão computacional.")
                logger.info(f"[ID {image_id}] Realizando reconhecimento facial biométrico local...")

                matcher = FaceIdentityMatcher()
                identified_labels: List[str] = []
                provisional_people: List[Dict[str, Any]] = []

                for f_idx, crop in enumerate(face_crops, start=1):
                    if stop_event and stop_event.is_set():
                        break

                    # Compara biometricamente o recorte com pessoas conhecidas
                    pil_crop = crop.get("pil_image")
                    if pil_crop is None and crop.get("image_bytes"):
                        try:
                            import io
                            pil_crop = Image.open(io.BytesIO(crop["image_bytes"]))
                        except Exception:
                            pil_crop = None

                    match_res = matcher.match(
                        crop=pil_crop,
                        known_people=known_people + provisional_people,
                        excluded_labels=list(identified_labels)
                    )
                    status = match_res.get("status")
                    score = match_res.get("score")
                    matched_label = match_res.get("person_label")

                    if status == "matched" and matched_label:
                        p_label = matched_label
                        identified_labels.append(p_label)
                        score_str = f"score: {score:.2f}" if score is not None else ""
                        desc_text = f"Identificado via SFace ({score_str})".strip()
                        logger.info(f"[ID {image_id}] Rosto {f_idx} reconhecido como '{p_label}' via SFace ({score_str})")

                        if progress_callback:
                            try:
                                progress_callback({
                                    "event": "face_recognized",
                                    "image_id": image_id,
                                    "face_idx": f_idx,
                                    "total_faces": len(face_crops),
                                    "person_label": p_label,
                                    "description": desc_text,
                                    "is_new": False,
                                    "match_status": status,
                                    "match_score": score,
                                    "crop": crop
                                })
                            except Exception as pce:
                                logger.debug(f"Erro no progress_callback (face_recognized): {pce}")
                    else:
                        # Rosto novo, ambíguo ou de baixa qualidade: cadastra localmente sem chamada de IA
                        next_id_num = next_person_id(known_people + provisional_people)
                        p_label = f"Pessoa {next_id_num}"
                        identified_labels.append(p_label)

                        # Salva o recorte de referência
                        crops_dir = Path(db_path).parent / "face_crops" if db_path else Path(DEFAULT_CROPS_DIR)
                        crops_dir.mkdir(parents=True, exist_ok=True)
                        safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', p_label.lower())
                        crop_file_path = str((crops_dir / f"crop_{safe_label}.jpg").resolve())
                        try:
                            with open(crop_file_path, "wb") as f_out:
                                f_out.write(crop["image_bytes"])
                        except Exception as ce:
                            logger.debug(f"Erro ao salvar arquivo de recorte para {p_label}: {ce}")
                            crop_file_path = None

                        bbox_json = json.dumps(crop.get("box_2d_norm"))
                        desc_text = "Catalogada via biometria facial."
                        provisional_people.append({
                            "person_label": p_label,
                            "description": desc_text,
                            "first_seen_image_id": image_id,
                            "face_crop_path": crop_file_path,
                            "face_bbox": bbox_json
                        })

                        logger.info(f"[ID {image_id}] Nova pessoa catalogada via biometria ({p_label})")

                        if progress_callback:
                            try:
                                progress_callback({
                                    "event": "face_recognized",
                                    "image_id": image_id,
                                    "face_idx": f_idx,
                                    "total_faces": len(face_crops),
                                    "person_label": p_label,
                                    "description": desc_text,
                                    "is_new": True,
                                    "match_status": status,
                                    "match_score": score,
                                    "crop": crop
                                })
                            except Exception as pce:
                                logger.debug(f"Erro no progress_callback (face_recognized): {pce}")

                if stop_event and stop_event.is_set():
                    logger.info("Processamento interrompido pelo usuário.")
                    break

                if describe_ai:
                    # Gera a descrição geral da foto com IA
                    if identified_labels:
                        scene_prompt = (
                            f"{prompt.strip()}\n\n"
                            f"Pessoas presentes na foto: {', '.join(identified_labels)}. "
                            f"Descreva detalhadamente a foto em geral (cenário, ambiente, objetos, iluminação e ações)."
                        )
                    else:
                        scene_prompt = (
                            f"{prompt.strip()}\n\n"
                            f"Descreva detalhadamente a foto em geral (cenário, ambiente, objetos, iluminação e ações)."
                        )

                    logger.debug(f"[ID {image_id}] Solicitando descrição geral da foto com IA...")
                    if progress_callback:
                        try:
                            progress_callback({
                                "event": "ai_thinking",
                                "image_id": image_id,
                                "stage": "scene_description",
                                "prompt": scene_prompt
                            })
                        except Exception:
                            pass

                    description = query_ollama_vision(
                        image_path=file_path,
                        model=model,
                        prompt=scene_prompt,
                        ollama_url=ollama_url,
                        timeout=timeout
                    )
                    model_used_record = model
                else:
                    if identified_labels:
                        description = f"Reconhecimento facial biométrico local. Pessoas identificadas: {', '.join(identified_labels)}."
                    else:
                        description = "Reconhecimento facial biométrico local."
                    model_used_record = "biometric_sface"
                    logger.info(f"[ID {image_id}] Descrição via IA ignorada (opção desativada).")

                if stop_event and stop_event.is_set():
                    logger.info("Processamento interrompido pelo usuário.")
                    break

                # Persiste as novas pessoas no banco apenas após a conclusão bem-sucedida do processamento
                for prov in provisional_people:
                    register_known_person(
                        db_path,
                        prov["person_label"],
                        prov["description"],
                        first_seen_image_id=image_id,
                        face_bbox=prov["face_bbox"]
                    )
                    if prov["face_crop_path"]:
                        update_person_face_crop(db_path, prov["person_label"], prov["face_crop_path"], face_bbox=prov["face_bbox"])
                    known_people.append(prov)

                people_present = json.dumps(identified_labels, ensure_ascii=False)

            else:
                if describe_ai:
                    logger.debug(f"[ID {image_id}] Analisando imagem completa...")
                    current_prompt = build_prompt_with_catalog(prompt, known_people)
                    logger.debug(f"[ID {image_id}] Codificando imagem em Base64 e enviando requisição ao Ollama...")
                    if progress_callback:
                        try:
                            progress_callback({
                                "event": "ai_thinking",
                                "image_id": image_id,
                                "stage": "full_scene",
                                "prompt": current_prompt
                            })
                        except Exception:
                            pass

                    description = query_ollama_vision(
                        image_path=file_path,
                        model=model,
                        prompt=current_prompt,
                        ollama_url=ollama_url,
                        timeout=timeout
                    )
                    people_present = extract_people_from_response(description)

                    # Identifica e cataloga novas pessoas detectadas via modelo geral
                    known_labels = [p["person_label"] for p in known_people]
                    new_people = extract_new_people_from_response(description, known_labels=known_labels)
                    for new_p in new_people:
                        bbox_str = json.dumps(new_p.get("bbox")) if new_p.get("bbox") else None
                        if register_known_person(
                            db_path,
                            new_p["id"],
                            new_p["descricao"],
                            first_seen_image_id=image_id,
                            face_bbox=bbox_str
                        ):
                            crop_p = None
                            try:
                                crop_p = ensure_face_crop_for_person(db_path, new_p["id"])
                            except Exception as ce:
                                logger.debug(f"[ID {image_id}] Recorte facial postergado para '{new_p['id']}': {ce}")

                            known_people.append({
                                "person_label": new_p["id"],
                                "description": new_p["descricao"],
                                "first_seen_image_id": image_id,
                                "face_crop_path": crop_p,
                                "face_bbox": bbox_str
                            })
                            logger.info(f"[ID {image_id}] Novo perfil cadastrado no catálogo: {new_p['id']} - {new_p['descricao'][:80]}...")
                    model_used_record = model
                else:
                    description = "Processamento concluído sem análise de IA."
                    people_present = "[]"
                    model_used_record = "none"
                    logger.info(f"[ID {image_id}] Análise com IA dispensada.")

            update_image_description(
                db_path=db_path,
                image_id=image_id,
                description=description,
                model_used=model_used_record,
                people_present=people_present
            )
            if identified_labels:
                update_image_type(db_path=db_path, image_id=image_id, image_type="photo")
            processed_count += 1
            item_elapsed = time.time() - item_start

            preview = description.replace("\n", " ")[:140] + ("..." if len(description) > 140 else "")
            logger.info(f"[ID {image_id}] Descrição gerada com sucesso ({item_elapsed:.2f}s): {preview}")
            try:
                people_list = json.loads(people_present)
                if people_list:
                    logger.info(f"[ID {image_id}] Pessoas identificadas na imagem: {', '.join(people_list)}")
                else:
                    logger.debug(f"[ID {image_id}] Nenhuma pessoa identificada na imagem.")
            except Exception:
                logger.debug(f"[ID {image_id}] Pessoas identificadas: {people_present}")

            if progress_callback:
                try:
                    progress_callback({
                        "event": "image_completed",
                        "image_id": image_id,
                        "file_path": file_path,
                        "description": description,
                        "people_present": people_present,
                        "duration": item_elapsed,
                        "processed_count": processed_count,
                        "error_count": error_count,
                        "total": total_to_process
                    })
                except Exception as pce:
                    logger.debug(f"Erro no progress_callback (image_completed): {pce}")

        except KeyboardInterrupt:
            logger.warning("Processamento interrompido pelo usuário via teclado.")
            break
        except ConnectionError as ce:
            logger.critical(f"Falha de conexão com o Ollama: {ce}")
            if progress_callback:
                try:
                    progress_callback({
                        "event": "image_error",
                        "image_id": image_id,
                        "file_path": file_path,
                        "error_message": f"Falha de conexão com Ollama: {ce}",
                        "processed_count": processed_count,
                        "error_count": error_count + 1,
                        "total": total_to_process
                    })
                except Exception:
                    pass
            break
        except Exception as e:
            err_msg = str(e)
            update_image_error(db_path, image_id, err_msg)
            error_count += 1
            logger.error(f"[ID {image_id}] Falha ao avaliar imagem com Ollama: {err_msg}")
            if progress_callback:
                try:
                    progress_callback({
                        "event": "image_error",
                        "image_id": image_id,
                        "file_path": file_path,
                        "error_message": err_msg,
                        "processed_count": processed_count,
                        "error_count": error_count,
                        "total": total_to_process
                    })
                except Exception:
                    pass

    total_elapsed = time.time() - start_total_time
    stats = get_statistics(db_path)
    logger.info("=" * 60)
    logger.info("RESUMO DO PROCESSAMENTO CONCLUÍDO")
    logger.info(f" - Tempo total decorrido: {total_elapsed:.2f} segundos")
    logger.info(f" - Processadas com sucesso nesta sessão: {processed_count}")
    logger.info(f" - Falhas / Erros nesta sessão: {error_count}")
    logger.info(f" - Situação geral do banco: {stats['processed']}/{stats['total']} concluídas "
                f"({stats['pending']} pendentes, {stats['errors']} com erro)")
    logger.info("=" * 60)

    if progress_callback:
        try:
            progress_callback({
                "event": "finished",
                "processed": processed_count,
                "errors": error_count,
                "total": total_to_process,
                "elapsed_seconds": int(total_elapsed)
            })
        except Exception as pce:
            logger.debug(f"Erro no progress_callback (finished): {pce}")

    return {
        "processed": processed_count,
        "errors": error_count,
        "total": total_to_process,
        "elapsed_seconds": int(total_elapsed)
    }


def get_available_ollama_models(ollama_url: str = DEFAULT_OLLAMA_URL, timeout: int = 4) -> List[str]:
    """
    Consulta a API do Ollama (/api/tags) e retorna a lista de nomes dos modelos instalados localmente.
    """
    api_endpoint = f"{ollama_url.rstrip('/')}/api/tags"
    req = urllib.request.Request(api_endpoint, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            models = data.get("models", [])
            names = [m.get("name") for m in models if m.get("name")]
            return names
    except Exception as e:
        logger.debug(f"Não foi possível obter lista de modelos do Ollama em {ollama_url}: {e}")
        return []


def interactive_describe_prompt(db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, int]]:
    """
    Interface interativa guiada no terminal para o processo de avaliação visual com Ollama.
    Permite escolher modelos instalados, ajustar prompt, definir limites e retentar erros.
    """
    init_db(db_path)
    stats = get_statistics(db_path)

    print("\n" + "=" * 60)
    print(" [*] AVALIACAO VISUAL DE IMAGENS COM IA (OLLAMA)")
    print("=" * 60)
    print(f"Banco de Dados: {os.path.abspath(db_path)}")
    print(f" - Total de imagens:     {stats['total']}")
    print(f" - Imagens pendentes:    {stats['pending']}")
    print(f" - Imagens processadas:  {stats['processed']}")
    print(f" - Imagens com erro:     {stats['errors']}")
    print(f" - Pessoas catalogadas:  {stats.get('known_people', 0)}")
    print("-" * 60)

    # Verifica imagens pendentes e com erro
    if stats["pending"] == 0:
        if stats["errors"] > 0:
            print(f"[i] Nenhuma imagem com status 'pending', mas existem {stats['errors']} com status 'error'.")
            try:
                retry = input("Deseja redefinir os erros para pendente e tentar novamente? [S/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return None
            if retry not in ("n", "nao", "não"):
                reset_errors_to_pending(db_path)
                stats = get_statistics(db_path)
            else:
                print("Nenhuma imagem para processar. Retornando ao menu.")
                return None
        else:
            print("[OK] Todas as imagens cadastradas no banco já foram processadas!")
            return None

    # Opção de Modo de Processamento
    print("\nModo de Processamento:")
    print(" 1. Completo: Biometria Facial Local + Descrição da Foto com IA (Ollama) [Recomendado]")
    print(" 2. Apenas Biometria Facial: Detecção e catalogação de pessoas (Sem chamada à IA)")
    try:
        mode_choice = input("Escolha [1/2, Padrao: 1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    describe_ai = (mode_choice != "2")
    selected_model = DEFAULT_MODEL
    custom_prompt = DEFAULT_PROMPT

    if describe_ai:
        # Consulta modelos no Ollama
        print("\nConectando ao Ollama...")
        models = get_available_ollama_models(DEFAULT_OLLAMA_URL)

        if models:
            print(f"Modelos detectados no Ollama ({len(models)}):")
            for i, m in enumerate(models, start=1):
                marker = " (Padrao sugerido)" if DEFAULT_MODEL in m else ""
                print(f" [{i}] {m}{marker}")
            print(f" [{len(models) + 1}] Digitar outro nome de modelo manualmente")

            try:
                choice = input(f"\nSelecione o modelo desejado [1-{len(models) + 1}, Padrao: 1]: ").strip()
            except (EOFError, KeyboardInterrupt):
                return None

            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(models):
                    selected_model = models[idx - 1]
                elif idx == len(models) + 1:
                    try:
                        custom_name = input("Digite o nome exato do modelo no Ollama: ").strip()
                        if custom_name:
                            selected_model = custom_name
                    except (EOFError, KeyboardInterrupt):
                        return None
            elif choice:
                selected_model = choice
        else:
            print(f"[!] Nao foi possivel listar os modelos automaticamente via {DEFAULT_OLLAMA_URL}.")
            try:
                m_input = input(f"Digite o nome do modelo do Ollama [Padrao: {DEFAULT_MODEL}]: ").strip()
                if m_input:
                    selected_model = m_input
            except (EOFError, KeyboardInterrupt):
                return None

        print(f"[OK] Modelo selecionado: {selected_model}")
    else:
        print("[OK] Modo selecionado: Apenas Biometria Facial Local (Sem IA).")

    # Limite de imagens
    print(f"\nQuantidade de imagens a processar nesta sessao (pendentes: {stats['pending']}):")
    try:
        limit_input = input("Digite o numero de imagens ou pressione ENTER para processar todas: ").strip()
        limit = int(limit_input) if limit_input.isdigit() and int(limit_input) > 0 else None
    except (EOFError, KeyboardInterrupt):
        return None

    if limit:
        print(f"[OK] Limite configurado: {limit} imagem(ns).")
    else:
        print(f"[OK] Processando todas as {stats['pending']} imagens pendentes.")

    # Opcao de redefinir erros
    if stats["errors"] > 0:
        try:
            retry_err = input(f"\nDeseja tambem incluir as {stats['errors']} imagens com falhas anteriores? [s/N]: ").strip().lower()
            if retry_err in ("s", "sim", "y", "yes"):
                reset_errors_to_pending(db_path)
        except (EOFError, KeyboardInterrupt):
            pass

    if describe_ai:
        # Prompt personalizado opcional
        print("\nOpcoes de Prompt:")
        print(" 1. Prompt padrao inteligente (Identificacao consistente de pessoas + Descricao geral) [Recomendado]")
        print(" 2. Digitar prompt customizado")
        try:
            p_choice = input("Escolha [1/2, Padrao: 1]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if p_choice == "2":
            try:
                p_text = input("Digite o prompt customizado: ").strip()
                if p_text:
                    custom_prompt = p_text
            except (EOFError, KeyboardInterrupt):
                return None

    print(f"\nIniciando processamento{' com IA' if describe_ai else ' biometrico local'}...")
    return process_images(
        db_path=db_path,
        model=selected_model,
        prompt=custom_prompt,
        ollama_url=DEFAULT_OLLAMA_URL,
        limit=limit,
        describe_ai=describe_ai,
        verbose=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Avalia imagens registradas no SQLite utilizando modelos de visão do Ollama e salva as descrições."
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Executa em modo interativo guiado com perguntas no terminal"
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Caminho do arquivo SQLite (Padrão: {DEFAULT_DB_PATH})"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Nome do modelo multimodal no Ollama (ex: llama3.2-vision, llava, moondream). Padrão: {DEFAULT_MODEL}"
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt de instrução para análise da imagem."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_OLLAMA_URL,
        help=f"URL do servidor Ollama (Padrão: {DEFAULT_OLLAMA_URL})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Quantidade máxima de imagens para processar nesta execução."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout em segundos para cada requisição ao Ollama (Padrão: 120)"
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Redefine imagens marcadas com status de erro para 'pending' antes de iniciar."
    )
    parser.add_argument(
        "--no-face-detect",
        action="store_true",
        help="Desativa a detecção facial rápida prévia e avalia apenas a imagem inteira diretamente."
    )
    parser.add_argument(
        "--no-ai", "--skip-ai",
        action="store_true",
        dest="no_ai",
        help="Desativa a descrição da imagem com IA (Ollama) e executa apenas a catalogação/reconhecimento biométrico local."
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

    if args.interactive or (len(sys.argv) == 1):
        interactive_describe_prompt(db_path=args.db)
        return

    if args.retry_errors:
        reopened = reset_errors_to_pending(args.db)

    process_images(
        db_path=args.db,
        model=args.model,
        prompt=args.prompt,
        ollama_url=args.url,
        limit=args.limit,
        timeout=args.timeout,
        verbose=True,
        use_face_detection=not args.no_face_detect,
        describe_ai=not args.no_ai
    )


if __name__ == "__main__":
    main()
