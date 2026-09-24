"""
Módulo de detecção e recorte facial para o ImageSorter.
Responsável por isolar o rosto das pessoas em imagens e salvar miniaturas/recortes de referência.
"""
import os
import io
import re
import json
import base64
import logging
import math
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
from PIL import Image, ImageDraw

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None
    np = None

logger = logging.getLogger("ImageSorter.FaceCropper")

DEFAULT_CROPS_DIR = "face_crops"
YUNET_MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
_YUNET_LOCAL = threading.local()


def _get_yunet_detector():
    """Mantém uma instância por thread, pois setInputSize altera o detector."""
    if not hasattr(_YUNET_LOCAL, "detector"):
        _YUNET_LOCAL.detector = None
        try:
            if not YUNET_MODEL_PATH.is_file():
                raise FileNotFoundError(YUNET_MODEL_PATH)
            _YUNET_LOCAL.detector = cv2.FaceDetectorYN.create(
                str(YUNET_MODEL_PATH), "", (320, 320), 0.85, 0.3, 5000
            )
            logger.info("Detector facial YuNet carregado (CPU, processamento local).")
        except Exception as e:
            logger.warning(f"YuNet indisponível; usando Haar Cascades: {e}")
    return _YUNET_LOCAL.detector


def _detect_faces_yunet(img_np, detector, min_size):
    """Detecta em duas escalas limitadas e restaura coordenadas da imagem original."""
    if img_np.ndim == 2:
        bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    sizes = [min(640, max(width, height))]
    if max(width, height) > 640:
        sizes.append(min(1280, max(width, height)))
    boxes, scores = [], []
    for size in sizes:
        scale = size / max(width, height)
        target_w, target_h = max(1, round(width * scale)), max(1, round(height * scale))
        resized = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        # YuNet exige uma entrada de pelo menos 32 pixels por dimensão.
        resized = cv2.copyMakeBorder(resized, 0, max(0, 32 - target_h),
                                     0, max(0, 32 - target_w), cv2.BORDER_CONSTANT)
        detector.setInputSize((resized.shape[1], resized.shape[0]))
        _, faces = detector.detect(resized)
        if faces is None:
            continue
        for face in faces:
            x, y, w, h = map(float, face[:4])
            score = float(face[-1])
            if not all(math.isfinite(v) for v in (x, y, w, h, score)) or score < 0.85:
                continue
            left = max(0, int(x * width / target_w))
            top = max(0, int(y * height / target_h))
            right = min(width, math.ceil((x + w) * width / target_w))
            bottom = min(height, math.ceil((y + h) * height / target_h))
            if right - left < min_size[0] or bottom - top < min_size[1]:
                continue
            boxes.append([left, top, right - left, bottom - top])
            scores.append(score)
    if not boxes:
        return []
    # Prioriza confiança, não tamanho, ao eliminar detecções repetidas entre escalas.
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.85, 0.3)
    result = []
    for index in np.asarray(indices).reshape(-1):
        x, y, w, h = boxes[int(index)]
        result.append((x, y, x + w, y + h))
    return sorted(result, key=lambda box: (box[0], box[1]))


# Cache de classificadores Haar Cascade
_CASCADE_CACHE: Dict[str, Any] = {}


def _get_cascade(cascade_name: str = "haarcascade_frontalface_default.xml"):
    """Carrega e armazena em cache o classificador Haar Cascade do OpenCV."""
    if not OPENCV_AVAILABLE or cv2 is None:
        return None
    if cascade_name in _CASCADE_CACHE:
        return _CASCADE_CACHE[cascade_name]
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, cascade_name)
        if os.path.exists(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                _CASCADE_CACHE[cascade_name] = cascade
                return cascade
    except Exception as e:
        logger.debug(f"Erro ao carregar Haar Cascade '{cascade_name}': {e}")
    return None


def _merge_overlapping_boxes(boxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.3) -> List[Tuple[int, int, int, int]]:
    """Combina caixas delimitadoras com sobreposição significativa (NMS simplificado)."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        l1, t1, r1, b1 = b
        area1 = (r1 - l1) * (b1 - t1)
        overlap = False
        for k in kept:
            l2, t2, r2, b2 = k
            area2 = (r2 - l2) * (b2 - t2)
            inter_l = max(l1, l2)
            inter_t = max(t1, t2)
            inter_r = min(r1, r2)
            inter_b = min(b1, b2)
            if inter_r > inter_l and inter_b > inter_t:
                inter_area = (inter_r - inter_l) * (inter_b - inter_t)
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou > iou_thresh or (inter_area / max(1, area1)) > 0.6:
                    overlap = True
                    break
        if not overlap:
            kept.append(b)
    # Ordena da esquerda para a direita na imagem
    return sorted(kept, key=lambda b: b[0])


def detect_faces_local(
    image_input: Union[str, Image.Image, bytes, Any],
    min_size: Tuple[int, int] = (20, 20),
    scale_factor: float = 1.1,
    min_neighbors: int = 4
) -> List[Tuple[int, int, int, int]]:
    """
    Detecta rostos na CPU com YuNet multiescala; usa Haar se o modelo estiver indisponível.
    Falhas de leitura/detecção geram RuntimeError, não uma lista de rostos vazia.

    :param image_input: Caminho do arquivo, objeto PIL.Image, bytes ou ndarray
    :param min_size: Tamanho mínimo (largura, altura) da face em pixels
    :param scale_factor: Fator de escala do detector Haar de reserva
    :param min_neighbors: Mínimo de vizinhos para confirmar detecção Haar
    :return: Lista de caixas delimitadoras [(left, top, right, bottom), ...] ordenadas da esquerda para a direita
    """
    if not OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV não disponível para detecção facial.")

    try:
        # Normaliza a imagem para array numpy / grayscale
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(image_input)
            with Image.open(image_input) as pil_img:
                pil_img = pil_img.convert("RGB")
                img_np = np.array(pil_img)
        elif isinstance(image_input, bytes):
            with Image.open(io.BytesIO(image_input)) as pil_img:
                pil_img = pil_img.convert("RGB")
                img_np = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            img_np = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            img_np = image_input
            if len(img_np.shape) == 2:
                # Já é grayscale
                gray = img_np
            elif img_np.shape[2] == 3:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
        else:
            raise TypeError("Formato de imagem não suportado para detecção facial.")

        detector = _get_yunet_detector()
        if detector is not None:
            try:
                boxes = _detect_faces_yunet(img_np, detector, min_size)
                logger.info(f"YuNet: {len(boxes)} rosto(s) detectado(s).")
                return boxes
            except Exception as e:
                logger.warning(f"Falha no YuNet; tentando Haar Cascades: {e}")

        if 'gray' not in locals():
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Equalização de histograma para melhor contraste facial
        try:
            gray_eq = cv2.equalizeHist(gray)
        except Exception:
            gray_eq = gray

        img_h, img_w = gray.shape[:2]
        all_boxes: List[Tuple[int, int, int, int]] = []

        # Cascades a testar: frontal principal e frontal alternativo
        cascades_to_try = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_frontalface_alt.xml"
        ]

        loaded_cascades = 0
        for cascade_name in cascades_to_try:
            cascade = _get_cascade(cascade_name)
            if cascade is None:
                continue
            loaded_cascades += 1

            faces = cascade.detectMultiScale(
                gray_eq,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size
            )

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    left = max(0, int(x))
                    top = max(0, int(y))
                    right = min(img_w, int(x + w))
                    bottom = min(img_h, int(y + h))
                    all_boxes.append((left, top, right, bottom))

        # Se nada foi encontrado na imagem equalizada, tenta na original
        if not all_boxes and gray_eq is not gray:
            cascade = _get_cascade("haarcascade_frontalface_default.xml")
            if cascade:
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=min_size
                )
                for (x, y, w, h) in faces:
                    left = max(0, int(x))
                    top = max(0, int(y))
                    right = min(img_w, int(x + w))
                    bottom = min(img_h, int(y + h))
                    all_boxes.append((left, top, right, bottom))

        if not loaded_cascades:
            raise RuntimeError("Nenhum detector facial disponível (YuNet ou Haar).")
        merged = _merge_overlapping_boxes(all_boxes)
        logger.info(f"Haar Cascades: {len(merged)} rosto(s) detectado(s).")
        return merged

    except Exception as e:
        raise RuntimeError(f"Erro na detecção facial local: {e}") from e


def encode_pil_to_base64(pil_img: Image.Image, format: str = "JPEG", quality: int = 90) -> str:
    """Converte um objeto PIL.Image diretamente para string Base64 em memória."""
    buffered = io.BytesIO()
    if format.upper() == "JPEG" and pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffered, format=format, quality=quality)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_face_crops_in_memory(
    image_input: Union[str, Image.Image, bytes],
    bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    padding_pct: float = 0.25,
    max_dim: int = 400
) -> List[Dict[str, Any]]:
    """
    Recorta e extrai todos os rostos da imagem diretamente em memória sem necessidade de escrita em disco.

    :param image_input: Caminho do arquivo, objeto PIL.Image ou bytes da imagem
    :param bboxes: Lista opcional de caixas delimitadoras [(left, top, right, bottom)].
                   Se None, executa detecção facial automática local.
    :param padding_pct: Margem percentual de contexto ao redor da face (cabelo/queixo)
    :param max_dim: Dimensão máxima do recorte gerado
    :return: Lista de dicionários contendo:
             - 'bbox': (left, top, right, bottom)
             - 'pil_image': Objeto PIL.Image do recorte
             - 'base64': String Base64 do recorte JPEG
             - 'image_bytes': Bytes brutos do recorte JPEG
             - 'box_2d_norm': [ymin, xmin, ymax, xmax] normalizados de 0 a 1000
    """
    should_close = False
    img = None
    try:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(image_input)
            img = Image.open(image_input)
            should_close = True
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
            should_close = True
        elif isinstance(image_input, Image.Image):
            img = image_input
            should_close = False
        else:
            raise TypeError("Formato de imagem não suportado para recorte facial.")

        img_rgb = img.convert("RGB")
        width, height = img_rgb.size

        if bboxes is None:
            bboxes = detect_faces_local(img_rgb)

        if not bboxes:
            return []

        crops_result: List[Dict[str, Any]] = []

        for box in bboxes:
            left, top, right, bottom = box
            box_w = right - left
            box_h = bottom - top

            pad_x = int(box_w * padding_pct)
            pad_y = int(box_h * padding_pct)

            c_left = max(0, left - pad_x)
            c_top = max(0, top - pad_y)
            c_right = min(width, right + pad_x)
            c_bottom = min(height, bottom + pad_y)

            cropped = img_rgb.crop((c_left, c_top, c_right, c_bottom))
            cropped.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=90)
            raw_bytes = buf.getvalue()
            b64_str = base64.b64encode(raw_bytes).decode("utf-8")

            # Normalização 0-1000
            norm_box = [
                int((top / height) * 1000),
                int((left / width) * 1000),
                int((bottom / height) * 1000),
                int((right / width) * 1000)
            ]

            crops_result.append({
                "bbox": (c_left, c_top, c_right, c_bottom),
                "original_face_bbox": (left, top, right, bottom),
                "pil_image": cropped,
                "base64": b64_str,
                "image_bytes": raw_bytes,
                "box_2d_norm": norm_box
            })

        return crops_result

    except Exception as e:
        logger.error(f"Erro ao extrair recortes faciais em memória: {e}")
        raise RuntimeError(f"Erro ao extrair recortes faciais em memória: {e}") from e
    finally:
        if should_close and img is not None:
            img.close()


def parse_bounding_box(raw_bbox: Any, img_width: int, img_height: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Normaliza e converte diferentes formatos de bounding box para coordenadas de pixel (left, top, right, bottom).
    Suporta:
    - [ymin, xmin, ymax, xmax] normalizados (0-1000 ou 0.0-1.0)
    - [xmin, ymin, xmax, ymax]
    - Dicionários {"ymin": ..., "xmin": ..., "ymax": ..., "xmax": ...}
    """
    if not raw_bbox:
        return None

    if isinstance(raw_bbox, str):
        try:
            raw_bbox = json.loads(raw_bbox)
        except Exception:
            nums = re.findall(r'[-+]?\d*\.?\d+', raw_bbox)
            if len(nums) >= 4:
                raw_bbox = [float(n) for n in nums[:4]]
            else:
                return None

    if isinstance(raw_bbox, dict):
        if "box_2d" in raw_bbox:
            raw_bbox = raw_bbox["box_2d"]
        elif "ymin" in raw_bbox and "xmin" in raw_bbox:
            raw_bbox = [raw_bbox["ymin"], raw_bbox["xmin"], raw_bbox["ymax"], raw_bbox["xmax"]]

    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
        return None

    try:
        y1, x1, y2, x2 = float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])
    except (ValueError, TypeError):
        return None

    # Se as coordenadas foram dadas em escala 0-1000 (padrão de modelos como Qwen-VL / LLaVA / Llama-Vision)
    if max(y1, x1, y2, x2) > 1.0 and max(y1, x1, y2, x2) <= 1000.0:
        top = int((y1 / 1000.0) * img_height)
        left = int((x1 / 1000.0) * img_width)
        bottom = int((y2 / 1000.0) * img_height)
        right = int((x2 / 1000.0) * img_width)
    # Se normalizadas 0.0 a 1.0
    elif max(y1, x1, y2, x2) <= 1.0:
        top = int(y1 * img_height)
        left = int(x1 * img_width)
        bottom = int(y2 * img_height)
        right = int(x2 * img_width)
    # Se já em pixels absolutos
    else:
        top, left, bottom, right = int(y1), int(x1), int(y2), int(x2)

    # Garante ordenação correta left < right e top < bottom
    left, right = min(left, right), max(left, right)
    top, bottom = min(top, bottom), max(top, bottom)

    # Clampa aos limites da imagem
    left = max(0, min(left, img_width - 1))
    top = max(0, min(top, img_height - 1))
    right = max(left + 1, min(right, img_width))
    bottom = max(top + 1, min(bottom, img_height))

    return left, top, right, bottom


def detect_face_bbox_with_ollama(
    image_path: str,
    person_description: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
    model: str = "llama3.2-vision",
    timeout: int = 45
) -> Optional[List[int]]:
    """
    Solicita ao modelo multimodal do Ollama as coordenadas 2D do rosto da pessoa na foto.
    """
    from describe_images import query_ollama_vision

    prompt = (
        "Localize o rosto da pessoa principal na imagem. "
        "Retorne OBRIGATORIAMENTE APENAS um JSON no formato: {\"box_2d\": [ymin, xmin, ymax, xmax]} "
        "com valores inteiros entre 0 e 1000 representando o retângulo delimitador exato do rosto."
    )
    if person_description:
        prompt += f" Pessoa a localizar: {person_description}"

    try:
        response_text = query_ollama_vision(
            image_path=image_path,
            model=model,
            prompt=prompt,
            ollama_url=ollama_url,
            timeout=timeout
        )
        match = re.search(r'\{[^{}]*"box_2d"\s*:\s*\[([0-9,\s]+)\][^{}]*\}', response_text)
        if match:
            box_str = f"[{match.group(1)}]"
            return json.loads(box_str)

        # Fallback para qualquer lista de 4 números
        list_match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', response_text)
        if list_match:
            return [int(list_match.group(i)) for i in range(1, 5)]
    except Exception as e:
        logger.debug(f"Detecção de rosto via Ollama falhou para {image_path}: {e}")

    return None


def crop_face(
    image_path: str,
    bbox: Optional[Any] = None,
    output_path: Optional[str] = None,
    padding_pct: float = 0.25,
    fallback_to_center_portrait: bool = True
) -> Optional[str]:
    """
    Recorta a região do rosto a partir de uma imagem no disco e salva o arquivo de recorte.

    :param image_path: Caminho da imagem de origem
    :param bbox: Bounding box [ymin, xmin, ymax, xmax] ou dict (opcional)
    :param output_path: Caminho de destino para salvar o recorte JPEG/PNG
    :param padding_pct: Percentual de margem extra ao redor do rosto para incluir cabelo/queixo
    :param fallback_to_center_portrait: Se True, recorta o terço superior central quando bbox não for fornecido
    :return: Caminho do arquivo de recorte gerado ou None em caso de falha
    """
    if not os.path.exists(image_path):
        logger.warning(f"Imagem não encontrada para recorte: {image_path}")
        return None

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size

            pixel_box = parse_bounding_box(bbox, width, height) if bbox else None

            # Se não houver bbox fornecido, tenta detectar com algoritmo local primeiro
            if not pixel_box:
                local_faces = detect_faces_local(img)
                if local_faces:
                    pixel_box = local_faces[0]

            if not pixel_box and fallback_to_center_portrait:
                # Fallback: Região do terço superior central da foto (onde rostos costumam estar)
                left = int(width * 0.20)
                top = int(height * 0.05)
                right = int(width * 0.80)
                bottom = int(height * 0.65)
                # Mantém formato aproximadamente quadrado
                box_w = right - left
                box_h = bottom - top
                if box_w > box_h:
                    mid_x = (left + right) // 2
                    left = max(0, mid_x - box_h // 2)
                    right = min(width, mid_x + box_h // 2)
                pixel_box = (left, top, right, bottom)

            if not pixel_box:
                return None

            left, top, right, bottom = pixel_box

            # Aplica padding ao redor do rosto
            box_w = right - left
            box_h = bottom - top
            pad_x = int(box_w * padding_pct)
            pad_y = int(box_h * padding_pct)

            crop_left = max(0, left - pad_x)
            crop_top = max(0, top - pad_y)
            crop_right = min(width, right + pad_x)
            crop_bottom = min(height, bottom + pad_y)

            cropped_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))

            # Redimensiona mantendo proporção com tamanho agradável para visualização (ex: max 400x400)
            cropped_img.thumbnail((400, 400), Image.Resampling.LANCZOS)

            if not output_path:
                os.makedirs(DEFAULT_CROPS_DIR, exist_ok=True)
                base_name = Path(image_path).stem
                output_path = os.path.join(DEFAULT_CROPS_DIR, f"face_{base_name}_{crop_left}_{crop_top}.jpg")
            else:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            cropped_img.save(output_path, format="JPEG", quality=90)
            logger.debug(f"Recorte facial salvo em: {output_path}")
            return output_path

    except Exception as e:
        logger.error(f"Erro ao gerar recorte facial para {image_path}: {e}")
        return None


def ensure_face_crop_for_person(
    db_path: str,
    person_label: str,
    crops_dir: str = DEFAULT_CROPS_DIR,
    use_ollama_locator: bool = False
) -> Optional[str]:
    """
    Garante que a pessoa indicada possua um arquivo de recorte facial disponível no disco e no SQLite.
    Se não possuir, busca a imagem de referência correspondente, gera o recorte e atualiza o banco.
    """
    from db import get_known_people, update_person_face_crop, get_person_images, search_images

    people = get_known_people(db_path)
    person = next((p for p in people if p["person_label"] == person_label), None)

    # 1. Verifica se já existe caminho registrado válido no disco
    if person and person.get("face_crop_path"):
        f_crop = person["face_crop_path"]
        if os.path.isfile(f_crop):
            return f_crop
        # Tenta resolver relativo ao diretório do banco
        db_dir = os.path.dirname(os.path.abspath(db_path)) if db_path else ""
        rel_crop = os.path.join(db_dir, f_crop)
        if os.path.isfile(rel_crop):
            update_person_face_crop(db_path, person_label, rel_crop)
            return rel_crop

    # 2. Verifica se o arquivo padrão já existe na pasta de recortes
    target_crops_dir = Path(db_path).parent / "face_crops" if db_path else Path(crops_dir)
    target_crops_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', person_label.lower())
    output_crop_path = str((target_crops_dir / f"crop_{safe_label}.jpg").resolve())

    if os.path.isfile(output_crop_path):
        update_person_face_crop(db_path, person_label, output_crop_path)
        return output_crop_path

    # 3. Identifica o caminho da imagem de origem
    source_img_path = None
    if person and person.get("first_seen_file_path"):
        cand = person["first_seen_file_path"]
        if os.path.isfile(cand):
            source_img_path = cand
        elif db_path and os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(db_path)), cand)):
            source_img_path = os.path.join(os.path.dirname(os.path.abspath(db_path)), cand)

    if not source_img_path:
        imgs = get_person_images(db_path, person_label)
        for img in imgs:
            c_path = img["file_path"] if isinstance(img, dict) or hasattr(img, '__getitem__') else None
            if c_path and os.path.isfile(c_path):
                source_img_path = c_path
                break
            elif c_path and db_path and os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(db_path)), c_path)):
                source_img_path = os.path.join(os.path.dirname(os.path.abspath(db_path)), c_path)
                break

    if not source_img_path:
        # Busca abrangente em search_images
        search_res = search_images(db_path, person_label=person_label, limit=10)
        for s_img in search_res:
            c_path = s_img.get("file_path")
            if c_path and os.path.isfile(c_path):
                source_img_path = c_path
                break

    if not source_img_path:
        logger.warning(f"Não foi possível encontrar arquivo de imagem para gerar recorte de '{person_label}'.")
        return None

    bbox = None
    if person and person.get("face_bbox"):
        bbox = person["face_bbox"]
    elif use_ollama_locator:
        bbox = detect_face_bbox_with_ollama(
            image_path=source_img_path,
            person_description=person.get("description") if person else None
        )

    crop_path = crop_face(
        image_path=source_img_path,
        bbox=bbox,
        output_path=output_crop_path,
        fallback_to_center_portrait=True
    )

    if crop_path:
        bbox_json = json.dumps(bbox) if bbox and not isinstance(bbox, str) else (bbox if isinstance(bbox, str) else None)
        update_person_face_crop(db_path, person_label, crop_path, face_bbox=bbox_json)

    return crop_path


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
