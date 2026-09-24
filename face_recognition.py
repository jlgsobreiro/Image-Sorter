"""Comparação facial local conservadora; nunca usa descrições como identidade."""
import hashlib
import logging
import math
import threading
from pathlib import Path

from PIL import Image

from face_cropper import YUNET_MODEL_PATH

logger = logging.getLogger("ImageSorter.FaceRecognition")
SFACE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_recognition_sface_2021dec.onnx"
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


class FaceIdentityMatcher:
    """Carrega modelos no primeiro match; cache contém vetores, nunca rótulos.

    `new` significa ausência de referência visual acima do limiar, não prova de
    identidade inédita. Recortes sem exatamente um rosto utilizável retornam
    `low_quality`. Instâncias serializam acesso aos modelos OpenCV mutáveis.
    """

    def __init__(self, threshold=0.5, margin=0.1):
        if not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("threshold deve estar entre -1 e 1.")
        if not math.isfinite(margin) or not 0 <= margin <= 2:
            raise ValueError("margin deve estar entre 0 e 2.")
        self.threshold = float(threshold)
        self.margin = float(margin)
        self._detector = None
        self._recognizer = None
        self._cache = {}
        self._lock = threading.RLock()

    def _load_models(self):
        if self._recognizer is not None:
            return
        try:
            import cv2
            import numpy as np

            for name, path, expected in (
                ("YuNet", YUNET_MODEL_PATH, YUNET_SHA256),
                ("SFace", SFACE_MODEL_PATH, SFACE_SHA256),
            ):
                try:
                    with Path(path).open("rb") as source:
                        digest = hashlib.file_digest(source, "sha256").hexdigest()
                except OSError as exc:
                    raise RuntimeError(f"Modelo {name} ausente ou ilegível: {path}") from exc
                if digest != expected:
                    raise RuntimeError(f"Modelo {name} corrompido (SHA-256 inválido): {path}")
            detector = cv2.FaceDetectorYN.create(
                str(YUNET_MODEL_PATH), "", (320, 320), 0.85, 0.3, 5000
            )
            recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL_PATH), "")
            if detector is None or recognizer is None:
                raise RuntimeError("OpenCV não criou os modelos YuNet/SFace.")
        except Exception as exc:
            raise RuntimeError(f"Reconhecimento facial local indisponível: {exc}") from exc
        self._cv2 = cv2
        self._np = np
        self._detector = detector
        self._recognizer = recognizer

    def _embedding(self, crop):
        if not isinstance(crop, Image.Image) or min(crop.size) < 40:
            return None
        cv2, np = self._cv2, self._np
        try:
            bgr = cv2.cvtColor(np.asarray(crop.convert("RGB")), cv2.COLOR_RGB2BGR)
        except (OSError, ValueError):
            return None
        height, width = bgr.shape[:2]
        scale = min(1.0, 1280 / max(height, width))
        if scale < 1:
            bgr = cv2.resize(bgr, (round(width * scale), round(height * scale)))
        height, width = bgr.shape[:2]
        if min(height, width) < 32:
            return None
        try:
            self._detector.setInputSize((width, height))
            _, faces = self._detector.detect(bgr)
            if faces is None or len(faces) != 1:
                return None
            face = np.asarray(faces[0], dtype=np.float32)
            if face.shape != (15,) or not np.isfinite(face).all():
                return None
            x, y, w, h = face[:4]
            if face[14] < 0.85 or min(w, h) < 40 * scale:
                return None
            if min(x + w, width) - max(x, 0) < 40 * scale:
                return None
            if min(y + h, height) - max(y, 0) < 40 * scale:
                return None
            landmarks = face[4:14].reshape(5, 2)
            if (landmarks[:, 0] < max(0, x)).any() or (landmarks[:, 0] >= min(width, x + w)).any():
                return None
            if (landmarks[:, 1] < max(0, y)).any() or (landmarks[:, 1] >= min(height, y + h)).any():
                return None
            eye_vector = landmarks[1] - landmarks[0]
            mouth_vector = landmarks[4] - landmarks[3]
            if np.linalg.norm(eye_vector) < 0.1 * w or np.linalg.norm(mouth_vector) < 0.1 * w:
                return None
            eyes = landmarks[:2].mean(axis=0)
            mouth = landmarks[3:].mean(axis=0)
            if mouth[1] - eyes[1] < 0.1 * h or not eyes[1] < landmarks[2, 1] < mouth[1]:
                return None
            aligned = self._recognizer.alignCrop(bgr, face)
            feature = np.asarray(self._recognizer.feature(aligned), dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise RuntimeError(f"Falha na inferência local YuNet/SFace: {exc}") from exc
        if feature.size != 128 or not np.isfinite(feature).all():
            return None
        norm = np.linalg.norm(feature)
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        return feature / norm

    def _reference_embedding(self, value):
        if not isinstance(value, (str, Path)) or not str(value).strip():
            return None
        try:
            path = Path(value).resolve()
        except (OSError, ValueError):
            logger.warning("Referência facial com caminho inválido: %r", value)
            return None
        try:
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        except (OSError, ValueError):
            signature = None
        cached = self._cache.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        embedding = None
        if signature is not None:
            try:
                with Image.open(path) as source:
                    embedding = self._embedding(source)
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                logger.warning("Referência facial ilegível %s: %s", path, exc)
        if embedding is None:
            logger.warning("Referência facial ignorada (ausente, ilegível ou sem um rosto bom): %s", path)
        self._cache[path] = (signature, embedding)
        return embedding

    def match(self, crop: Image.Image, known_people: list[dict], excluded_labels=()) -> dict:
        """Compara apenas face_crop_path atuais; nunca atualiza referências."""
        with self._lock:
            self._load_models()
            embedding = self._embedding(crop)
            if embedding is None:
                return {"person_label": None, "status": "low_quality", "score": None}
            scores = {}
            for person in known_people:
                if not isinstance(person, dict):
                    continue
                label = person.get("person_label")
                if not isinstance(label, str) or not label.strip():
                    continue
                reference = self._reference_embedding(person.get("face_crop_path"))
                if reference is None:
                    continue
                score = float(self._np.clip(self._np.dot(embedding, reference), -1.0, 1.0))
                scores[label] = max(score, scores.get(label, -1.0))
            if not scores:
                return {"person_label": None, "status": "new", "score": None}
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            label, score = ranked[0]
            if label in excluded_labels:
                status = "ambiguous"
            elif score < self.threshold:
                status = "new"
            elif len(ranked) > 1 and (score == ranked[1][1] or score - ranked[1][1] < self.margin):
                status = "ambiguous"
            else:
                status = "matched"
            return {"person_label": label if status == "matched" else None,
                    "status": status, "score": score}