"""
Módulo de banco de dados SQLite para o ImageSorter.
Responsável pela inicialização da estrutura de tabelas e operações comuns.
"""
import sqlite3
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from contextlib import contextmanager

logger = logging.getLogger("ImageSorter.DB")

DEFAULT_DB_PATH = "images.db"


@contextmanager
def get_connection(db_path: str = DEFAULT_DB_PATH):
    """
    Context manager que cria conexão SQLite, gerencia transação
    e garante o fechamento da conexão (essencial no Windows).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Inicializa o esquema do banco de dados caso as tabelas ainda não existam,
    ou aplica migrações incrementais necessárias para tabelas existentes.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                file_size INTEGER,
                status TEXT DEFAULT 'pending',  -- 'pending', 'processed', 'error', 'skipped'
                description TEXT,
                people_present TEXT,            -- JSON array ou texto listando pessoas identificadas (ex: '["Pessoa 1", "Pessoa 2"]')
                model_used TEXT,
                error_message TEXT,
                image_hash TEXT,                -- Hash SHA-256 ou perceptual para detecção de duplicatas
                is_duplicate INTEGER DEFAULT 0, -- 0 = única/principal, 1 = duplicata confirmada
                image_type TEXT DEFAULT 'unclassified', -- 'photo', 'screenshot', 'icon_or_graphic', 'other', 'unclassified'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP
            );
        """)

        # Migração segura para bancos existentes sem colunas adicionais
        cursor.execute("PRAGMA table_info(images);")
        columns = [row["name"] for row in cursor.fetchall()]
        if "people_present" not in columns:
            cursor.execute("ALTER TABLE images ADD COLUMN people_present TEXT;")
            logger.info("Coluna 'people_present' adicionada à tabela 'images' com sucesso.")
        if "image_hash" not in columns:
            cursor.execute("ALTER TABLE images ADD COLUMN image_hash TEXT;")
            logger.info("Coluna 'image_hash' adicionada à tabela 'images' com sucesso.")
        if "is_duplicate" not in columns:
            cursor.execute("ALTER TABLE images ADD COLUMN is_duplicate INTEGER DEFAULT 0;")
            logger.info("Coluna 'is_duplicate' adicionada à tabela 'images' com sucesso.")
        if "image_type" not in columns:
            cursor.execute("ALTER TABLE images ADD COLUMN image_type TEXT DEFAULT 'unclassified';")
            logger.info("Coluna 'image_type' adicionada à tabela 'images' com sucesso.")

        # Tabela de catálogo de pessoas conhecidas para consistência entre imagens
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS known_people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_label TEXT UNIQUE NOT NULL, -- ex: 'Pessoa 1'
                description TEXT NOT NULL,         -- Características físicas duradouras (rosto, idade, cabelo, etc.)
                first_seen_image_id INTEGER,
                face_crop_path TEXT,               -- Caminho do recorte contendo apenas o rosto
                face_bbox TEXT,                    -- Coordenadas do rosto JSON [ymin, xmin, ymax, xmax]
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (first_seen_image_id) REFERENCES images(id)
            );
        """)

        # Migrações seguras para tabelas existentes
        cursor.execute("PRAGMA table_info(known_people);")
        kp_columns = [row["name"] for row in cursor.fetchall()]
        if "face_crop_path" not in kp_columns:
            cursor.execute("ALTER TABLE known_people ADD COLUMN face_crop_path TEXT;")
            logger.info("Coluna 'face_crop_path' adicionada à tabela 'known_people'.")
        if "face_bbox" not in kp_columns:
            cursor.execute("ALTER TABLE known_people ADD COLUMN face_bbox TEXT;")
            logger.info("Coluna 'face_bbox' adicionada à tabela 'known_people'.")

        # Índices para acelerar consultas por status, caminhos e hashes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_file_path ON images(file_path);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_hash ON images(image_hash);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_type ON images(image_type);
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_known_people_label ON known_people(person_label);
        """)
        conn.commit()
    logger.debug(f"Banco de dados inicializado/verificado em: {db_path}")


def insert_images_batch(db_path: str, records: List[Dict[str, Any]]) -> int:
    """
    Insere uma lista de imagens no banco de dados em lote.
    Ignora caminhos duplicados.
    Retorna a quantidade de novas linhas inseridas.
    """
    if not records:
        return 0

    sanitized_records = []
    for r in records:
        rec = dict(r)
        if "file_size" not in rec or rec["file_size"] is None:
            rec["file_size"] = 0
        sanitized_records.append(rec)

    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        before_count = cursor.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        cursor.executemany("""
            INSERT OR IGNORE INTO images (file_path, file_name, file_size, status)
            VALUES (:file_path, :file_name, :file_size, 'pending')
        """, sanitized_records)
        conn.commit()
        after_count = cursor.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        inserted = after_count - before_count
        logger.info(f"Lote de {len(records)} registro(s) processado no banco ({inserted} novo(s) inserido(s)).")
        return inserted


def get_pending_images(db_path: str, limit: Optional[int] = None) -> List[sqlite3.Row]:
    """Retorna imagens que ainda estão com status 'pending'."""
    init_db(db_path)
    query = "SELECT * FROM images WHERE status = 'pending' ORDER BY id ASC"
    if limit is not None and limit > 0:
        query += f" LIMIT {int(limit)}"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        return cursor.execute(query).fetchall()


def update_image_description(
    db_path: str,
    image_id: int,
    description: str,
    model_used: str,
    people_present: Optional[str] = None,
    status: str = 'processed'
) -> None:
    """Atualiza a descrição, a lista de pessoas presentes e o status de uma imagem processada."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE images
            SET description = ?,
                people_present = ?,
                model_used = ?,
                status = ?,
                error_message = NULL,
                processed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (description, people_present, model_used, status, image_id))
        conn.commit()


def update_image_error(
    db_path: str,
    image_id: int,
    error_message: str,
    status: str = 'error'
) -> None:
    """Registra falha no processamento de uma imagem."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE images
            SET status = ?,
                error_message = ?,
                processed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status, error_message, image_id))
        conn.commit()


def get_known_people(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retorna todos os indivíduos já registrados no catálogo com suas descrições e identificadores."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT kp.id, kp.person_label, kp.description, kp.first_seen_image_id, 
                   kp.face_crop_path, kp.face_bbox, kp.created_at,
                   img.file_path AS first_seen_file_path, img.file_name AS first_seen_file_name
            FROM known_people kp
            LEFT JOIN images img ON kp.first_seen_image_id = img.id
            ORDER BY kp.id ASC
        """)
        return [dict(row) for row in cursor.fetchall()]


def register_known_person(
    db_path: str,
    person_label: str,
    description: str,
    first_seen_image_id: Optional[int] = None,
    face_crop_path: Optional[str] = None,
    face_bbox: Optional[str] = None
) -> bool:
    """
    Cadastra uma nova pessoa no catálogo caso o person_label ainda não exista.
    Retorna True se foi inserida, False se já existia.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO known_people (person_label, description, first_seen_image_id, face_crop_path, face_bbox)
                VALUES (?, ?, ?, ?, ?)
            """, (person_label.strip(), description.strip(), first_seen_image_id, face_crop_path, face_bbox))
            conn.commit()
            logger.info(f"Nova pessoa catalogada no banco de dados: '{person_label}' (ID Imagem: {first_seen_image_id})")
            return True
        except sqlite3.IntegrityError:
            # Já existe esse rótulo
            return False


def update_person_face_crop(
    db_path: str,
    person_label: str,
    face_crop_path: str,
    face_bbox: Optional[str] = None
) -> bool:
    """Atualiza o caminho do recorte de rosto e coordenadas para uma pessoa cadastrada."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE known_people
            SET face_crop_path = ?,
                face_bbox = COALESCE(?, face_bbox)
            WHERE person_label = ?
        """, (face_crop_path, face_bbox, person_label.strip()))
        conn.commit()
        return cursor.rowcount > 0


def rename_or_merge_known_person(
    db_path: str,
    old_label: str,
    new_label: str,
    new_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Renomeia uma pessoa no catálogo OU unifica (merge) se 'new_label' já existir.
    Atualiza todas as referências em 'images.people_present' de forma consistente.

    :param db_path: Caminho do banco SQLite
    :param old_label: Rótulo atual (ex: 'Pessoa 2')
    :param new_label: Novo rótulo/nome desejado (ex: 'João' ou 'Pessoa 1')
    :param new_description: Descrição atualizada opcional
    :return: Dicionário com o resumo da operação
    """
    init_db(db_path)
    old_clean = old_label.strip()
    new_clean = new_label.strip()

    if not old_clean or not new_clean or old_clean == new_clean:
        return {"action": "none", "affected_images": 0, "old_label": old_clean, "new_label": new_clean}

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Verifica se o novo rótulo já existe
        target_row = cursor.execute(
            "SELECT * FROM known_people WHERE person_label = ?", (new_clean,)
        ).fetchone()
        source_row = cursor.execute(
            "SELECT * FROM known_people WHERE person_label = ?", (old_clean,)
        ).fetchone()

        is_merge = target_row is not None
        affected_images_count = 0

        # Atualiza a lista de pessoas presentes em todas as imagens afetadas
        cursor.execute("SELECT id, people_present FROM images WHERE people_present IS NOT NULL")
        all_images = cursor.fetchall()

        for img in all_images:
            raw_pp = img["people_present"]
            if not raw_pp:
                continue
            try:
                people_list = json.loads(raw_pp)
                if not isinstance(people_list, list):
                    continue

                if old_clean in people_list:
                    # Substitui old_clean por new_clean e elimina duplicatas mantendo a ordem
                    updated_list = []
                    for item in people_list:
                        repl = new_clean if item == old_clean else item
                        if repl not in updated_list:
                            updated_list.append(repl)

                    cursor.execute(
                        "UPDATE images SET people_present = ? WHERE id = ?",
                        (json.dumps(updated_list, ensure_ascii=False), img["id"])
                    )
                    affected_images_count += 1
            except Exception as e:
                logger.debug(f"Erro ao processar people_present da imagem {img['id']}: {e}")

        if is_merge:
            # Unificação: remove a entrada antiga do catálogo de pessoas
            cursor.execute("DELETE FROM known_people WHERE person_label = ?", (old_clean,))
            if new_description:
                cursor.execute(
                    "UPDATE known_people SET description = ? WHERE person_label = ?",
                    (new_description.strip(), new_clean)
                )
            action = "merged"
            logger.info(f"Pessoa '{old_clean}' unificada em '{new_clean}' com sucesso ({affected_images_count} fotos atualizadas).")
        else:
            # Renomeação simples
            if source_row:
                desc = new_description.strip() if new_description else source_row["description"]
                cursor.execute("""
                    UPDATE known_people
                    SET person_label = ?,
                        description = ?
                    WHERE person_label = ?
                """, (new_clean, desc, old_clean))
            else:
                # Caso a pessoa estivesse nas imagens mas ainda não no catálogo
                cursor.execute("""
                    INSERT INTO known_people (person_label, description)
                    VALUES (?, ?)
                """, (new_clean, new_description or f"Perfil de {new_clean}."))
            action = "renamed"
            logger.info(f"Pessoa '{old_clean}' renomeada para '{new_clean}' ({affected_images_count} fotos atualizadas).")

        conn.commit()
        return {
            "action": action,
            "affected_images": affected_images_count,
            "old_label": old_clean,
            "new_label": new_clean
        }


def get_person_images(db_path: str, person_label: str) -> List[sqlite3.Row]:
    """Retorna todas as imagens cadastradas em que a pessoa especificada aparece."""
    init_db(db_path)
    clean_label = person_label.strip()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images WHERE people_present IS NOT NULL ORDER BY id ASC")
        rows = cursor.fetchall()
        matching = []
        for r in rows:
            try:
                people_list = json.loads(r["people_present"] or "[]")
                if clean_label in people_list:
                    matching.append(r)
            except Exception:
                if clean_label in (r["people_present"] or ""):
                    matching.append(r)
        return matching


def search_images(
    db_path: str,
    person_label: Optional[str] = None,
    status: Optional[str] = None,
    query_text: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Busca imagens no banco por pessoa presente, status ou texto na descrição.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM images WHERE 1=1"
        params: List[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if query_text:
            query += " AND (description LIKE ? OR file_name LIKE ?)"
            params.append(f"%{query_text}%")
            params.append(f"%{query_text}%")

        query += " ORDER BY id ASC"
        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for r in rows:
            row_dict = dict(r)
            if person_label:
                clean_person = person_label.strip()
                try:
                    p_list = json.loads(row_dict["people_present"] or "[]")
                    if clean_person not in p_list:
                        continue
                except Exception:
                    if clean_person not in (row_dict["people_present"] or ""):
                        continue
            results.append(row_dict)
            if len(results) >= limit:
                break

        return results


def get_people_summary(db_path: str) -> List[Dict[str, Any]]:
    """
    Retorna o resumo de todas as pessoas catalogadas com o total de fotos onde cada uma aparece.
    """
    init_db(db_path)
    known = get_known_people(db_path)
    summary = []
    for p in known:
        label = p["person_label"]
        imgs = get_person_images(db_path, label)
        summary.append({
            **p,
            "photo_count": len(imgs)
        })
    return summary


def get_image_by_id(db_path: str, image_id: int) -> Optional[Dict[str, Any]]:
    """Retorna os dados de uma imagem pelo ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        row = cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        return dict(row) if row else None


def update_image_people_present(
    db_path: str,
    image_id: int,
    people_present: Union[List[str], str]
) -> bool:
    """
    Atualiza a lista de pessoas presentes em uma imagem específica.
    """
    init_db(db_path)
    if isinstance(people_present, list):
        people_json = json.dumps(people_present, ensure_ascii=False)
    else:
        people_json = people_present

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE images
            SET people_present = ?
            WHERE id = ?
        """, (people_json, image_id))
        conn.commit()
        return cursor.rowcount > 0


def get_all_images(
    db_path: str,
    status: Optional[str] = None,
    person_label: Optional[str] = None,
    query_text: Optional[str] = None,
    image_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retorna todas as imagens cadastradas no banco de dados com filtros opcionais.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM images WHERE 1=1"
        params: List[Any] = []

        if status and status != "all":
            query += " AND status = ?"
            params.append(status)

        if image_type and image_type != "all":
            query += " AND image_type = ?"
            params.append(image_type)

        if query_text:
            query += " AND (description LIKE ? OR file_name LIKE ? OR file_path LIKE ?)"
            params.append(f"%{query_text}%")
            params.append(f"%{query_text}%")
            params.append(f"%{query_text}%")

        query += " ORDER BY id ASC"
        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for r in rows:
            row_dict = dict(r)
            if person_label:
                clean_person = person_label.strip()
                try:
                    p_list = json.loads(row_dict.get("people_present") or "[]")
                    if clean_person not in p_list:
                        continue
                except Exception:
                    if clean_person not in (row_dict.get("people_present") or ""):
                        continue
            results.append(row_dict)

        return results


def get_statistics(db_path: str) -> Dict[str, int]:
    """Retorna estatísticas gerais do banco de dados."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        total = cursor.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        processed = cursor.execute("SELECT COUNT(*) FROM images WHERE status = 'processed'").fetchone()[0]
        pending = cursor.execute("SELECT COUNT(*) FROM images WHERE status = 'pending'").fetchone()[0]
        errors = cursor.execute("SELECT COUNT(*) FROM images WHERE status = 'error'").fetchone()[0]
        known_people_count = cursor.execute("SELECT COUNT(*) FROM known_people").fetchone()[0]
        
        # Estatísticas de duplicatas
        hashed_count = cursor.execute("SELECT COUNT(*) FROM images WHERE image_hash IS NOT NULL AND image_hash != ''").fetchone()[0]
        dup_groups = cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT image_hash FROM images 
                WHERE image_hash IS NOT NULL AND image_hash != '' 
                GROUP BY image_hash HAVING COUNT(*) > 1
            )
        """).fetchone()[0]
        dup_images_count = cursor.execute("""
            SELECT COUNT(*) FROM images WHERE image_hash IN (
                SELECT image_hash FROM images 
                WHERE image_hash IS NOT NULL AND image_hash != '' 
                GROUP BY image_hash HAVING COUNT(*) > 1
            )
        """).fetchone()[0]

        # Estatísticas de tipos de imagem (Fotos Reais, Prints, Ícones)
        photos_count = cursor.execute("SELECT COUNT(*) FROM images WHERE image_type = 'photo'").fetchone()[0]
        screenshots_count = cursor.execute("SELECT COUNT(*) FROM images WHERE image_type = 'screenshot'").fetchone()[0]
        icons_count = cursor.execute("SELECT COUNT(*) FROM images WHERE image_type = 'icon_or_graphic'").fetchone()[0]
        other_types_count = cursor.execute("SELECT COUNT(*) FROM images WHERE image_type = 'other'").fetchone()[0]
        unclassified_count = cursor.execute("""
            SELECT COUNT(*) FROM images 
            WHERE image_type IS NULL OR image_type = '' OR image_type = 'unclassified'
        """).fetchone()[0]

        return {
            "total": total,
            "processed": processed,
            "pending": pending,
            "errors": errors,
            "known_people": known_people_count,
            "hashed_images": hashed_count,
            "duplicate_groups": dup_groups,
            "duplicate_images_total": dup_images_count,
            "photos_count": photos_count,
            "screenshots_count": screenshots_count,
            "icons_count": icons_count,
            "other_types_count": other_types_count,
            "unclassified_count": unclassified_count
        }


def update_image_hash(db_path: str, image_id: int, image_hash: str) -> None:
    """Atualiza o hash de conteúdo de uma imagem."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE images SET image_hash = ? WHERE id = ?", (image_hash, image_id))
        conn.commit()


def update_image_duplicate_status(db_path: str, image_id: int, is_duplicate: int) -> None:
    """Atualiza o status de duplicata de uma imagem (0 = original/única, 1 = duplicata)."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE images SET is_duplicate = ? WHERE id = ?", (is_duplicate, image_id))
        conn.commit()


def get_images_for_hashing(db_path: str, force: bool = False) -> List[sqlite3.Row]:
    """Retorna imagens que necessitam de cálculo de hash."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if force:
            return cursor.execute("SELECT * FROM images ORDER BY id ASC").fetchall()
        else:
            return cursor.execute("SELECT * FROM images WHERE image_hash IS NULL OR image_hash = '' ORDER BY id ASC").fetchall()


def get_duplicate_groups(db_path: str) -> List[Dict[str, Any]]:
    """
    Retorna todos os grupos de imagens que compartilham o mesmo hash (duplicadas).
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT image_hash, COUNT(*) as count
            FROM images
            WHERE image_hash IS NOT NULL AND image_hash != ''
            GROUP BY image_hash
            HAVING count > 1
            ORDER BY count DESC, id ASC
        """)
        groups_meta = cursor.fetchall()

        duplicate_groups = []
        for g in groups_meta:
            h = g["image_hash"]
            cursor.execute("SELECT * FROM images WHERE image_hash = ? ORDER BY is_duplicate ASC, id ASC", (h,))
            img_rows = cursor.fetchall()
            duplicate_groups.append({
                "hash": h,
                "count": g["count"],
                "images": [dict(r) for r in img_rows]
            })

        return duplicate_groups


def get_unique_images(db_path: str) -> List[Dict[str, Any]]:
    """
    Retorna a lista de imagens únicas para exportação ou organização.
    Para grupos de duplicatas, seleciona a imagem principal (is_duplicate = 0 ou primeiro id).
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # 1. Imagens com hash calculado (agrupadas por hash)
        cursor.execute("""
            SELECT * FROM images
            WHERE image_hash IS NOT NULL AND image_hash != ''
            ORDER BY image_hash, is_duplicate ASC, id ASC
        """)
        hashed_images = cursor.fetchall()
        
        unique_images_dict = {}
        for r in hashed_images:
            h = r["image_hash"]
            if h not in unique_images_dict:
                # Seleciona a primeira imagem (priorizando is_duplicate == 0)
                unique_images_dict[h] = dict(r)
            else:
                # Se a atual for explicitamente marcada como is_duplicate == 0 e a salva não for, substitui
                if r["is_duplicate"] == 0 and unique_images_dict[h]["is_duplicate"] != 0:
                    unique_images_dict[h] = dict(r)

        unique_list = list(unique_images_dict.values())

        # 2. Imagens sem hash calculado que não foram marcadas como duplicatas
        cursor.execute("""
            SELECT * FROM images
            WHERE (image_hash IS NULL OR image_hash = '') AND is_duplicate = 0
            ORDER BY id ASC
        """)
        unhashed = cursor.fetchall()
        for r in unhashed:
            unique_list.append(dict(r))

        return unique_list


def delete_image_by_id(db_path: str, image_id: int) -> bool:
    """Remove o registro de uma imagem do banco de dados."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
        conn.commit()
        return cursor.rowcount > 0


def update_image_type(db_path: str, image_id: int, image_type: str) -> bool:
    """
    Atualiza o tipo classificado da imagem (ex: 'photo', 'screenshot', 'icon_or_graphic', 'other').
    """
    init_db(db_path)
    clean_type = (image_type or "unclassified").strip().lower()
    valid_types = {"photo", "screenshot", "icon_or_graphic", "other", "unclassified"}
    if clean_type not in valid_types:
        # Mapeamentos tolerantes
        if "photo" in clean_type or "foto" in clean_type:
            clean_type = "photo"
        elif "screen" in clean_type or "print" in clean_type or "captura" in clean_type:
            clean_type = "screenshot"
        elif "icon" in clean_type or "ícone" in clean_type or "icone" in clean_type or "graphic" in clean_type or "logo" in clean_type or "asset" in clean_type:
            clean_type = "icon_or_graphic"
        else:
            clean_type = "other"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE images SET image_type = ? WHERE id = ?", (clean_type, image_id))
        conn.commit()
        return cursor.rowcount > 0


def get_images_for_classification(
    db_path: str,
    force: bool = False,
    limit: Optional[int] = None
) -> List[sqlite3.Row]:
    """
    Retorna as imagens elegíveis para classificação de tipo (fotos vs prints vs ícones).
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if force:
            query = "SELECT * FROM images ORDER BY id ASC"
            params: List[Any] = []
        else:
            query = """
                SELECT * FROM images 
                WHERE image_type IS NULL OR image_type = '' OR image_type = 'unclassified'
                ORDER BY id ASC
            """
            params = []

        if limit and limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        return cursor.execute(query, params).fetchall()


def get_image_type_statistics(db_path: str) -> Dict[str, int]:
    """Retorna a contagem agrupada por tipos de imagem cadastrados."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        photos = cursor.execute("SELECT COUNT(*) FROM images WHERE image_type = 'photo'").fetchone()[0]
        screenshots = cursor.execute("SELECT COUNT(*) FROM images WHERE image_type = 'screenshot'").fetchone()[0]
        icons = cursor.execute("SELECT COUNT(*) FROM images WHERE image_type = 'icon_or_graphic'").fetchone()[0]
        others = cursor.execute("SELECT COUNT(*) FROM images WHERE image_type = 'other'").fetchone()[0]
        unclassified = cursor.execute("""
            SELECT COUNT(*) FROM images 
            WHERE image_type IS NULL OR image_type = '' OR image_type = 'unclassified'
        """).fetchone()[0]

        return {
            "photo": photos,
            "screenshot": screenshots,
            "icon_or_graphic": icons,
            "other": others,
            "unclassified": unclassified,
            "total": photos + screenshots + icons + others + unclassified
        }


def get_real_photos(db_path: str, only_unique: bool = True) -> List[Dict[str, Any]]:
    """
    Retorna a lista de imagens classificadas como fotos reais ('photo').
    Se only_unique=True, descarta duplicatas confirmadas ou redundantes.
    """
    init_db(db_path)
    if only_unique:
        all_unique = get_unique_images(db_path)
        return [img for img in all_unique if img.get("image_type") == "photo"]
    else:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            rows = cursor.execute("""
                SELECT * FROM images 
                WHERE image_type = 'photo'
                ORDER BY id ASC
            """).fetchall()
            return [dict(r) for r in rows]
