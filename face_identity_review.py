"""Localiza pares de pessoas catalogadas com similaridade facial para revisão manual."""
import json
import logging
import math
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger("ImageSorter.FaceIdentityReview")

try:
    from db import (
        DEFAULT_DB_PATH,
        get_connection,
        get_known_people,
        rename_or_merge_known_person,
    )
    from face_recognition import FaceIdentityMatcher
except ImportError:
    from .db import (
        DEFAULT_DB_PATH,
        get_connection,
        get_known_people,
        rename_or_merge_known_person,
    )
    from .face_recognition import FaceIdentityMatcher


DEFAULT_MIN_SIMILARITY = 0.35


def get_people_cooccurrence_and_counts(
    db_path: str = DEFAULT_DB_PATH,
) -> Tuple[Set[frozenset], Dict[str, int]]:
    """Calcula pares de pessoas que aparecem na mesma foto e contagem de fotos por pessoa.

    Pessoas que aparecem simultaneamente no mesmo enquadramento geralmente são
    indivíduos diferentes, servindo como forte sinal contra fusão acidental.
    """
    cooccurring: Set[frozenset] = set()
    counts: Dict[str, int] = {}

    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT people_present FROM images WHERE people_present IS NOT NULL"
            )
            rows = cursor.fetchall()

            for row in rows:
                raw = row["people_present"]
                if not raw:
                    continue
                try:
                    present = json.loads(raw)
                    if isinstance(present, list):
                        labels = [
                            p.strip()
                            for p in present
                            if isinstance(p, str) and p.strip()
                        ]
                        # Remove duplicados na mesma imagem
                        unique_labels = sorted(set(labels))
                        for label in unique_labels:
                            counts[label] = counts.get(label, 0) + 1

                        if len(unique_labels) >= 2:
                            for i, l1 in enumerate(unique_labels):
                                for l2 in unique_labels[i + 1 :]:
                                    cooccurring.add(frozenset((l1, l2)))
                except Exception as exc:
                    logger.debug("Falha ao analisar people_present para co-ocorrência: %s", exc)
    except Exception as exc:
        logger.debug("Falha ao consultar co-ocorrência no banco de dados: %s", exc)

    return cooccurring, counts


def find_similar_people(
    db_path: str = DEFAULT_DB_PATH,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    filter_cooccurring: bool = False,
) -> List[Dict[str, Any]]:
    """Retorna pares acima do limiar, ordenados do mais para o menos similar.

    A similaridade cosseno serve apenas para priorizar uma revisão humana; não é
    uma probabilidade de identidade e nunca mescla registros automaticamente.
    """
    if not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
        raise ValueError("min_similarity deve estar entre -1 e 1.")

    people = get_known_people(db_path)
    people = [person for person in people if person.get("face_crop_path")]
    if len(people) < 2:
        return []

    cooccurring, photo_counts = get_people_cooccurrence_and_counts(db_path)
    matcher = FaceIdentityMatcher()
    candidates = []

    for index, first in enumerate(people):
        first_label = first.get("person_label", "")
        for second in people[index + 1 :]:
            second_label = second.get("person_label", "")
            score = matcher.compare_people(first, second)
            if score is None or score < min_similarity:
                continue

            pair_key = frozenset((first_label, second_label))
            co_occurs = pair_key in cooccurring
            if filter_cooccurring and co_occurs:
                continue

            if score >= 0.65:
                confidence = "Alta"
                confidence_tag = "🟢 Alta"
            elif score >= 0.45:
                confidence = "Média"
                confidence_tag = "🟡 Média"
            else:
                confidence = "Possível"
                confidence_tag = "⚪ Possível"

            candidates.append(
                {
                    "first": first,
                    "second": second,
                    "score": score,
                    "confidence": confidence,
                    "confidence_tag": confidence_tag,
                    "co_occurs": co_occurs,
                    "first_photo_count": photo_counts.get(first_label, 0),
                    "second_photo_count": photo_counts.get(second_label, 0),
                }
            )

    return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)


def run_cli_similar_people_review(
    db_path: str = DEFAULT_DB_PATH,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> None:
    """Interface de terminal para revisar e unificar pessoas com rostos semelhantes."""
    print("\n" + "=" * 65)
    print("REVISAO FACIAL INTERATIVA DE PESSOAS SEMELHANTES")
    print("=" * 65)
    print(f"Limiar de similaridade biometrica: >= {min_similarity:.2f}")
    print("Calculando correspondencias nos recortes faciais locais...")

    candidates = find_similar_people(db_path, min_similarity=min_similarity)
    if not candidates:
        print("\n[OK] Nenhum par de pessoas com similaridade facial acima do limiar foi encontrado.")
        input("\nPressione Enter para voltar ao menu...")
        return

    print(f"\nForam encontrados {len(candidates)} par(es) para verificacao manual.")

    idx = 0
    while idx < len(candidates):
        item = candidates[idx]
        p_a = item["first"]
        p_b = item["second"]
        score = item["score"]
        conf = item["confidence"]
        cooccur = item["co_occurs"]
        count_a = item["first_photo_count"]
        count_b = item["second_photo_count"]

        print("\n" + "-" * 65)
        print(f"PAR {idx + 1} de {len(candidates)}:")
        print(f" * Pessoa A: '{p_a['person_label']}' ({count_a} foto(s)) - Descricao: {p_a.get('description', '')[:60]}")
        print(f" * Pessoa B: '{p_b['person_label']}' ({count_b} foto(s)) - Descricao: {p_b.get('description', '')[:60]}")
        print(f" * Similaridade: {score:.3f} (Confianca: {conf})")
        if cooccur:
            print(" [!] AVISO: Ambas aparecem juntas na mesma foto! Provavelmente sao pessoas diferentes.")

        print("\nOpcoes:")
        print(f"  [1] Mesclar '{p_b['person_label']}' em '{p_a['person_label']}' (Manter '{p_a['person_label']}')")
        print(f"  [2] Mesclar '{p_a['person_label']}' em '{p_b['person_label']}' (Manter '{p_b['person_label']}')")
        print("  [3] Pular este par (nao sao a mesma pessoa)")
        print("  [0] Encerrar revisao e voltar")

        choice = input("\nEscolha uma opcao (1/2/3/0): ").strip()
        if choice == "0":
            break
        elif choice == "1":
            confirm = input(f"Confirma mesclar '{p_b['person_label']}' em '{p_a['person_label']}'? (s/n): ").strip().lower()
            if confirm == "s":
                res = rename_or_merge_known_person(db_path, p_b["person_label"], p_a["person_label"])
                print(f"[OK] Sucesso: {res['affected_images']} imagem(ns) atualizada(s).")
                # Recalcula candidatos após merge
                candidates = find_similar_people(db_path, min_similarity=min_similarity)
                continue
        elif choice == "2":
            confirm = input(f"Confirma mesclar '{p_a['person_label']}' em '{p_b['person_label']}'? (s/n): ").strip().lower()
            if confirm == "s":
                res = rename_or_merge_known_person(db_path, p_a["person_label"], p_b["person_label"])
                print(f"[OK] Sucesso: {res['affected_images']} imagem(ns) atualizada(s).")
                # Recalcula candidatos após merge
                candidates = find_similar_people(db_path, min_similarity=min_similarity)
                continue
        elif choice == "3":
            idx += 1
        else:
            print("Opcao invalida.")

    print("\nRevisao concluida.")
    input("\nPressione Enter para continuar...")
