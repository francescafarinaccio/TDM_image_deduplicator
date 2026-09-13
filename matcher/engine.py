"""
Modulo: engine.py
Descrizione: Engine di matching e clustering per l'individuazione di immagini
             duplicate o quasi-duplicate e la selezione del file Master.
Corso: Trattamento Dati Multimediali (TDM)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Set
import numpy as np

from ..classifier.enums import ImageType
from ..extractors.doc_extractor import DocFeatureExtractor
from ..extractors.photo_extractor import PhotoFeatureExtractor


@dataclass
class DuplicateCluster:
    """Rappresenta un gruppo di immagini duplicate identificate."""
    cluster_id: int
    image_type: ImageType
    master_file: str
    duplicate_files: List[str]
    similarity_scores: Dict[str, float]  # Similarità rispetto al file master


class DeduplicationEngine:
    """
    Engine principale per il calcolo delle matrici di similarità
    e la generazione dei cluster di duplicati.
    """

    def __init__(self, doc_threshold: float = 0.70, photo_threshold: float = 0.75):
        """
        :param doc_threshold: Soglia minima di similarità (0..1) per considerare due DOC duplicati.
        :param photo_threshold: Soglia minima di similarità (0..1) per considerare due PHOTO duplicate.
        """
        self.doc_threshold = doc_threshold
        self.photo_threshold = photo_threshold

        self.doc_extractor = DocFeatureExtractor()
        self.photo_extractor = PhotoFeatureExtractor()

    def find_duplicates(self, scanned_records: List[Dict[str, Any]]) -> List[DuplicateCluster]:
        """
        Elabora i record prodotti da scanner.py e regioner.py e restituisce i cluster di duplicati.
        
        :param scanned_records: Lista di dizionari contenenti 'filepath', 'quality_score', 'image_type'.
        :return: Lista di oggetti DuplicateCluster.
        """
        # 1. Separazione dei file per macro-categoria
        doc_records = [r for r in scanned_records if r.get("image_type") == ImageType.DOCUMENT]
        photo_records = [r for r in scanned_records if r.get("image_type") == ImageType.PHOTO]

        clusters: List[DuplicateCluster] = []
        cluster_counter = 1

        # 2. Clustering per Documenti
        doc_clusters, cluster_counter = self._process_category(
            doc_records, self.doc_extractor, self.doc_threshold, ImageType.DOCUMENT, cluster_counter
        )
        clusters.extend(doc_clusters)

        # 3. Clustering per Fotografie
        photo_clusters, cluster_counter = self._process_category(
            photo_records, self.photo_extractor, self.photo_threshold, ImageType.PHOTO, cluster_counter
        )
        clusters.extend(photo_clusters)

        return clusters

    def _process_category(
        self,
        records: List[Dict[str, Any]],
        extractor: Any,
        threshold: float,
        image_type: ImageType,
        start_cluster_id: int
    ) -> Tuple[List[DuplicateCluster], int]:
        """Esegue estrazione feature, calcolo matrice e clustering a componenti connesse."""
        if len(records) < 2:
            return [], start_cluster_id

        # Estrazione feature per tutti i file della categoria
        descriptors = []
        for rec in records:
            desc = extractor.extract(rec["filepath"])
            descriptors.append(desc)

        n = len(records)
        adj_matrix = np.zeros((n, n), dtype=bool)
        sim_matrix = np.zeros((n, n), dtype=np.float32)

        # Calcolo pairwise della matrice di similarità
        for i in range(n):
            adj_matrix[i, i] = True
            sim_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                sim = extractor.compute_similarity(descriptors[i], descriptors[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

                if sim >= threshold:
                    adj_matrix[i, j] = True
                    adj_matrix[j, i] = True

        # Algoritmo Graph-based: ricerca delle componenti connesse
        visited: Set[int] = set()
        clusters: List[DuplicateCluster] = []
        current_cluster_id = start_cluster_id

        for i in range(n):
            if i in visited:
                continue

            # Breadth-First Search (BFS) per trovare tutti gli elementi connessi
            component: List[int] = []
            queue = [i]
            visited.add(i)

            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in range(n):
                    if adj_matrix[curr, neighbor] and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            # Se la componente ha almeno 2 elementi, abbiamo trovato un gruppo di duplicati
            if len(component) > 1:
                group_records = [records[idx] for idx in component]

                # Selezione del MASTER: l'immagine con il punteggio di qualità più alto
                group_records.sort(key=lambda x: x.get("quality_score", 0.0), reverse=True)
                master_rec = group_records[0]
                duplicate_recs = group_records[1:]

                master_idx = records.index(master_rec)
                sim_scores = {}
                for dup in duplicate_recs:
                    dup_idx = records.index(dup)
                    sim_scores[dup["filepath"]] = float(sim_matrix[master_idx, dup_idx])

                clusters.append(
                    DuplicateCluster(
                        cluster_id=current_cluster_id,
                        image_type=image_type,
                        master_file=master_rec["filepath"],
                        duplicate_files=[d["filepath"] for d in duplicate_recs],
                        similarity_scores=sim_scores
                    )
                )
                current_cluster_id += 1

        return clusters, current_cluster_id