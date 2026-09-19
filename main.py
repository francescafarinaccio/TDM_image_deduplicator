"""
Main entrypoint per l'esecuzione della pipeline unificata di deduplicazione.
Corso: Trattamento Dati Multimediali (TDM)
"""

from pathlib import Path

from src.ingestion.scanner import ImageScanner
from src.classifier.regioner import ImageClassifier
from src.matcher.engine import DeduplicationEngine
from src.safe_ops.isolation import IsolationManager


DATASET_DIR = "./dataset"      # Cartella di input contenente le immagini
DUPLICATES_DIR = "./duplicati"  # Cartella target per l'isolamento dei duplicati



def run_pipeline():
    print("=" * 60)
    print(" PIPELINE DI DEDUPLICAZIONE IMMAGINI (TDM)")
    print("=" * 60)

    # 1. Scansione ed estrazione metadati / quality score logaritmico
    print(f"\n[1/4] Scansione directory statica: {DATASET_DIR}")
    scanner = ImageScanner(target_dir=DATASET_DIR)
    scanned_files = scanner.scan()
    print(f" -> Trovate {len(scanned_files)} immagini valide.")

    if len(scanned_files) < 2:
        print(" -> File insufficienti per la deduplicazione. Inserisci almeno 2 immagini in ./dataset")
        return

    # 2. Classificazione JBIG2 Regioning (DOC vs PHOTO)
    print("\n[2/4] Classificazione contenuto (DOC vs PHOTO)...")
    classifier = ImageClassifier()
    processed_records = []

    for item in scanned_files:
        image_type, metrics = classifier.classify(item["filepath"])
        processed_records.append({
            "filepath": item["filepath"],
            "quality_score": item["quality_score"],
            "image_type": image_type,
            "metrics": metrics
        })
        print(f" -> {Path(item['filepath']).name}: {image_type.value} (Quality Score: {item['quality_score']:.2f})")

    # 3. Matching e Clustering (HOG + HSV 3D per foto, ORB + Layout per doc)
    print("\n[3/4] Calcolo similarita' e ricerca duplicati...")
    engine = DeduplicationEngine(doc_threshold=0.70, photo_threshold=0.75)
    clusters = engine.find_duplicates(processed_records)

    print(f" -> Trovati {len(clusters)} cluster di duplicati.")

    # 4. Isolamento nella cartella 'duplicati'
    if clusters:
        print(f"\n[4/4] Spostamento duplicati nella cartella target '{DUPLICATES_DIR}'...")
        iso_manager = IsolationManager(target_folder=DUPLICATES_DIR)
        manifest_path = iso_manager.isolate_duplicates(clusters)
        
        for c in clusters:
            print(f"\n Cluster #{c.cluster_id} [{c.image_type.value}]")
            print(f"   MASTER (conservato): {Path(c.master_file).name}")
            for dup in c.duplicate_files:
                score = c.similarity_scores.get(dup, 0.0)
                print(f"   -> SPOSTATO IN DUPLICATI: {Path(dup).name} (Similarita': {score:.2%})")

        print(f"\n Operazione completata. Manifest salvato in: {manifest_path}")
    else:
        print("\n[4/4] Nessun duplicato trovato da isolare.")


if __name__ == "__main__":
    run_pipeline()