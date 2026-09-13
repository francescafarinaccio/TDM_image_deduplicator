"""
Main entrypoint per l'esecuzione della pipeline completa di deduplicazione.
"""

import argparse
from pathlib import Path

from src.scanner.scanner import ImageScanner
from src.classifier.regioner import ImageClassifier
from src.matcher.engine import DeduplicationEngine
from src.safe_ops.isolation import IsolationManager


def run_pipeline(dataset_dir: str, isolation_dir: str):
    print("=" * 60)
    print(" PIPELINE DI DEDUPLICAZIONE IMMAGINI (TDM)")
    print("=" * 60)

    # 1. Scansione ed estrazione qualita'
    print(f"\n[1/4] Scansione directory: {dataset_dir}")
    scanner = ImageScanner(target_dir=dataset_dir)
    scanned_files = scanner.scan()
    print(f" -> Trovate {len(scanned_files)} immagini valide.")

    if len(scanned_files) < 2:
        print(" -> File insufficienti per eseguire la deduplicazione.")
        return

    # 2. Classificazione JBIG2 Regioning
    print("\n[2/4] Classificazione delle regioni (DOC vs PHOTO)...")
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
        print(f" -> {Path(item['filepath']).name}: {image_type.value} (Quality: {item['quality_score']:.2f})")

    # 3. Matching e Clustering
    print("\n[3/4] Calcolo similarita' e ricerca duplicati...")
    engine = DeduplicationEngine(doc_threshold=0.70, photo_threshold=0.75)
    clusters = engine.find_duplicates(processed_records)

    print(f" -> Trovati {len(clusters)} cluster di duplicati.")

    # 4. Isolamento nella cartella 'duplicati'
    if clusters:
        print(f"\n[4/4] Spostamento duplicati nella cartella '{isolation_dir}'...")
        iso_manager = IsolationManager(target_folder=isolation_dir)
        manifest_path = iso_manager.isolate_duplicates(clusters)
        
        for c in clusters:
            print(f"\n Cluster #{c.cluster_id} [{c.image_type.value}]")
            print(f"   MASTER (conservato): {Path(c.master_file).name}")
            for dup in c.duplicate_files:
                score = c.similarity_scores.get(dup, 0.0)
                print(f"   -> SPOSTATO: {Path(dup).name} (Similarita': {score:.2%})")

        print(f"\n Operazione completata. Manifest salvato in: {manifest_path}")
    else:
        print("\n[4/4] Nessun duplicato da isolare.")


def run_rollback(manifest_path: str):
    print(f"Ripristino in corso da manifest: {manifest_path}")
    iso_manager = IsolationManager()
    restored = iso_manager.rollback(manifest_path)
    print(f" Operazione completata: {restored} file ripristinati nella posizione originale.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline TDM per Deduplicazione Immagini")
    parser.add_argument("--dataset", type=str, default="./dataset", help="Directory contenente le immagini")
    parser.add_argument("--output", type=str, default="duplicati", help="Cartella per i duplicati isolati")
    parser.add_argument("--rollback", type=str, default=None, help="Percorso al file manifest.json per annulare lo spostamento")

    args = parser.parse_args()

    if args.rollback:
        run_rollback(args.rollback)
    else:
        run_pipeline(args.dataset, args.output)