# Gestione dell'isolamento dei file duplicati nella cartella 'duplicati', generazione del file manifest.json e procedura di rollback.


from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import json
import shutil

from ..matcher.engine import DuplicateCluster


# classe che gestisce lo spostamento sicuro dei file duplicati e il loro ripristino
class IsolationManager:
   

    def __init__(self, target_folder: str = "duplicati"):
       #percorso assoluto della cartella di destinazione per i duplicati
        self.target_dir = Path(target_folder)


    #sposta i file duplicati nella cartella target e genera il file manifest.json
    def isolate_duplicates(self, clusters: List[DuplicateCluster]) -> Path:
      
        if not self.target_dir.exists():
            self.target_dir.mkdir(parents=True, exist_ok=True)

        moved_files_map: Dict[str, str] = {}
        manifest_clusters: List[Dict[str, Any]] = []

        for cluster in clusters:
            cluster_info = {
                "cluster_id": cluster.cluster_id,
                "image_type": cluster.image_type.value,
                "master_file": cluster.master_file,
                "isolated_files": []
            }

            for dup_path_str in cluster.duplicate_files:
                original_path = Path(dup_path_str)
                if not original_path.exists():
                    continue

                # Evita collisioni di nomi aggiungendo l'ID cluster come prefisso se necessario
                destination_name = f"c{cluster.cluster_id}_{original_path.name}"
                destination_path = self.target_dir / destination_name

                # Spostamento sicuro del file
                shutil.move(str(original_path), str(destination_path))

                moved_files_map[str(destination_path)] = str(original_path.resolve())
                
                cluster_info["isolated_files"].append({
                    "original_path": str(original_path.resolve()),
                    "quarantine_path": str(destination_path.resolve()),
                    "similarity_to_master": cluster.similarity_scores.get(dup_path_str, 0.0)
                })

            manifest_clusters.append(cluster_info)

        # Generazione del file manifest.json
        manifest_data = {
            "timestamp": datetime.now().isoformat(),
            "total_clusters": len(clusters),
            "total_isolated_files": len(moved_files_map),
            "isolation_directory": str(self.target_dir.resolve()),
            "clusters": manifest_clusters,
            "rollback_map": moved_files_map
        }

        manifest_path = self.target_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=4, ensure_ascii=False)

        return manifest_path

    #ripristina i file spostati nella cartella 'duplicati' alle loro posizioni originali leggendo il file manifest.json
    def rollback(self, manifest_path: str) -> int:
        
        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            raise FileNotFoundError(f"File manifest non trovato: {manifest_path}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        rollback_map = manifest_data.get("rollback_map", {})
        restored_count = 0

        for quarantine_path_str, original_path_str in rollback_map.items():
            quarantine_path = Path(quarantine_path_str)
            original_path = Path(original_path_str)

            if quarantine_path.exists():
                # Assicurati che la directory originale esista ancora
                original_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(quarantine_path), str(original_path))
                restored_count += 1

        # Rimuovi il file manifest dopo un rollback completato
        manifest_file.unlink()
        return restored_count