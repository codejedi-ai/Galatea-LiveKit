import csv
import os
from pathlib import Path
from .manager import manager

def update_model_csvs():
    root = Path(__file__).resolve().parent.parent.parent
    models_path = root / "models.csv"
    media_path = root / "model_media.csv"

    # 1. models.csv: modelid, provider, is_instruct, base, is_deprecated
    with open(models_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["modelid", "provider", "is_instruct", "base", "is_deprecated"])
        for p_name in manager.list_providers():
            provider = manager.get_provider(p_name)
            for model in provider.list_models():
                writer.writerow([
                    model.id, 
                    model.provider, 
                    str(model.is_instruct).lower(),
                    model.base,
                    str(model.is_deprecated).lower()
                ])

    # 2. model_media.csv: modelid, media, type
    with open(media_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["modelid", "media", "type"])
        for p_name in manager.list_providers():
            provider = manager.get_provider(p_name)
            for model in provider.list_models():
                for m in model.input_media:
                    writer.writerow([model.id, m, "in"])
                for m in model.output_media:
                    writer.writerow([model.id, m, "out"])

    return str(models_path), str(media_path)
