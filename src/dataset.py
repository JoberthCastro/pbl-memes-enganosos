import os
from typing import List

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class MemeDataset(Dataset):
    """
    Dataset que busca automaticamente imagens nas subpastas:
        data/raw/authentic/
        data/raw/manipulated/

    Espera um CSV (labels.csv) com colunas:
        filename, label, manipulation_type, original_text_content
    """

    def __init__(self, csv_file: str, root_dir: str, tokenizer, max_len: int = 100, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

        df = pd.read_csv(csv_file)

        # Verifica caminhos reais das imagens
        valid_rows = []
        for _, row in df.iterrows():
            fname = row["filename"]

            possible_paths = [
                os.path.join(root_dir, "authentic", fname),
                os.path.join(root_dir, "manipulated", fname),
                os.path.join(root_dir, fname),
            ]

            for p in possible_paths:
                if os.path.exists(p):
                    row = row.copy()
                    row["real_path"] = p
                    valid_rows.append(row)
                    break

        if len(valid_rows) == 0:
            raise RuntimeError(
                f"Nenhuma imagem encontrada em '{root_dir}'. "
                "Verifique se o synthetic_generator gerou as imagens corretamente."
            )

        if len(valid_rows) < len(df):
            print(f"⚠ Aviso: {len(df) - len(valid_rows)} amostras ignoradas porque as imagens não foram encontradas.")

        self.df = pd.DataFrame(valid_rows)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = row["real_path"]
        label_str = str(row["label"]).lower()

        # Converte string de label em inteiro (0: authentic, 1: manipulated)
        label = 0 if label_str == "authentic" else 1
        text = str(row.get("original_text_content", ""))

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Tokenização
        tokens_list: List[List[int]] = self.tokenizer.texts_to_sequences([text])
        tokens = tokens_list[0] if len(tokens_list) > 0 else []

        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[: self.max_len]

        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image, tokens_tensor, label_tensor