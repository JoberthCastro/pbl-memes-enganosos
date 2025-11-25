import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import MemeDataset
from src.preprocessing import get_transforms
from src.text_model import Tokenizer, TextModel
from src.visual_extractor import VisualExtractor
from src.fusion_model import FusionModel


CSV_PATH = "data/labels.csv"
DATA_DIR = "data/raw"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


def train(num_epochs: int = 3, batch_size: int = 4, lr: float = 1e-4):
    print("📦 Lendo labels...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"labels.csv não encontrado em {CSV_PATH}. Rode:  python data/synthetic_generator.py"
        )

    df = pd.read_csv(CSV_PATH)
    texts = df["original_text_content"].astype(str).tolist()

    print("📝 Criando tokenizer...")
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)

    print("🖼 Preparando dataset...")
    transform = get_transforms(mode="train")
    dataset = MemeDataset(
        csv_file=CSV_PATH,
        root_dir=DATA_DIR,
        tokenizer=tokenizer,
        max_len=100,
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("🎨 Carregando modelo visual...")
    visual = VisualExtractor(model_name="mobilenet_v2", pretrained=True)

    print("✍ Carregando modelo textual...")
    text_model = TextModel(
        vocab_size=len(tokenizer.word_index) + 2,
        embedding_dim=128,
        hidden_dim=128
    )

    print("🔗 Carregando modelo de fusão...")
    fusion = FusionModel({
        "num_classes": 2,
        "text_output_dim": text_model.output_dim
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📌 Usando dispositivo: {device}")

    visual.to(device)
    text_model.to(device)
    fusion.to(device)

    optimizer = torch.optim.Adam(
        list(visual.parameters()) +
        list(text_model.parameters()) +
        list(fusion.parameters()),
        lr=lr
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    print("🚀 Iniciando treinamento...\n")

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        visual.train()
        text_model.train()
        fusion.train()

        for images, tokens, labels in dataloader:
            images = images.to(device)
            tokens = tokens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            v_emb = visual(images)
            t_emb = text_model(tokens)

            # OCR stats zerados (treino básico)
            ocr_stats = torch.zeros(images.size(0), 3, device=device)

            logits = fusion(v_emb, t_emb, ocr_stats)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        acc = correct / total * 100
        print(f"📌 Época {epoch+1}/{num_epochs} — Loss: {epoch_loss:.4f} | Acc: {acc:.2f}%")

    print("\n💾 Salvando pesos...")
    torch.save(visual.state_dict(), os.path.join(MODEL_DIR, "visual_model.pth"))
    torch.save(text_model.state_dict(), os.path.join(MODEL_DIR, "text_model.pth"))
    torch.save(fusion.state_dict(), os.path.join(MODEL_DIR, "fusion_model.pth"))

    print("🎉 Treinamento concluído com sucesso!")


if __name__ == "__main__":
    train()