import torch
import torch.nn as nn
import torch.optim as optim
import os

# Imports locais sem prefixo de pacote, para funcionar com `python src/train.py`
from visual_extractor import VisualExtractor  # noqa: F401  (pode ser usado em modo end-to-end)
from text_model import TextModel  # noqa: F401

class FusionModel(nn.Module):
    def __init__(self, config, visual_extractor=None, text_model=None):
        """
        Modelo Multimodal que funde features visuais, textuais e metadados de OCR.
        
        Inputs:
        - Visual: [Batch, 1280] (do MobileNet/EfficientNet)
        - Textual: [Batch, 256] (do LSTM/TextModel)
        - OCR Stats: [Batch, 3] (mean_conf, std_conf, low_conf_count)
        """
        super(FusionModel, self).__init__()
        
        # Se os modelos de extração forem passados, usamos eles (end-to-end).
        # Caso contrário, assumimos que a entrada já são os embeddings pré-calculados.
        self.visual_extractor = visual_extractor
        self.text_model = text_model
        self.use_precomputed_embeddings = (visual_extractor is None) and (text_model is None)

        # Definição das dimensões de entrada
        # Visual: 1280 (default MobileNet)
        visual_dim = 1280 
        # Textual: output_dim do TextModel (default 256)
        text_dim = config.get('text_output_dim', 256)
        # OCR Stats: 3 features manuais
        ocr_dim = 3
        
        # Projeção Visual
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Projeção Textual
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Projeção OCR Stats
        self.ocr_proj = nn.Sequential(
            nn.Linear(ocr_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Fusão
        fusion_dim = 512 + 256 + 32
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, config.get('num_classes', 2)) 
            # No Softmax here, using CrossEntropyLoss outside
        )

    def forward(self, visual_input, text_input, ocr_stats):
        """
        visual_input: [B, 1280] (embeddings) ou [B, 3, 224, 224] (raw images)
        text_input: [B, Text_Dim] (embeddings) ou [B, Seq_Len] (tokens)
        ocr_stats: [B, 3]
        """
        
        # Se não estiver usando embeddings pré-calculados, extrair on-the-fly
        if not self.use_precomputed_embeddings:
            # Assumimos que visual_extractor e text_model estão setados
            v_emb = self.visual_extractor(visual_input)
            t_emb = self.text_model(text_input)
        else:
            v_emb = visual_input
            t_emb = text_input
            
        # Projeções
        v_proj = self.visual_proj(v_emb)
        t_proj = self.text_proj(t_emb)
        o_proj = self.ocr_proj(ocr_stats)
        
        # Concatenação
        combined = torch.cat((v_proj, t_proj, o_proj), dim=1)
        
        # Classificação
        logits = self.classifier(combined)
        return logits

def train_fusion(model, train_loader, val_loader, epochs=10, lr=1e-4, save_path='models/fusion_model.pth'):
    """
    Loop de treinamento para o modelo de fusão.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for v_emb, t_emb, ocr_stats, labels in train_loader:
            v_emb, t_emb = v_emb.to(device), t_emb.to(device)
            ocr_stats, labels = ocr_stats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(v_emb, t_emb, ocr_stats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for v_emb, t_emb, ocr_stats, labels in val_loader:
                v_emb, t_emb = v_emb.to(device), t_emb.to(device)
                ocr_stats, labels = ocr_stats.to(device), labels.to(device)
                
                outputs = model(v_emb, t_emb, ocr_stats)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")
        
        scheduler.step(avg_val_loss)
        
        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            
    print("Training complete.")
    return model

if __name__ == "__main__":
    # Teste Unitário da Arquitetura
    config = {'num_classes': 2, 'text_output_dim': 256}
    model = FusionModel(config) # Modo Precomputed Embeddings
    
    # Dados Dummy
    B = 4
    v_in = torch.randn(B, 1280)
    t_in = torch.randn(B, 256)
    ocr_in = torch.randn(B, 3) # Mean, Std, Low_Count
    
    out = model(v_in, t_in, ocr_in)
    print(f"Output Shape: {out.shape}") # [4, 2]
