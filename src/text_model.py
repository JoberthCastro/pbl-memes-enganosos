import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import re
import os
from collections import Counter
import pickle

# --- Tokenizer (Mimics Keras Tokenizer) ---
class Tokenizer:
    def __init__(self, num_words=None, lower=True, oov_token="<OOV>"):
        self.num_words = num_words
        self.lower = lower
        self.oov_token = oov_token
        self.word_index = {}
        self.index_word = {}
        self.word_counts = Counter()
        
    def fit_on_texts(self, texts):
        """
        Constrói o vocabulário baseado na lista de textos.
        """
        for text in texts:
            if self.lower:
                text = text.lower()
            # Tokenização simples via regex (palavras e pontuações básicas)
            words = re.findall(r'\b\w+\b', text)
            self.word_counts.update(words)
            
        # Ordenar por frequência
        sorted_words = self.word_counts.most_common(self.num_words - 2 if self.num_words else None)
        
        # 0 reservado para padding
        self.word_index = {self.oov_token: 1}
        self.index_word = {1: self.oov_token}
        
        for i, (word, _) in enumerate(sorted_words):
            idx = i + 2 # Start from 2 (0=pad, 1=OOV)
            self.word_index[word] = idx
            self.index_word[idx] = word
            
    def texts_to_sequences(self, texts):
        """
        Converte lista de textos em lista de sequências de índices.
        """
        sequences = []
        for text in texts:
            if self.lower:
                text = text.lower()
            words = re.findall(r'\b\w+\b', text)
            seq = []
            for w in words:
                idx = self.word_index.get(w, self.word_index[self.oov_token])
                # Se num_words definido, clipar índices maiores (se logicamente necessário, mas fit já cuida)
                seq.append(idx)
            sequences.append(seq)
        return sequences

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

def pad_sequences(sequences, maxlen=None, padding='post', value=0):
    """
    Padding manual similar ao Keras.
    """
    if not maxlen:
        maxlen = max(len(s) for s in sequences) if sequences else 0
        
    padded = np.full((len(sequences), maxlen), value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        if not seq: continue
        trunc = seq[:maxlen]
        if padding == 'post':
            padded[i, :len(trunc)] = trunc
        else:
            padded[i, -len(trunc):] = trunc
            
    return padded

# --- Text Model ---
class TextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, pretrained_embeddings=None, output_dim=256):
        super(TextModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            # pretrained_embeddings deve ser um tensor [vocab_size, embedding_dim]
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = True # Fine-tuning opcional
            
        # Arquitetura: 1D CNN + LSTM ou apenas LSTM
        # Aqui usamos Bi-LSTM + Global Max Pooling para robustez
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # CNN1D Opcional (descomentar para experimentar híbrido)
        # self.conv1d = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)
        
        # Output dim público para fusão
        self.output_dim = output_dim

    def forward(self, x):
        # x: [Batch, Seq_Len]
        embedded = self.embedding(x) # [B, L, Emb]
        
        # LSTM
        # output: [B, L, 2*Hidden]
        output, (hn, cn) = self.lstm(embedded)
        
        # Global Max Pooling (max over time)
        # Permuta para [B, 2*Hidden, L] para usar max_pool1d ou fazer manual
        output = output.permute(0, 2, 1) # [B, 2*Hidden, L]
        pooled = torch.max(output, dim=2)[0] # [B, 2*Hidden]
        
        features = self.dropout(pooled)
        features = self.fc(features) # [B, Output_Dim]
        
        return features

def get_text_embedding(text, tokenizer, model, max_len=100, device='cpu'):
    """
    Gera o embedding vetorial para um texto único.
    """
    model.eval()
    model.to(device)
    
    seqs = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seqs, maxlen=max_len, padding='post')
    
    tensor_in = torch.tensor(padded, dtype=torch.long).to(device)
    
    with torch.no_grad():
        embedding = model(tensor_in)
        
    return embedding.cpu().numpy().flatten()

def load_glove_embeddings(path, word_index, embedding_dim=100):
    """
    Carrega vetores GloVe e cria matriz de embeddings alinhada ao vocabulário.
    """
    embeddings_index = {}
    if not os.path.exists(path):
        print(f"Warning: GloVe path {path} not found. Using random embeddings.")
        return None

    with open(path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    hits = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_vector) == embedding_dim:
                embedding_matrix[i] = embedding_vector
                hits += 1
                
    print(f"GloVe: Loaded {hits} vectors from {len(word_index)} words.")
    return embedding_matrix

# --- Standalone Training Script ---
def train_standalone():
    print("--- Standalone Text Model Training (Debug) ---")
    
    # Dados Dummy
    texts = [
        "This is a fake news article about aliens.",
        "Official report confirms economic growth.",
        "Click here to win a free iphone immediately.",
        "Scientific study shows coffee is good.",
        "Shocking truth revealed by anonymous source.",
        "Weather forecast for tomorrow is sunny."
    ]
    labels = [1, 0, 1, 0, 1, 0] # 1: Fake, 0: Real
    
    # Config
    MAX_WORDS = 1000
    MAX_LEN = 20
    EMBED_DIM = 50
    HIDDEN_DIM = 32
    EPOCHS = 5
    
    # Pipeline
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=MAX_LEN)
    y = np.array(labels)
    
    # Dataset
    dataset = TensorDataset(torch.from_numpy(X).long(), torch.from_numpy(y).float())
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Model (Modificado para classificação binária neste script de teste)
    vocab_size = len(tokenizer.word_index) + 1
    model = TextModel(vocab_size, EMBED_DIM, HIDDEN_DIM, output_dim=16) # output 16 features
    
    # Head de classificação temporário para o treino
    classifier_head = nn.Linear(16, 1)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(classifier_head.parameters()), lr=0.01)
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            
            features = model(batch_x)
            logits = classifier_head(features).squeeze()
            
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")
        
    print("Training complete.")
    
    # Teste Embedding
    sample_text = "Fake news about aliens"
    emb = get_text_embedding(sample_text, tokenizer, model, max_len=MAX_LEN)
    print(f"Embedding shape for '{sample_text}': {emb.shape}")

if __name__ == "__main__":
    train_standalone()
