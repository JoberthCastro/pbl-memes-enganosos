# 🔧 Melhorias Implementadas para o Modelo

## 📊 Problema Identificado

O modelo estava extremamente desbalanceado:
- **Recall = 1.0** (detecta todos os manipulados) ✅
- **Specificity = 0.1** (só acerta 10% dos autênticos) ❌
- **FPR = 0.9** (90% de falsos positivos) ❌

O modelo estava classificando quase tudo como "manipulado".

## ✅ Soluções Implementadas

### 1. **Class Weights no Loss Function**

Adicionado balanceamento automático de classes no treinamento:

```python
# Calcula pesos inversamente proporcionais à frequência das classes
class_weights = torch.tensor([
    total / (2 * count_authentic),  # Peso maior para classe minoritária
    total / (2 * count_manipulated)
])
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
```

**Como usar:**
- O treinamento agora calcula automaticamente os pesos
- Classes menos frequentes recebem peso maior
- Isso força o modelo a prestar mais atenção na classe minoritária

### 2. **Métricas Detalhadas por Classe**

Agora o treinamento mostra métricas separadas:
```
📌 Época 1/3 — Loss: 13.48 | Acc: 56.25%
   Authentic: 45.0% (9/20) | Manipulated: 67.5% (27/40)
```

Isso permite monitorar o balanceamento durante o treino.

### 3. **Script de Ajuste de Threshold**

Criado `src/evaluate_with_threshold.py` para encontrar o threshold ótimo:

```bash
# Encontrar threshold ótimo automaticamente
python src/evaluate_with_threshold.py

# Testar threshold específico
python src/evaluate_with_threshold.py --threshold 0.6
```

**O que faz:**
- Testa thresholds de 0.3 a 0.8
- Encontra o threshold com melhor F1-score balanceado
- Mostra confusion matrix para cada threshold
- Salva análise em `reports/threshold_analysis.json`

## 🚀 Como Aplicar as Melhorias

### Passo 1: Retreinar com Class Weights

```bash
python run_train.py
```

O modelo agora será treinado com class weights, o que deve melhorar o balanceamento.

### Passo 2: Encontrar Threshold Ótimo

```bash
python src/evaluate_with_threshold.py
```

Isso vai testar diferentes thresholds e mostrar qual dá o melhor resultado.

### Passo 3: Aplicar Threshold na Inferência

Você pode modificar `src/evaluate.py` ou `src/api/main.py` para usar o threshold ótimo:

```python
# Em vez de:
pred_idx = torch.argmax(probs, dim=1)

# Use:
prob_manipulated = probs[0, 1].item()
threshold = 0.6  # Threshold ótimo encontrado
pred_idx = 1 if prob_manipulated >= threshold else 0
```

## 📈 Próximas Melhorias Sugeridas

### 1. **Aumentar Dataset**
- Gerar mais dados sintéticos (200+ de cada classe)
- Adicionar dados reais se possível

### 2. **Data Augmentation Mais Agressiva**
- Adicionar mais transformações visuais
- Variações de texto (sinônimos, paráfrases)

### 3. **Oversampling (SMOTE)**
```python
from imblearn.over_sampling import SMOTE
# Aplicar SMOTE nos embeddings antes do treino
```

### 4. **Focal Loss**
Substituir CrossEntropyLoss por Focal Loss, que foca em exemplos difíceis:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

### 5. **Calibração de Probabilidades**
```python
from sklearn.calibration import CalibratedClassifierCV
# Calibrar probabilidades após o treino
```

### 6. **Early Stopping com Validação**
Adicionar validação durante o treino para evitar overfitting:

```python
# Validar a cada época e parar se não melhorar
if val_f1 < best_f1:
    patience += 1
    if patience >= 3:
        break
```

## 📝 Checklist de Melhorias

- [x] Class weights no loss function
- [x] Métricas detalhadas por classe
- [x] Script de análise de threshold
- [ ] Aumentar tamanho do dataset
- [ ] Implementar Focal Loss
- [ ] Adicionar early stopping
- [ ] Calibração de probabilidades
- [ ] Oversampling (SMOTE)

## 🎯 Resultados Esperados

Após aplicar as melhorias, você deve ver:

- **Specificity > 0.5** (pelo menos 50% dos autênticos corretos)
- **FPR < 0.5** (menos de 50% de falsos positivos)
- **F1-Score balanceado** (não só alto recall)
- **Confusion Matrix mais equilibrada**

---

**Execute `python run_train.py` para retreinar com as melhorias!** 🚀

