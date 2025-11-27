# 📊 Resultados do Diagnóstico

## ✅ Gráficos Gerados

Os seguintes arquivos foram criados em `reports/`:

1. **`probability_distribution.png`** - Distribuição de probabilidades por classe
2. **`pr_curve_analysis.png`** - Precision-Recall Curve com ponto ótimo
3. **`threshold_recommendation.json`** - Recomendações de threshold

## 🔍 Análise dos Resultados

### 📈 Estatísticas das Probabilidades

- **Média**: 0.4750
- **Mediana**: 0.4743
- **Desvio Padrão**: 0.0197 (MUITO BAIXO!)
- **Range**: 0.4047 - 0.5005 (muito concentrado)

### ⚠️ Problema Identificado

**As probabilidades estão COLAPSADAS!**

- Todas as probabilidades estão muito próximas de 0.5
- Desvio padrão muito baixo (0.0197) indica que o modelo não está confiante
- Diferença entre classes é mínima:
  - Authentic: 0.4673
  - Manipulated: 0.4827
  - **Diferença de apenas 1.5%!**

### 🎯 Threshold Recomendado

- **Threshold Ótimo (F1 máximo)**: 0.48
  - Precision: 0.8571
  - Recall: 0.6000
  - F1-Score: 0.7059

- **Melhor Threshold Geral**: 0.10
  - F1-Score: 0.6667
  - Precision: 0.500
  - Recall: 1.000

### 📊 PR AUC = 0.796

O PR AUC de 0.796 é **bom**, mas as probabilidades colapsadas indicam que:
- ✅ O modelo tem capacidade de distinguir (PR AUC alto)
- ❌ Mas as features não são suficientemente discriminativas
- ❌ O modelo não está confiante nas suas predições

## 🛠 O Que Isso Significa

### Problema Raiz

O modelo não está aprendendo diferenças significativas entre as classes porque:

1. **Features não são discriminativas**
   - Diferença de apenas 1.5% entre classes
   - Probabilidades todas muito próximas

2. **Dataset pode ser muito pequeno ou similar**
   - 100 amostras (50/50) pode não ser suficiente
   - Imagens sintéticas podem ser muito similares

3. **Modelo pode estar subutilizado**
   - Backbone congelado pode não estar extraindo features úteis
   - Modelo textual pode não estar capturando diferenças semânticas

## 🚀 Soluções Recomendadas

### 1. **Imediato: Usar Threshold 0.48**

Modifique a inferência para usar threshold 0.48 em vez de 0.5:

```python
# Em src/evaluate.py ou src/api/main.py
threshold = 0.48  # Threshold ótimo encontrado
prob_manipulated = probs[0, 1].item()
pred_idx = 1 if prob_manipulated >= threshold else 0
```

### 2. **Curto Prazo: Aumentar Dataset**

```bash
python data/synthetic_generator.py --n_authentic 200 --n_manipulated 200
```

Mais dados = melhor aprendizado de padrões.

### 3. **Médio Prazo: Melhorar Features**

- **Usar OCR stats reais** (não zeros)
- **Adicionar mais features** (metadados, estatísticas de imagem)
- **Fine-tuning do backbone** (descongelar algumas camadas)

### 4. **Longo Prazo: Melhorar Arquitetura**

- **Aumentar capacidade do modelo**
- **Usar modelos pré-treinados melhores**
- **Adicionar atenção entre modalidades**

## 📋 Próximos Passos

1. ✅ **Gráficos gerados** - Veja em `reports/`
2. 🔄 **Aplicar threshold 0.48** na inferência
3. 📈 **Aumentar dataset** para 200+ amostras
4. 🔄 **Retreinar** com mais dados
5. 📊 **Reavaliar** após melhorias

---

**Os gráficos estão prontos! Abra `reports/probability_distribution.png` e `reports/pr_curve_analysis.png` para visualizar.** 📊

