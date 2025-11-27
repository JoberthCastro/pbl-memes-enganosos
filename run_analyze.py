#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrapper para rodar análise de probabilidades com encoding correto
"""
import sys
import os

# Configurar encoding UTF-8
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar e executar
from src.analyze_probabilities import analyze_probability_distribution
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyze_probability_distribution("data/raw", "models/fusion_model.pth", device)

