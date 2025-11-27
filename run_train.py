#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wrapper script para rodar o treinamento com encoding correto no Windows
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
from src.train import train

if __name__ == "__main__":
    train(num_epochs=3)

