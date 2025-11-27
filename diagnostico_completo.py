#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script completo de diagnóstico do modelo.
Executa todas as análises necessárias para entender e corrigir o problema.
"""
import sys
import os
import subprocess

# Configurar encoding UTF-8
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("="*70)
    print("🔍 DIAGNÓSTICO COMPLETO DO MODELO")
    print("="*70)
    print()
    
    # 1. Verificar dataset
    print("📊 1. Verificando distribuição do dataset...")
    try:
        import pandas as pd
        if os.path.exists("data/labels.csv"):
            df = pd.read_csv("data/labels.csv")
            counts = df['label'].value_counts()
            print(f"   Total de amostras: {len(df)}")
            print(f"   Distribuição:")
            for label, count in counts.items():
                pct = count / len(df) * 100
                print(f"     {label}: {count} ({pct:.1f}%)")
            
            if abs(counts.get('authentic', 0) - counts.get('manipulated', 0)) > len(df) * 0.2:
                print("   ⚠️  Dataset desbalanceado! Considere gerar mais dados.")
            else:
                print("   ✅ Dataset balanceado")
        else:
            print("   ❌ data/labels.csv não encontrado!")
    except Exception as e:
        print(f"   ❌ Erro: {e}")
    
    print()
    
    # 2. Analisar probabilidades
    print("📈 2. Analisando distribuição de probabilidades...")
    print("   (Isso pode demorar alguns minutos)")
    try:
        result = subprocess.run(
            [sys.executable, "src/analyze_probabilities.py"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print("   ✅ Análise concluída!")
            # Mostrar threshold recomendado
            if "Use threshold" in result.stdout:
                for line in result.stdout.split('\n'):
                    if "Use threshold" in line or "Melhor Threshold" in line:
                        print(f"   {line.strip()}")
        else:
            print(f"   ⚠️  Análise com avisos: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("   ⏱️  Análise demorou muito (timeout)")
    except Exception as e:
        print(f"   ❌ Erro: {e}")
    
    print()
    
    # 3. Testar diferentes thresholds
    print("🎯 3. Testando diferentes thresholds...")
    try:
        result = subprocess.run(
            [sys.executable, "src/evaluate_with_threshold.py"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print("   ✅ Teste de thresholds concluído!")
            # Mostrar melhor threshold
            if "MELHOR THRESHOLD" in result.stdout:
                lines = result.stdout.split('\n')
                start_idx = next((i for i, l in enumerate(lines) if "MELHOR THRESHOLD" in l), None)
                if start_idx:
                    for i in range(start_idx, min(start_idx + 8, len(lines))):
                        print(f"   {lines[i]}")
        else:
            print(f"   ⚠️  Teste com avisos")
    except Exception as e:
        print(f"   ❌ Erro: {e}")
    
    print()
    print("="*70)
    print("📋 RESUMO E PRÓXIMOS PASSOS")
    print("="*70)
    print()
    print("1. ✅ Verifique os gráficos em reports/")
    print("   - probability_distribution.png")
    print("   - pr_curve_analysis.png")
    print()
    print("2. ✅ Veja o threshold recomendado em reports/threshold_recommendation.json")
    print()
    print("3. 🔄 Retreine o modelo com class weights ajustados:")
    print("   python run_train.py")
    print()
    print("4. 🎯 Use o threshold ótimo na inferência")
    print()
    print("📖 Para mais detalhes, veja: DIAGNOSTICO_E_SOLUCOES.md")
    print("="*70)

if __name__ == "__main__":
    main()

