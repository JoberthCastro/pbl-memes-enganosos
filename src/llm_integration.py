import json
import os
import re
from dotenv import load_dotenv
import logging

# Tenta importar Google Generative AI, mas não falha se não instalado (fallback para mock)
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

load_dotenv()

logger = logging.getLogger(__name__)

class LLMIntegration:
    def __init__(self, api_key=None, model_name="models/gemini-flash-latest", mock_mode=False):
        """
        Módulo de integração com LLM para validação semântica de memes.
        
        Args:
            api_key: Chave de API (opcional se estiver no .env)
            model_name: Nome do modelo (ex: 'gemini-pro')
            mock_mode: Se True, usa respostas simuladas localmente.
        """
        self.mock_mode = mock_mode
        self.model_name = model_name
        
        if not mock_mode:
            if not HAS_GENAI:
                logger.warning("google-generativeai not installed. Switching to mock mode.")
                self.mock_mode = True
            else:
                key = api_key or os.getenv("GEMINI_API_KEY")
                if not key:
                    logger.warning("No API Key found. Switching to mock mode.")
                    self.mock_mode = True
                else:
                    genai.configure(api_key=key)
                    # Modelos recomendados (v1): 'gemini-1.5-flash', 'gemini-1.5-pro'
                    self.model = genai.GenerativeModel(model_name)

    def construct_prompt(self, text_extracted, metadata=None):
        """
        Gera o prompt estruturado para o LLM.
        """
        platform = metadata.get('platform', 'desconhecida') if metadata else 'desconhecida'
        metrics = metadata.get('metrics', 'não informadas') if metadata else 'não informadas'
        visual_cues = metadata.get('visual_cues', []) if metadata else []
        model_pred = metadata.get('model_prediction', 'desconhecida') if metadata else 'desconhecida'
        model_conf = metadata.get('model_confidence', 0.0) if metadata else 0.0
        
        prompt = f"""
Você é um analista forense digital especializado em detectar desinformação e manipulação em mídias sociais.
Sua tarefa é analisar o conteúdo extraído de uma imagem (possível meme ou print de rede social) e determinar a probabilidade de ser enganoso, manipulado ou fora de contexto.

**Dados do Input:**
- **Texto Extraído (OCR):** "{text_extracted}"
- **Plataforma Estimada:** {platform}
- **Métricas Visíveis:** {metrics}
- **Indícios Visuais:** {", ".join(visual_cues)}
- **Previsão do Modelo Multimodal (Vision+Text+OCR)**: {model_pred} (confiança {model_conf:.2f})

**Instruções CRUCIAIS:**
1. **IGNORE ERROS DE OCR:** O texto foi extraído automaticamente de uma imagem de baixa qualidade. Erros como 'Satide' (Saúde), 'reforga' (reforça), 'vacinagao' (vacinação) ou caracteres estranhos são artefatos técnicos, NÃO prova de fraude. NÃO mencione esses erros como suspeitas.
2. **CONSIDERE O MODELO MULTIMODAL:** A previsão `{model_pred}` do modelo treinado é um sinal forte, mas você PODE discordar dela se o conteúdo textual indicar o contrário. Se concordar, mencione isso explicitamente na explicação. Se discordar, explique claramente por quê.
3. **FOQUE NA MENTIRA:** Só classifique como 'suspeito' se a **informação** transmitida for falsa, enganosa, alarmista ou impossível (ex: 'Vacina causa autismo', 'Terra é plana', '999 trilhões de likes').
4. **CONTEXTO:** Se o texto for uma notícia plausível (ex: campanha de vacinação, obra em ponte, resultado de jogo), classifique como 'autêntico', mesmo que a formatação esteja feia.
5. **FONTES:** Nomes de jornais genéricos (ex: 'Jornal_Real_10') são comuns em datasets sintéticos de teste. Não use o nome da fonte como único critério para suspeita, a menos que imite um jornal famoso para enganar.

**Decisão:**
- Se a mensagem for uma verdade aceita ou notícia plausível -> 'autêntico'.
- Se a mensagem for absurda, teoria da conspiração ou golpe -> 'suspeito'.

**Formato de Saída Obrigatório (JSON):**
Retorne APENAS um objeto JSON válido (sem markdown ```json ... ```) com os seguintes campos:
{{
    "label": "suspeito" ou "autêntico",
    "score": <float entre 0.0 e 1.0, onde 1.0 é certeza absoluta de manipulação>,
    "issues": ["lista", "de", "problemas", "SEMÂNTICOS", "encontrados"],
    "explanation": "Breve justificativa focada no conteúdo da mensagem."
}}
"""
        return prompt.strip()

    def parse_llm_response(self, response_text):
        """
        Limpa e converte a resposta do LLM em dicionário Python.
        """
        try:
            # Remove marcadores de código markdown se houver
            cleaned_text = re.sub(r'```json\s*', '', response_text)
            cleaned_text = re.sub(r'```\s*', '', cleaned_text)
            
            data = json.loads(cleaned_text)
            
            # Validação de Schema Mínimo
            required_keys = ["label", "score", "issues", "explanation"]
            for key in required_keys:
                if key not in data:
                    data[key] = None # Preenche faltantes
                    
            return data
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM: {response_text}")
            return {
                "label": "erro",
                "score": 0.0,
                "issues": ["Erro de parsing na resposta do LLM"],
                "explanation": "Não foi possível interpretar a resposta do modelo."
            }

    def analyze(self, text_extracted, metadata=None):
        """
        Executa a análise completa (Prompt -> LLM -> Parse).
        """
        prompt = self.construct_prompt(text_extracted, metadata)
        
        if self.mock_mode:
            return self._mock_inference(text_extracted)
            
        try:
            response = self.model.generate_content(prompt)
            return self.parse_llm_response(response.text)
        except Exception as e:
            logger.error(f"LLM API Error: {e}")
            return {
                "label": "erro",
                "score": 0.0, 
                "issues": [str(e)],
                "explanation": "Falha na comunicação com a API."
            }

    def _mock_inference(self, text):
        """
        Simulação local baseada em heurísticas simples para testes.
        """
        text_lower = text.lower()
        issues = []
        score = 0.1
        label = "autêntico"
        
        # Heurísticas simples de "Fake News"
        suspicious_keywords = [
            "urgente", "compartilhe antes que apaguem", "a verdade secreta", 
            "cura milagrosa", "terra plana", "chip 5g", "acabou de sair"
        ]
        
        for kw in suspicious_keywords:
            if kw in text_lower:
                issues.append(f"Uso da palavra-chave sensacionalista: '{kw}'")
                score += 0.3
                
        if len(text) < 10:
            issues.append("Texto muito curto para análise confiável")
            
        if score > 0.6:
            label = "suspeito"
            score = min(score, 0.99)
            explanation = "Texto contém múltiplos indícios de linguagem sensacionalista comum em desinformação."
        else:
            explanation = "Texto parece neutro e sem indícios óbvios de manipulação (Análise Mock)."
            
        return {
            "label": label,
            "score": round(score, 2),
            "issues": issues,
            "explanation": explanation
        }

if __name__ == "__main__":
    # Exemplo de uso
    llm = LLMIntegration(mock_mode=True)
    
    sample_text = "URGENTE: A terra é plana e a NASA mentiu! Compartilhe antes que apaguem."
    meta = {"platform": "Facebook", "metrics": "10k Shares"}
    
    result = llm.analyze(sample_text, meta)
    print(json.dumps(result, indent=2, ensure_ascii=False))
