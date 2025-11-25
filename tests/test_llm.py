import pytest
from src.llm_integration import LLMIntegration

def test_prompt_construction():
    llm = LLMIntegration(mock_mode=True)
    prompt = llm.construct_prompt("Teste", {"platform": "Twitter"})
    assert "Twitter" in prompt
    assert "Teste" in prompt
    assert "JSON" in prompt

def test_mock_inference_suspicious():
    llm = LLMIntegration(mock_mode=True)
    text = "URGENTE: Compartilhe antes que apaguem essa verdade secreta."
    result = llm.analyze(text)
    
    assert result['label'] == 'suspeito'
    assert result['score'] > 0.5
    assert len(result['issues']) > 0

def test_mock_inference_authentic():
    llm = LLMIntegration(mock_mode=True)
    text = "Hoje o dia está bonito para aprender Python."
    result = llm.analyze(text)
    
    assert result['label'] == 'autêntico'
    assert result['score'] < 0.5

def test_parse_response():
    llm = LLMIntegration(mock_mode=True)
    
    # Teste com markdown code block
    raw_response = """
    Aqui está a análise:
    ```json
    {
        "label": "suspeito",
        "score": 0.8,
        "issues": ["Fake news"],
        "explanation": "Teste"
    }
    ```
    """
    parsed = llm.parse_llm_response(raw_response)
    assert parsed['label'] == "suspeito"
    assert parsed['score'] == 0.8

    # Teste JSON quebrado
    bad_response = "{ label: 'erro' " # json inválido
    parsed_bad = llm.parse_llm_response(bad_response)
    assert parsed_bad['label'] == "erro"

