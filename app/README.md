# Wheel Inspector - Aplicacao de Demonstracao

Aplicacao web interativa construida em [Streamlit](https://streamlit.io/) para demonstrar a detecao de componentes de rodas (jantes, parafusos, rodas) utilizando modelos YOLOv8.

## Funcionalidades

- **Upload de imagem** para inspecao visual
- **Selecao de modelo** via dropdown (caso existam multiplos modelos treinados)
- **Slider de confianca** ajustavel para controlar o limiar de detecao
- **Visualizacao lado a lado** da imagem original e das bounding boxes
- **Tabela de detecoes** com classe e nivel de confianca
- **Output JSON** estruturado com todas as detecoes
- **Exportacao** de resultados para ficheiro JSON
- **Comparacao de modelos** lado a lado sobre a mesma imagem
- **Historico de inferencias** das ultimas sessoes

## Execucao

```bash
cd app
pip install -r requirements.txt
python -m streamlit run app.py
```

## Requisitos

- Python 3.10+
- Pelo menos um ficheiro de pesos `.pt` presente em `modelos/`
- Dependencias listadas em `requirements.txt`