# Wheel Inspector - Streamlit App

Esta representa a prova de conceito e aplicação web de demonstração do projeto, desenvolvida em [Streamlit](https://streamlit.io/).

## Guia de Execução

1. Aceder ao diretório transacional `app/`:
   ```bash
   cd app
   ```

2. Proceder à instalação da árvore de dependências associadas:
   ```bash
   pip install -r requirements.txt
   ```

3. Iniciar a execução do servidor local Streamlit:
   ```bash
   python -m streamlit run app.py
   ```

## Notas Técnicas
- A aplicação procede autonomamente à procura e reflexão de binários compilados `.pt` (pesos treinados pela arquitetura YOLOv8) quando colocados dentro do diretório `/modelos/`. 
- Deverá ser assegurada a prévia conclusão de um ciclo de treino, e a sua importação de métricas com sucesso para o respetivo diretório de artefactos finais, antes da instanciação deste servidor analítico.