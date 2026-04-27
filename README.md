# Wheel Components Inspector

Solucao de ponta a ponta para detecao automatica de **rodas**, **jantes** e **parafusos** em imagens, utilizando YOLOv8 e visao computacional. Projeto desenvolvido no ambito do Trabalho Pratico da unidade curricular de Inteligencia Artificial (Engenharia Informatica, ESTG - P.Porto, 2025/2026).

## Caso de Uso

Inspecao visual automatizada de componentes de rodas em contexto industrial/oficinal. O sistema deteta e localiza tres classes de objetos:

| Classe | Descricao |
|--------|-----------|
| `roda` | Conjunto completo de pneu + jante |
| `jante` | Parte central metalica (rim/hub) |
| `parafuso` | Pernos de fixacao da jante ao eixo |

## Estrutura do Projeto

```
wheel-components-inspector/
  dataset/                    # Imagens e anotacoes (formato YOLOv8)
    train/images/ labels/
    valid/images/ labels/
    test/images/  labels/
    data.yaml                 # Configuracao do dataset (Roboflow)
    dataset.md                # Documentacao do dataset
  modelos/                    # Artefactos dos modelos treinados
    modelo1/                  # Modelo base (v1.0)
      weights.pt              # Pesos do modelo
      model_card.md           # Documentacao legivel por humanos
      model_manifest.json     # Metadados estruturados (legivel por maquina)
    modelo2/                  # Modelo melhorado (v2.0 - foco em parafusos)
      weights.pt
      model_card.md
      model_manifest.json
      train_config.yaml       # Configuracao de treino reproduzivel
  src/                        # Scripts de treino e inferencia CLI
    train.py                  # Pipeline de treino (Roboflow + YOLOv8)
    infer.py                  # Inferencia em imagem individual
    runs1/                    # Artefactos de treino do modelo 1
    runs2/                    # Artefactos de treino do modelo 2
  app/                        # Aplicacao de demonstracao (Streamlit)
    app.py                    # Aplicacao web interativa
    requirements.txt          # Dependencias Python
    README.md                 # Instrucoes de execucao da app
```

## Instalacao

### Requisitos

- Python 3.10+
- pip

### Setup

```bash
# 1. Clonar o repositorio
git clone <url-do-repositorio>
cd wheel-components-inspector

# 2. Criar ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows

# 3. Instalar dependencias
pip install -r app/requirements.txt

# 4. Configurar chave API do Roboflow
cp .env_template .env
# Editar .env e inserir a chave
```

## Workflow

### 1. Download do Dataset e Treino

```bash
cd src

# Apenas download do dataset
python train.py \
  --workspace "jose-barbosa-rdg0j" \
  --project "wheel-components-inspector" \
  --version 6 \
  --download-only

# Treino completo
python train.py \
  --workspace "jose-barbosa-rdg0j" \
  --project "wheel-components-inspector" \
  --version 6 \
  --epochs 100 \
  --batch 8 \
  --device 0
```

Argumentos disponiveis: `--epochs`, `--imgsz`, `--batch`, `--device` (`cpu`, `0`, `mps`), `--run-name`, `--download-only`.

### 2. Monitorizacao com TensorBoard

```bash
# Modelo 1
tensorboard --logdir src/runs1/train
# Modelo 2
tensorboard --logdir src/runs2/train
# Aceder a http://localhost:6006
```

### 3. Preservacao dos Artefactos

Apos o treino, copiar o melhor modelo para a pasta do modelo correspondente:

```bash
# Modelo 1
cp src/runs1/train/wheel-inspector-model/weights/best.pt modelos/modelo1/weights.pt

# Modelo 2
cp src/runs2/train/wheel-inspector-model/weights/best.pt modelos/modelo2/weights.pt
```

### 4. Inferencia CLI

```bash
cd src
# Usando modelo 1
python infer.py --model ../modelos/modelo1/weights.pt --image ../test-image.jpg
# Usando modelo 2
python infer.py --model ../modelos/modelo2/weights.pt --image ../test-image.jpg
```

Gera um ficheiro JSON com as detecoes e uma imagem anotada no diretorio `output/`.

### 5. Aplicacao de Demonstracao

```bash
cd app
python -m streamlit run app.py
```

Funcionalidades:
- Upload de imagem para inspecao
- Selecao de modelo e ajuste do limiar de confianca
- Visualizacao lado a lado (original vs. bounding boxes)
- Tabela de detecoes e output JSON
- Exportacao de resultados
- Comparacao de modelos lado a lado
- Historico de inferencias

## Modelos

Foram treinados dois modelos com arquitetura YOLOv8m (medium), pesos pre-treinados no COCO:

| Metrica | Modelo 1 (v1.0) | Modelo 2 (v2.0) |
|---------|-----------------|-----------------|
| mAP@0.5 | 0.8564 | 0.8878 |
| mAP@0.5:0.95 | 0.7116 | 0.7207 |
| Precision | 0.8616 | 0.9113 |
| Recall | 0.8939 | 0.9041 |

O Modelo 2 foi treinado com imagens adicionais de alta qualidade focadas em parafusos, resultando em melhoria significativa na detecao desta classe (Precision: 0.64 → 0.76, Recall: 0.68 → 0.71).

Detalhes completos:
- [`modelos/modelo1/model_card.md`](modelos/modelo1/model_card.md) | [`model_manifest.json`](modelos/modelo1/model_manifest.json)
- [`modelos/modelo2/model_card.md`](modelos/modelo2/model_card.md) | [`model_manifest.json`](modelos/modelo2/model_manifest.json)

## Dataset

- **Fonte**: 200 imagens recolhidas pelo grupo, etiquetadas no Roboflow
- **Formato**: YOLOv8 (bounding boxes normalizadas)
- **Classes**: 3 (jante, parafuso, roda)
- **Split base**: 70% treino / 20% validacao / 10% teste
- **Split efetivo** (apos augmentation no Roboflow aplicada ao treino): 87.5% treino / 8.3% validacao / 4.2% teste (480 imagens)

Detalhes em [`dataset/dataset.md`](dataset/dataset.md).

## Tecnologias

| Componente | Tecnologia |
|------------|-----------|
| Detecao de objetos | [Ultralytics YOLOv8](https://docs.ultralytics.com/) |
| Gestao de dados | [Roboflow](https://roboflow.com/) |
| Interface web | [Streamlit](https://streamlit.io/) |
| Monitorizacao | [TensorBoard](https://www.tensorflow.org/tensorboard) |
| Visao computacional | OpenCV, Pillow |

## Autores

- Jose Barbosa
- Pedro Rocha
- Agostinho Ferreira

**Unidade Curricular**: Inteligencia Artificial, ESTG - P.Porto, 2025/2026
