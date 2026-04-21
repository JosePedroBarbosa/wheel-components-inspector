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
  dataset/                  # Imagens e anotacoes (formato YOLOv8)
    train/images/ labels/
    valid/images/ labels/
    test/images/  labels/
    data.yaml               # Configuracao do dataset (Roboflow)
    dataset.md              # Documentacao do dataset
  modelos/                  # Artefactos do modelo treinado
    weights.pt              # Pesos do modelo (apos treino)
    model_card.md           # Documentacao legivel por humanos
    model_manifest.json     # Metadados estruturados (legivel por maquina)
    train_config.yaml       # Configuracao de treino reproduzivel
  src/                      # Scripts de treino e inferencia CLI
    train.py                # Pipeline de treino (Roboflow + YOLOv8)
    infer.py                # Inferencia em imagem individual
  app/                      # Aplicacao de demonstracao (Streamlit)
    app.py                  # Aplicacao web interativa
    requirements.txt        # Dependencias Python
    README.md               # Instrucoes de execucao da app
  tutorial_yolo.ipynb       # Notebook de referencia do professor
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
  --version 1 \
  --download-only

# Treino completo
python train.py \
  --workspace "jose-barbosa-rdg0j" \
  --project "wheel-components-inspector" \
  --version 1 \
  --epochs 100 \
  --batch 16 \
  --device cpu
```

Argumentos disponiveis: `--epochs`, `--imgsz`, `--batch`, `--device` (`cpu`, `0`, `mps`), `--run-name`, `--download-only`.

### 2. Monitorizacao com TensorBoard

```bash
tensorboard --logdir src/runs/train
# Aceder a http://localhost:6006
```

### 3. Preservacao dos Artefactos

Apos o treino, copiar o melhor modelo para a pasta de entrega:

```bash
cp src/runs/train/wheel-inspector-model/weights/best.pt modelos/weights.pt
```

O ficheiro `train_config.yaml` e gerado automaticamente pelo script de treino.

### 4. Inferencia CLI

```bash
cd src
python infer.py --model ../modelos/weights.pt --image ../test-image.jpg
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
- Comparacao de modelos (quando existem 2+)
- Historico de inferencias

## Modelo

- **Arquitetura**: YOLOv8m (medium) com pesos pre-treinados
- **Imagem de entrada**: 640x640 RGB
- **Augmentations**: HSV shift, horizontal flip, scale, mosaic
- **Early stopping**: patience=50 epocas

Detalhes completos em [`modelos/model_card.md`](modelos/model_card.md) e [`modelos/model_manifest.json`](modelos/model_manifest.json).

## Dataset

- **Fonte**: Imagens recolhidas pelo grupo, etiquetadas no Roboflow
- **Formato**: YOLOv8 (bounding boxes normalizadas)
- **Classes**: 3 (jante, parafuso, roda)
- **Split**: 70% treino / 20% validacao / 10% teste

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