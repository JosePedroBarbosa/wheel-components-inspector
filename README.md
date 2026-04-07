# Wheel Inspector

Solução completa para a deteção automática de rodas, jantes e parafusos com recurso a métodos de Visão Computacional (YOLOv8). Este projeto cumpre os requisitos do Trabalho Prático da unidade curricular de Inteligência Artificial, providenciando as métricas e processos requeridos sob a perspetiva de um pacote industrial.

## Estrutura do Projeto

- `/dataset` - Diretório destinado a armazenar os ficheiros de imagens e anotações obtidos através do Roboflow.
- `/modelos` - Diretório de destino para os modelos finais treinados (`.pt`), bem como o respetivo manifesto (formato JSON) e documentação associada (Model Card).
- `/src` - Módulos de treino e inferência via linha de comandos, contendo a lógica central de extração e modulação de aprendizagem automática.
- `/app` - Aplicação visual interativa (prova de conceito construída em Streamlit) concebida para demonstrar de imediato as capacidades do algoritmo em tempo real.

## Instalação e Configuração Inicial

1. Proceder à instalação das dependências centrais do projeto (listadas em `app/requirements.txt`) e bibliotecas auxiliares de conexão como o Roboflow.
2. Copiar ou renomear o ficheiro provisório `.env_template` para `.env` e inserir a chave de API privativa do Roboflow necessária à extração de ficheiros da Cloud.
   ```bash
   cp .env_template .env
   ```

## Workflow Principal

### 1. Extração do Dataset e Treino do Modelo

O script `train.py` encarrega-se da obtenção automatizada da referida fonte de dados no Roboflow, prosseguindo com o acionamento do processo de treino do motor YOLO arquitetado:

```bash
cd src
python train.py --workspace "nome-da-workspace" --project "foe-bot" --version 2
```

*(Caso se pretenda aprofundar as configurações de treino ou correr iterações em ambientes com diferentes capacidades gráficas, recomenda-se a consulta `python train.py --help` para verificação dos argumentos admitidos, como `--epochs` ou `--device`).*

#### Monitorização Através de TensorBoard
É possível verificar e analisar o desempenho em tempo real das métricas (e.g. *Loss*, precisão convergente) pelo acoplamento gerido sobre o TensorBoard. Durante ou após a finalização temporal do treino, deverá abrir-se um terminal complementar na raiz do projeto contendo a seguinte instrução:
```bash
tensorboard --logdir runs/train/
```
Após iniciação de servidor, as métricas podem ser avaliadas acedendo ao ponto centralizado indicado na interface original (geralmente [http://localhost:6006](http://localhost:6006)).

### 2. Preservação de Artefactos de Entrega

1. Após a concretização da ronda de treino selecionada, o motor gerará o diretório estrutural `runs/train/`. 
2. Acede-se à subcamada `runs/train/.../weights/` para se proceder à cópia do artefacto `best.pt` em direção à diretoria de submissão do trabalho final (`modelos/nome-do-modelo/best.pt`).
3. Posteriormente, deve certificar-se o enriquecimento e atualização manual dos templates textuais fornecidos (`model_card.md` e `model_manifest.json`) registando as observações e debilidades inerentes identificadas de forma individual ou colaborativa pelo grupo.

### 3. Validação Exploratória via Linha de Comandos

A fim de testar analiticamente se o modelo detém as capacidades generalizadoras perante novas observações, a invocação pode ser concretizada explicitamente:

```bash
cd src
python infer.py --model ../modelos/best.pt --image ../test-image.jpg
``` 
Este comando fará compilar um desfecho estático no diretório `/output`, materializado sob a forma de um objeto exportado (`.json`) descrevendo atributos, bem como a réplica da sua matriz renderizando os polígonos correspondentes às anotações sobrepostas (frequentemente denominados *bounding boxes*).

### 4. Demonstração Executável (Front-End)

Para finalização da Prova de Conceito ilustrativa exigida (secção D4):

```bash
cd app
streamlit run app.py
```