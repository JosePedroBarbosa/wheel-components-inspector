# Documentacao do Conjunto de Dados (Dataset)

## Definicao de Classes

| ID | Classe     | Descricao |
|----|------------|-----------|
| 0  | `jante`    | Parte central metalica da roda, visivel quando o pneu esta montado. Inclui a face frontal do aro, raios e zona central. |
| 1  | `parafuso` | Pernos ou parafusos que fixam a jante ao cubo do eixo do veiculo. Anotados individualmente. |
| 2  | `roda`     | Conjunto completo de pneu + jante montado no veiculo. Bounding box engloba a totalidade do pneu e da jante. |

## Semantica e Regras de Etiquetagem

### Principio geral
Cada objeto e anotado com uma bounding box retangular que cobre a extensao visivel do objeto, no formato YOLO normalizado`.

### Roda vs Jante
Quando a roda completa esta visivel, gera-se **duas anotacoes sobrepostas** na mesma imagem:
- `roda` — bounding box que engloba o pneu completo (aro + borracha).
- `jante` — bounding box mais pequena, centrada na face metalica visivel do aro.

Quando apenas a jante e visivel sem pneu (ex.: jante desmontada), anota-se apenas `jante`.

### Parafusos
- Anotados individualmente com `parafuso` quando sao visualmente distinguiveis (contorno claro, contraste suficiente).
- **Nao sao anotados** se mais de 70% do parafuso estiver ocluido ou se o tamanho em pixeis for inferior a ~10x10 px apos redimensionamento para 640x640.
- Em imagens de distancia, parafusos podem nao ser anotados por falta de resolucao suficiente.

### Oclusao e corte de margem
- Objetos **parcialmente ocultos** sao anotados se pelo menos **30%** da area estiver visivel.
- Objetos **cortados pela margem** da imagem sao anotados se mais de **50%** do objeto estiver dentro da imagem.
- A bounding box cobre apenas a parte visivel/dentro da imagem.

### Casos ambiguos documentados
| Situacao | Decisao tomada |
|----------|---------------|
| Jante coberta por calota | Anota-se `roda`; nao se anota `jante` (nao visivel). |
| Reflexo de jante em superficie | Nao anotado (nao e objeto real). |
| Multiplas rodas em frame | Cada roda e anotada independentemente com as suas proprias etiquetas. |
| Parafuso semi-ocluido (~50%) | Caso a caso; se o contorno for identificavel, anota-se. |

## Recolha de Imagens

- **Total de imagens**: 200 (antes de augmentation).
- **Dispositivos**: Smartphones (cameras traseiras).
- **Condicoes**: Iluminacao natural diurna e iluminacao artificial interior (garagem/oficina).
- **Angulos**: Frontal, lateral, diagonal e de cima, a distancias variadas (0.5 m a 3 m).
- **Variabilidade**: Diferentes marcas/modelos de veiculos, cores de jantes, tipos de parafusos e materiais (aco, liga leve, cromo).

## Distribuicao do Dataset

| Split      | Imagens | Percentagem |
|------------|---------|-------------|
| Train      | 420     | 88%         |
| Validation | 40      | 8%          |
| Test       | 20      | 4%          |
| **Total**  | **480** | **100%**    |

> Valores exportados diretamente do Roboflow (versao atual do projeto).

## Pre-processamento e Augmentations

### Pre-processamento (Roboflow)
| Operacao | Detalhe |
|----------|---------|
| Auto-orientacao | Aplicada (corrige rotacao EXIF dos smartphones). |
| Redimensionamento | Stretch para **640 × 640** px. |

### Augmentations (Roboflow — aplicadas apenas ao split de treino)
Cada imagem de treino original gera **3 versoes aumentadas**, perfazendo efetivamente ~1260 exemplos de treino.

| Augmentation | Parametros |
|--------------|-----------|
| Flip | Horizontal |
| Crop | Zoom minimo 0%, zoom maximo 15% |
| Rotacao | Entre -15° e +15° |
| Saturacao | Entre -20% e +20% |
| Brilho | Entre -25% e +25% |

> As augmentations de validacao e teste **nao** sao aumentadas para garantir avaliacao justa.

## Questoes e Limitacoes Conhecidas

- **Iluminacao**: Faltam imagens noturnas ou com pouca iluminacao (ex.: garagem escura, contraluz).
- **Proximidade extrema**: Cobertura limitada de angulos muito fechados (close-up a menos de 20 cm), onde parafusos ocupam a maior parte do frame.
- **Jantes atipicas**: Jantes de design nao convencional (ex.: racing rims com raios muito finos, jantes full-face) podem estar sub-representadas.
- **Parafusos em jantes escuras**: Baixo contraste dificulta a anotacao e pode causar falsos negativos no modelo.
- **Redimensionamento com distorcao**: O uso de Stretch (em vez de letterbox) para 640×640 introduz distorcao geometrica em imagens com razao de aspeto diferente de 1:1; pode afectar a precisao das bounding boxes em imagens originais nao quadradas.
