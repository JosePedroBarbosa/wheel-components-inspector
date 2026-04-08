# Documentacao do Conjunto de Dados (Dataset)

## Definicao de Classes

| ID | Classe     | Descricao |
|----|------------|-----------|
| 0  | `jante`    | Parte central metalica da roda (rim/hub), visivel quando o pneu esta montado. |
| 1  | `parafuso` | Pernos ou parafusos que fixam a jante ao eixo do veiculo. |
| 2  | `roda`     | Conjunto completo de pneu + jante montado no veiculo. |

## Semantica e Regras de Etiquetagem

- Cada objeto e anotado com uma bounding box que cobre a sua extensao visivel.
- **Roda vs Jante**: Quando a roda completa (pneu + jante) e visivel, a roda recebe a anotacao `roda` e a parte central visivel da jante recebe separadamente a anotacao `jante`. Ou seja, uma roda completa gera tipicamente 2 anotacoes sobrepostas (roda + jante).
- **Parafusos**: Apenas anotados individualmente quando sao visualmente distinguiveis. Parafusos tapados em mais de 70% por oclusao nao sao anotados.
- **Oclusao parcial**: Objetos parcialmente ocultos sao anotados se pelo menos 30% do objeto estiver visivel.
- **Objetos cortados pela margem**: Anotados se mais de 50% do objeto estiver dentro da imagem.

## Recolha de Imagens

- **Dispositivos**: Smartphones (cameras traseiras).
- **Condicoes**: Iluminacao natural diurna e iluminacao artificial interior (garagem/oficina).
- **Angulos**: Frontal, lateral, diagonal e de cima, a distancias variadas (0.5m a 3m).
- **Variabilidade**: Diferentes marcas/modelos de veiculos, cores de jantes, tipos de parafusos.

## Distribuicao do Dataset

| Split      | Imagens | Percentagem |
|------------|---------|-------------|
| Train      | ~140    | 70%         |
| Validation | ~40     | 20%         |
| Test       | ~20     | 10%         |

> Nota: valores aproximados; os numeros exatos dependem da versao final exportada do Roboflow.

## Augmentations

- **Pre-processamento (Roboflow)**: Auto-orientacao, redimensionamento para 640x640.
- **Augmentations (treino local via YOLO)**: HSV shift, horizontal flip (50%), scale (50%), mosaic.
- Nao foram aplicadas augmentations adicionais na exportacao do Roboflow.

## Questoes e Limitacoes Conhecidas

- Faltam imagens noturnas / com pouca iluminacao.
- Cobertura limitada de angulos muito fechados (close-up extremo a <20cm).
- Jantes de design muito diferente do habitual (ex: racing rims) podem estar sub-representadas.
- Parafusos em jantes escuras podem ser dificeis de distinguir do fundo.
