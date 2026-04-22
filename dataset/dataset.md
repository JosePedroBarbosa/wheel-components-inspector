# Documentação do Dataset

## 1. Classes e Nomenclatura

| ID | Classe     | Descrição |
|----|------------|-----------|
| 0  | `jante`    | Parte central metálica da roda, visível quando o pneu está montado. Inclui a face frontal do aro, raios e zona central. |
| 1  | `parafuso` | Pernos ou parafusos que fixam a jante ao cubo do eixo do veículo. Anotados individualmente. |
| 2  | `roda`     | Conjunto completo de pneu + jante montado no veículo. A bounding box engloba a totalidade do pneu e da jante. |

## 2. Interpretação das Etiquetas

### Sobreposição de classes

Quando uma roda completa está visível, são geradas duas anotações sobrepostas: `roda` (pneu + aro) e `jante` (apenas a parte metálica). Esta sobreposição é intencional e reflete a hierarquia semântica das classes.

### Regras de visibilidade e oclusão

- **Parafusos**: anotados individualmente quando visualmente distinguíveis. Não são anotados se mais de 70% do objeto estiver ocluído ou se o tamanho for muito reduzido.
- **Oclusão parcial**: um objeto é anotado se pelo menos 30% da sua área estiver visível.
- **Corte de margem**: objetos cortados pela borda da imagem são incluídos se mais de 50% do objeto estiver dentro do frame.

## 3. Problemas Conhecidos

### Casos ambíguos

- **Parafusos com baixa iluminação**: quando a cor do parafuso se confunde com a da jante em condições de pouca luz, o parafuso não é anotado. Pode originar falsos negativos nessas condições.

### Limitações do dataset

- **Iluminação**: escassez de imagens em condições noturnas ou com contraluz acentuado.
- **Baixo contraste**: em jantes escuras, a deteção de parafusos é dificultada, aumentando a probabilidade de etiquetas em falta.
- **Sub-representação**: jantes de design atípico ou com oclusão por lama/sujidade estão menos representadas no dataset.
- **Ângulo de captura**: o dataset é maioritariamente composto por imagens frontais da roda, com poucas capturas em ângulos laterais, o que pode afetar a generalização do modelo a perspetivas menos representadas.