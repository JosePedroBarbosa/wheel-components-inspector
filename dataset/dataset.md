# Documentação do Conjunto de Dados (Dataset)

Neste ficheiro (`dataset.md`), o grupo deverá documentar as propriedades dos dados de imagens utilizados pelo projeto.

## Definição de Classes

* `roda` - O pneu com jante completa agrupado.
* `jante` - (ex. a parte central de metal).
* `parafuso` - (ex. pernos que unem a jante ao eixo).

## Semântica e Regras de Etiquetagem

* (Exemplo a preencher pelo grupo): *Decidimos não anotar parafusos caso estejam tapados em mais de 70%.*
* (Exemplo a preencher pelo grupo): *Para fotografias escuras, adicionámos augmentation Brightness via Roboflow na build version 2.*
* ...

## Questões e Limitações Conhecidas

* *(Exemplo: Faltaram-nos fotos de ângulos muito fechados de alguns dos pneus)*

LER NO RELATORIO ESTA PARTE: 

Roboflow, este pode ser exportado sob a forma de um ficheiro .zip. O grupo deve certificar-se de que, no 
momento da exportação, o ficheiro .zip contém divisões explícitas de treino/validação/teste (train/val/test). 
Deve também ser incluído um ficheiro dataset.md que descreva: 
• As classes e a respetiva nomenclatura 
• A forma de interpretar as etiquetas (o seu significado e semântica) 
• Quaisquer problemas conhecidos (casos ambíguos, etiquetas em falta) 