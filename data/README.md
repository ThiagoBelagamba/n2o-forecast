Coloque aqui o arquivo `Dados-nacionais-13.0.xlsx` (SEEG Coleção 13, ~136 MB). Ele não entra no Git.

Fonte: SEEG / Observatório do Clima  
Download: https://seeg.eco.br/dados/  
Plataforma: https://plataforma.seeg.eco.br/

O `main.py` lê o xlsx, filtra `N2O (t)` + `Emissão`, agrega UF/bioma ao nacional e grava o cache `n2o_nacional_longo.csv` (também local).

Série: **1970–2024**. Split padrão: treino até 2019, teste 2020–2024.
