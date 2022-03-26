# Projeto_Vendas_De_Seguro-Data_Science_Academy
Código personalizado do Projeto de deploy no Curso Deep Learning II da Data Science Academy.

## Sobre o desenvolvimento
Neste repositório disponibilizo meu código personalizado para o projeto de previsão do **total de vendas de seguros de viagem** de uma empresa de seguros.

Refiz o projeto da DSA usando Pytorch e uma aplicação Flask para o deploy do modelo treinado.

Durante o desenvolvimento foram utilizadas funcionalidades para criação de dataset personalizado no Pytorch, uso de tensorboard para registrar os modelos e métricas do Pytorch durante o processo de treino, modularização de código, salvamento do scaler do sklearn com dump, etc.

## Sobre o dataset
O	dataset	contém	informações	sobre	o	total	de vendas e o valor unitário do seguro, além de informações relativas ao seguro, como por exemplo se há cobertura para maiores de 65 anos, se o seguro sobre	despesas dentárias,	se o seguro cobre esportes radicais, se o seguro é internacional, etc.

**Inputs:**
media_aval_cliente,
seguro_internacional,
cobertura_dentaria,
cobertura_maior_65,
reembolso_despesas_medicas,
cobertura_esportes_radicais,
capital_segurado,
inclui_bonus,
valor_unitario

**Output:**
total_vendas

## Sobre os arquivos do projeto:

**modelo_pytorch.py** é o arquivo de treinamento do modelo. 
Nele está implementado todo o processo de construção do dataset, criação, treinamento, validação e gravação do modelo no disco. 

**loadNetwork.py** é o módulo que irá carregar o modelo na memória durante a etapa de inferência.

**app.py** é o código responsável pela criação da aplicação para deploy do modelo treinado com flask.

**test.py** é o código de teste para realiza a inferência usando uma chamada request.

**test_streamlit.py** é o dódigo de teste com uma interface gráfica usando streamlit.

## Como usar a aplição?

Excute o arquivo app.py para gerar a aplicação com flask e deixar o modelo disponível para inferência. Depois execute o arquivo test_streamlit.py para realizar as predições com a interface gráfica.
