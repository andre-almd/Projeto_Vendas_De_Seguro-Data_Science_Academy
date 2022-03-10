# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:45:29 2022

@author: André Almeida
"""

'''
# ------------------ Módulo para testar a execução do deploy com flask ------------------
'''

import requests 

# Dados de entrada
media_aval_cliente = 2.5
seguro_internacional = 1
cobertura_dentaria = 0
cobertura_maior_65 = 1
reembolso_despesas_medicas = 0
cobertura_esportes_radicais = 1
capital_segurado = 0
inclui_bonus = 0
valor_unitario = 59.99

# Enviando a requisição para a predição do total de vendas do seguro.
resp = requests.post("http://127.0.0.1:5000/predict", data={'data':[media_aval_cliente,
                                                                    seguro_internacional,
                                                                    cobertura_dentaria,
                                                                    cobertura_maior_65,
                                                                    reembolso_despesas_medicas,
                                                                    cobertura_esportes_radicais,
                                                                    capital_segurado,
                                                                    inclui_bonus,
                                                                    valor_unitario]})

print(resp.text)
