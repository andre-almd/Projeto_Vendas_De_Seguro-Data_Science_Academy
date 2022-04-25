import streamlit as st
import requests 
import numpy as np

st.title('Predição de Vendas de Seguro')
st.write('Aplicativo personalizado do Projeto de deploy no Curso Deep Learning II da Data Science Academy.')
st.write('')
st.write('')

st.sidebar.header('Insira os dados de entrada aqui.')

media_aval_cliente = st.sidebar.number_input('Avaliação média dos clientes',
                                    min_value=0., max_value=5.0)

seg_int = st.sidebar.selectbox('Possui seguro internacional?', ('Sim', 'Não'))
if seg_int == 'Sim':
    seg_int = 1
else:
    seg_int = 0

cob_dent = st.sidebar.selectbox('Possui cobertura dentária?', ('Sim', 'Não'))
if cob_dent == 'Sim':
    cob_dent = 1
else:
    cob_dent = 0

cob_maior_65 = st.sidebar.selectbox('Possui cobertura para maiores de 65 anos?', ('Sim', 'Não'))
if cob_maior_65 == 'Sim':
    cob_maior_65 = 1
else:
    cob_maior_65 = 0

remb_desp_med = st.sidebar.selectbox('Possui reembolso para despesas médicas?', ('Sim', 'Não'))
if remb_desp_med == 'Sim':
    remb_desp_med = 1
else:
    remb_desp_med = 0

cob_esp_rad = st.sidebar.selectbox('Possui cobertura para esportes radicais?', ('Sim', 'Não'))
if cob_esp_rad == 'Sim':
    cob_esp_rad = 1
else:
    cob_esp_rad = 0

cap_seg = st.sidebar.selectbox('Possui capital segurado?', ('Sim', 'Não'))
if cap_seg == 'Sim':
    cap_seg = 1
else:
    cap_seg = 0

inc_bonus = st.sidebar.selectbox('Inclui bônus?', ('Sim', 'Não'))
if inc_bonus == 'Sim':
    inc_bonus = 1
else:
    inc_bonus = 0

valor_unitario = st.sidebar.number_input('Valor unitário do seguro', min_value=0.)

make_pred = st.button('Realizar predição')

if make_pred:
    resp = requests.post("http://127.0.0.1:5000/predict", data={'data':[media_aval_cliente,
                                                                    seg_int,
                                                                    cob_dent,
                                                                    cob_maior_65,
                                                                    remb_desp_med,
                                                                    cob_esp_rad,
                                                                    cap_seg,
                                                                    inc_bonus,
                                                                    valor_unitario]})
    
    prediction = resp.json()['Sales amount prediction']
    st.write(f'O total de vendas predito é: {prediction:.2f}')
else:
    st.write('Aguardando predição...')
