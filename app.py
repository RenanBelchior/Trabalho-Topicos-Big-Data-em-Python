import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Previsão de Demanda - Autopeças", layout="wide")
st.title("📦 Sistema de Previsão de Demanda de Autopeças")

# Leitura dos dados
dados_url = "https://raw.githubusercontent.com/RenanBelchior/Trabalho-Topicos-Big-Data-em-Python/main/historico_vendas.csv"
df = pd.read_csv(dados_url, encoding='utf-8-sig')

# Pré-processamento
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

col_auxiliares = ['Preco', 'Quantidade']
col_saida = 'Demanda'

X = df[col_auxiliares]
y = df[col_saida]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Estado da sessão
if 'historico_dt' not in st.session_state:
    st.session_state.historico_dt = []
if 'historico_svm' not in st.session_state:
    st.session_state.historico_svm = []
if 'melhor' not in st.session_state:
    st.session_state.melhor = {'modelo': None, 'acuracia': 0}
if 'modelo_dt' not in st.session_state:
    st.session_state.modelo_dt = None
if 'modelo_svm' not in st.session_state:
    st.session_state.modelo_svm = None
if 'menu' not in st.session_state:
    st.session_state.menu = "inicio"

# Tela de introdução
if st.session_state.menu == "inicio":
    col_esq, col_centro, col_dir = st.columns([1, 2, 1])
    with col_centro:
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZGlpODdhMGtmYmZwbjM1eDhkcjFqNjN6bXF1ZGZvaThlNnNrcDV2dSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/J1kWQjF5b0M6W9Vi9n/giphy.gif", width=400)
        st.markdown("""
        ## Bem-vindo ao Sistema de Previsão de Demanda de Autopeças 🚗🔧
        Este sistema utiliza algoritmos de aprendizado de máquina para prever a demanda de produtos com base em seu preço e quantidade vendida.  
        Você pode treinar classificadores, visualizar seus desempenhos e comparar resultados entre diferentes modelos.
        """)
        if st.button("Ir para o Menu Principal"):
            st.session_state.menu = None

# Menu principal
if st.session_state.menu is None:
    st.subheader("Menu Principal")
    col1, col2, col3, col4 = st.columns(4)

    if col1.button("Árvore de Decisão"):
        st.session_state.menu = "arvore"
    elif col2.button("SVM"):
        st.session_state.menu = "svm"
    elif col3.button("Exibir Desempenho dos Classificadores"):
        st.session_state.menu = "comparativo"
    elif col4.button("Limpar Histórico Geral"):
        st.session_state.historico_dt.clear()
        st.session_state.historico_svm.clear()
        st.session_state.melhor = {'modelo': None, 'acuracia': 0}
        st.success("Histórico geral limpo com sucesso!")

# Submenu Árvore de Decisão
if st.session_state.menu == "arvore":
    st.subheader("Menu - Árvore de Decisão")
    if st.button("Fazer Nova Classificação"):
        modelo = DecisionTreeClassifier(random_state=42)
        modelo.fit(X_train, y_train)
        acc = accuracy_score(y_test, modelo.predict(X_test))
        st.session_state.modelo_dt = modelo
        st.session_state.historico_dt.append(acc)
        if acc > st.session_state.melhor['acuracia']:
            st.session_state.melhor = {'modelo': 'Árvore de Decisão', 'acuracia': acc}
        st.success("Classificador treinado com sucesso")

    if st.button("Mostrar Desempenho"):
        if st.session_state.historico_dt:
            for i, acc in enumerate(reversed(st.session_state.historico_dt)):
                st.markdown(f"**Teste {len(st.session_state.historico_dt)-i}:** {acc * 100:.2f}%")
        else:
            st.warning("Nenhum desempenho registrado ainda.")

    if st.button("Mostrar Árvore"):
        if st.session_state.modelo_dt:
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_tree(st.session_state.modelo_dt, feature_names=col_auxiliares, class_names=True, filled=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Treine o modelo primeiro usando 'Fazer Nova Classificação'.")

    if st.button("Remover Histórico da Árvore de Decisão"):
        st.session_state.historico_dt.clear()
        st.success("Histórico da Árvore de Decisão limpo.")

    if st.button("Voltar ao Menu Principal"):
        st.session_state.menu = None

# Submenu SVM
if st.session_state.menu == "svm":
    st.subheader("Menu - SVM")
    if st.button("Fazer Nova Classificação"):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='linear'))
        ])
        pipeline.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        st.session_state.modelo_svm = pipeline
        st.session_state.historico_svm.append(acc)
        if acc > st.session_state.melhor['acuracia']:
            st.session_state.melhor = {'modelo': 'SVM', 'acuracia': acc}
        st.success("Classificador treinado com sucesso")

    if st.button("Mostrar Desempenho"):
        if st.session_state.historico_svm:
            for i, acc in enumerate(reversed(st.session_state.historico_svm)):
                st.markdown(f"**Teste {len(st.session_state.historico_svm)-i}:** {acc * 100:.2f}%")
        else:
            st.warning("Nenhum desempenho registrado ainda.")

    if st.button("Remover Histórico do SVM"):
        st.session_state.historico_svm.clear()
        st.success("Histórico do SVM limpo.")

    if st.button("Voltar ao Menu Principal"):
        st.session_state.menu = None

# Comparativo de classificadores
if st.session_state.menu == "comparativo":
    st.subheader("📊 Comparativo de Desempenho")
    acc_dt = max(st.session_state.historico_dt) if st.session_state.historico_dt else 0
    acc_svm = max(st.session_state.historico_svm) if st.session_state.historico_svm else 0

    st.markdown(f"**Árvore de Decisão:** {acc_dt * 100:.2f}%")
    st.markdown(f"**SVM:** {acc_svm * 100:.2f}%")

    if acc_dt > acc_svm:
        st.success("🔍 Melhor desempenho: Árvore de Decisão")
    elif acc_svm > acc_dt:
        st.success("🔍 Melhor desempenho: SVM")
    else:
        st.info("🔍 Ambos os classificadores possuem desempenho igual ou não foram testados ainda.")

    if st.button("Voltar ao Menu Principal"):
        st.session_state.menu = None
