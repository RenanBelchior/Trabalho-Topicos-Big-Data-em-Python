import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Sistema de Classificação", layout="centered")
st.title("🌿 Sistema de Classificação de Plantas")

# Carregamento dos dados
df = pd.read_csv("https://raw.githubusercontent.com/RenanBelchior/Trabalho-Topicos-Big-Data-em-Python/main/historico_vendas.csv")
colunas_entrada = ['Preco', 'Quantidade']
coluna_saida = 'Demanda'

X = df[colunas_entrada]
y = df[coluna_saida]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicialização do estado
if 'tela' not in st.session_state:
    st.session_state.tela = 'menu_principal'
if 'historico' not in st.session_state:
    st.session_state.historico = []

# Funções de classificação
def testar_arvore():
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.session_state.historico.append({'modelo': 'Árvore de Decisão', 'acuracia': acc})
    return modelo, acc

def testar_svm():
    modelo = SVC(kernel='linear')
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.session_state.historico.append({'modelo': 'SVM', 'acuracia': acc})
    return modelo, acc

# Telas
if st.session_state.tela == "menu_principal":
    st.subheader("Menu Principal")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("1 - Árvore de Decisão"):
            st.session_state.tela = "arvore_menu"
        if st.button("2 - SVM"):
            st.session_state.tela = "svm_menu"
    with col2:
        if st.button("3 - Exibir Desempenho dos Classificadores"):
            st.session_state.tela = "comparativo"

elif st.session_state.tela == "arvore_menu":
    st.subheader("Árvore de Decisão")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("1 - Mostrar desempenho"):
            historico = [h for h in st.session_state.historico if h['modelo'] == 'Árvore de Decisão']
            if historico:
                for i, item in enumerate(historico[::-1], 1):
                    st.write(f"{i}º teste: Acurácia = {item['acuracia']*100:.2f}%")
            else:
                st.info("Nenhum teste registrado ainda.")

        if st.button("2 - Mostrar árvore"):
            modelo, _ = testar_arvore()
            st.write("Visualização da Árvore de Decisão:")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_tree(modelo, filled=True, feature_names=colunas_entrada, class_names=[str(c) for c in sorted(y.unique())], ax=ax)
            st.pyplot(fig)

    with col2:
        if st.button("3 - Fazer nova classificação"):
            _, acc = testar_arvore()
            st.success(f"Classificação realizada. Acurácia = {acc * 100:.2f}%")

        if st.button("4 - Voltar"):
            st.session_state.tela = "menu_principal"

        if st.button("5 - Remover histórico"):
            st.session_state.historico = [h for h in st.session_state.historico if h['modelo'] != 'Árvore de Decisão']
            st.success("Histórico de Árvore de Decisão removido com sucesso.")

elif st.session_state.tela == "svm_menu":
    st.subheader("SVM")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("1 - Mostrar desempenho"):
            historico = [h for h in st.session_state.historico if h['modelo'] == 'SVM']
            if historico:
                for i, item in enumerate(historico[::-1], 1):
                    st.write(f"{i}º teste: Acurácia = {item['acuracia']*100:.2f}%")
            else:
                st.info("Nenhum teste registrado ainda.")

    with col2:
        if st.button("2 - Fazer nova classificação"):
            _, acc = testar_svm()
            st.success(f"Classificação realizada. Acurácia = {acc * 100:.2f}%")

        if st.button("3 - Voltar"):
            st.session_state.tela = "menu_principal"

        if st.button("4 - Remover histórico"):
            st.session_state.historico = [h for h in st.session_state.historico if h['modelo'] != 'SVM']
            st.success("Histórico de SVM removido com sucesso.")

elif st.session_state.tela == "comparativo":
    st.subheader("Comparativo de Desempenhos")
    historico = st.session_state.historico
    acc_arvore = [h['acuracia'] for h in historico if h['modelo'] == 'Árvore de Decisão']
    acc_svm = [h['acuracia'] for h in historico if h['modelo'] == 'SVM']

    melhor_modelo = None
    melhor_acc = 0

    if acc_arvore:
        media_arvore = sum(acc_arvore) / len(acc_arvore)
        st.write(f"Acurácia média Árvore de Decisão: {media_arvore * 100:.2f}%")
        if media_arvore > melhor_acc:
            melhor_acc = media_arvore
            melhor_modelo = "Árvore de Decisão"
    else:
        st.info("Sem dados para Árvore de Decisão.")

    if acc_svm:
        media_svm = sum(acc_svm) / len(acc_svm)
        st.write(f"Acurácia média SVM: {media_svm * 100:.2f}%")
        if media_svm > melhor_acc:
            melhor_acc = media_svm
            melhor_modelo = "SVM"
    else:
        st.info("Sem dados para SVM.")

    if melhor_modelo:
        st.success(f"Melhor desempenho geral: {melhor_modelo} com {melhor_acc * 100:.2f}% de acurácia média")

    if st.button("Remover todo o histórico"):
        st.session_state.historico = []
        st.success("Histórico de todos os classificadores removido com sucesso.")

    if st.button("Voltar ao menu principal"):
        st.session_state.tela = "menu_principal"
