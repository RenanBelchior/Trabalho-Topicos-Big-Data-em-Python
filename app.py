import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Classificação de Plantas", layout="wide")
st.title("🌱 Software para Classificação de Plantas")

# Carrega e prepara os dados
url_dados = "https://raw.githubusercontent.com/RenanBelchior/Trabalho-Topicos-Big-Data-em-Python/main/historico_vendas.csv"
df = pd.read_csv(url_dados, encoding='utf-8-sig')

colunas_entrada = ['Preco', 'Quantidade']
coluna_saida = 'Demanda'

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

X = df[colunas_entrada]
y = df[coluna_saida]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializa estado
if "tela" not in st.session_state:
    st.session_state.tela = "menu_principal"

if "historico" not in st.session_state:
    st.session_state.historico = []

if "melhor_modelo" not in st.session_state:
    st.session_state.melhor_modelo = {'modelo': None, 'acuracia': 0}


# Funções
def testar_arvore():
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.session_state.historico.append({'modelo': 'Árvore de Decisão', 'acuracia': acc})
    if acc > st.session_state.melhor_modelo['acuracia']:
        st.session_state.melhor_modelo = {'modelo': 'Árvore de Decisão', 'acuracia': acc}
    return modelo, acc


def testar_svm():
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="linear"))
    ])
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.session_state.historico.append({'modelo': 'SVM', 'acuracia': acc})
    if acc > st.session_state.melhor_modelo['acuracia']:
        st.session_state.melhor_modelo = {'modelo': 'SVM', 'acuracia': acc}
    return modelo, acc


# Menu Principal
if st.session_state.tela == "menu_principal":
    st.subheader("Menu Principal")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("1 - Árvore de Decisão"):
            st.session_state.tela = "arvore_menu"
    with col2:
        if st.button("2 - SVM"):
            st.session_state.tela = "svm_menu"
    with col3:
        if st.button("3 - Exibir Desempenho dos Classificadores"):
            st.session_state.tela = "comparativo"

# Submenu Árvore de Decisão
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

# Submenu SVM
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

# Comparativo entre os classificadores
elif st.session_state.tela == "comparativo":
    st.subheader("📊 Comparativo de Desempenho dos Classificadores")
    historico_arvore = [h for h in st.session_state.historico if h['modelo'] == 'Árvore de Decisão']
    historico_svm = [h for h in st.session_state.historico if h['modelo'] == 'SVM']

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Árvore de Decisão:**")
        if historico_arvore:
            for i, item in enumerate(historico_arvore[::-1], 1):
                st.write(f"{i}º teste: Acurácia = {item['acuracia']*100:.2f}%")
        else:
            st.info("Nenhum teste da árvore ainda.")

    with col2:
        st.markdown("**SVM:**")
        if historico_svm:
            for i, item in enumerate(historico_svm[::-1], 1):
                st.write(f"{i}º teste: Acurácia = {item['acuracia']*100:.2f}%")
        else:
            st.info("Nenhum teste do SVM ainda.")

    st.markdown("---")
    melhor = st.session_state.melhor_modelo
    if melhor['modelo']:
        st.success(f"🏆 Melhor desempenho até agora: **{melhor['modelo']}** com acurácia de **{melhor['acuracia']*100:.2f}%**")
    else:
        st.warning("Nenhum classificador foi testado ainda.")

    if st.button("Voltar"):
        st.session_state.tela = "menu_principal"
