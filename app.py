import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Classificador Inteligente MasterPeças", layout="wide")
st.title("Classificador Inteligente Master Peças")

menu = st.sidebar.radio("Menu Principal", ["Árvore de Decisão", "SVM", "Comparativo", "Limpar Histórico"])

# Carregamento dos dados com cache
@st.cache_data
def carregar_dados():
    df = pd.read_csv("https://raw.githubusercontent.com/RenanBelchior/Trabalho-Topicos-Big-Data-em-Python/main/historico_vendas.csv", encoding='utf-8-sig')
    le = LabelEncoder()
    for c in df.select_dtypes(include='object').columns:
        df[c] = le.fit_transform(df[c])
    return df

df = carregar_dados()
colunas_disponiveis = [col for col in df.columns if col != 'Demanda']

# Inicializa session_state
for chave in ['historico_dt', 'historico_svm']:
    if chave not in st.session_state:
        st.session_state[chave] = []

if 'melhor' not in st.session_state:
    st.session_state.melhor = {'modelo': None, 'acuracia': 0}
if 'modelo_dt' not in st.session_state:
    st.session_state.modelo_dt = None
if 'modelo_svm' not in st.session_state:
    st.session_state.modelo_svm = None
if 'teste_final_dt' not in st.session_state:
    st.session_state.teste_final_dt = 0
if 'teste_final_svm' not in st.session_state:
    st.session_state.teste_final_svm = 0

def exibir_historico(lista):
    if lista:
        for i, acc in enumerate(reversed(lista), 1):
            st.write(f"Treino {len(lista)-i+1}: **{acc*100:.2f}%**")
    else:
        st.info("Nenhum histórico registrado.")

# Árvore de Decisão
if menu == "Árvore de Decisão":
    st.header("🌳 Árvore de Decisão - Menu")

    st.subheader("Selecionar colunas de entrada")
    colunas_selecionadas_dt = st.multiselect("Selecione as colunas de entrada:", colunas_disponiveis, default=['Preco', 'Quantidade'])

    if st.button("Testar Nova Classificação"):
        if colunas_selecionadas_dt:
            X = df[colunas_selecionadas_dt]
            y = df['Demanda']

            X_treino_completo, X_teste_final, y_treino_completo, y_teste_final = train_test_split(X, y, test_size=0.3, random_state=42)
            X_treino_modelo, X_teste_modelo, y_treino_modelo, y_teste_modelo = train_test_split(X_treino_completo, y_treino_completo, test_size=0.3, random_state=42)

            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_treino_modelo, y_treino_modelo)

            acc_treino = accuracy_score(y_teste_modelo, model.predict(X_teste_modelo))
            acc_final = accuracy_score(y_teste_final, model.predict(X_teste_final))

            st.session_state.modelo_dt = model
            st.session_state.historico_dt.append(acc_treino)
            st.session_state.teste_final_dt = acc_final

            if acc_treino > st.session_state.melhor['acuracia']:
                st.session_state.melhor = {'modelo': 'Árvore de Decisão', 'acuracia': acc_treino}

            st.success(f"Teste final (70% dos dados): {acc_final*100:.2f}%")
        else:
            st.warning("Selecione ao menos uma coluna.")

    if st.button("Mostrar Desempenho"):
        st.subheader("Histórico de desempenho (Treino)")
        exibir_historico(st.session_state.historico_dt)

    if st.button("Mostrar Árvore de Decisão"):
        if st.session_state.modelo_dt:
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_tree(st.session_state.modelo_dt, feature_names=colunas_selecionadas_dt, filled=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Treine o modelo antes de visualizar a árvore.")

# SVM
elif menu == "SVM":
    st.header("🔎 SVM - Menu")

    st.subheader("Selecionar colunas de entrada")
    colunas_selecionadas_svm = st.multiselect("Selecione as colunas de entrada:", colunas_disponiveis, default=['Preco', 'Quantidade'], key="svm")

    if st.button("Testar Nova Classificação", key="svm_treinar"):
        if colunas_selecionadas_svm:
            X = df[colunas_selecionadas_svm]
            y = df['Demanda']

            X_treino_completo, X_teste_final, y_treino_completo, y_teste_final = train_test_split(X, y, test_size=0.3, random_state=42)
            X_treino_modelo, X_teste_modelo, y_treino_modelo, y_teste_modelo = train_test_split(X_treino_completo, y_treino_completo, test_size=0.3, random_state=42)

            pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='linear'))])
            pipeline.fit(X_treino_modelo, y_treino_modelo)

            acc_treino = accuracy_score(y_teste_modelo, pipeline.predict(X_teste_modelo))
            acc_final = accuracy_score(y_teste_final, pipeline.predict(X_teste_final))

            st.session_state.modelo_svm = pipeline
            st.session_state.historico_svm.append(acc_treino)
            st.session_state.teste_final_svm = acc_final

            if acc_treino > st.session_state.melhor['acuracia']:
                st.session_state.melhor = {'modelo': 'SVM', 'acuracia': acc_treino}

            st.success(f"Teste final (70% dos dados): {acc_final*100:.2f}%")
        else:
            st.warning("Selecione ao menos uma coluna.")

    if st.button("Mostrar Desempenho", key="svm_dsp"):
        st.subheader("Histórico de desempenho (Treino)")
        exibir_historico(st.session_state.historico_svm)

# Comparativo
elif menu == "Comparativo":
    st.header("📊 Comparativo de Desempenho - Teste Final")
    acc_dt = st.session_state.teste_final_dt
    acc_svm = st.session_state.teste_final_svm
    st.markdown(f"**Árvore de Decisão:** {acc_dt*100:.2f}%")
    st.markdown(f"**SVM:** {acc_svm*100:.2f}%")

    if acc_dt > acc_svm:
        st.success("🔍 Melhor desempenho no teste final: Árvore de Decisão")
    elif acc_svm > acc_dt:
        st.success("🔍 Melhor desempenho no teste final: SVM")
    else:
        st.info("🔍 Desempenho igual ou modelos não treinados.")

# Limpar Histórico
elif menu == "Limpar Histórico":
    st.session_state.historico_dt.clear()
    st.session_state.historico_svm.clear()
    st.session_state.melhor = {'modelo': None, 'acuracia': 0}
    st.session_state.teste_final_dt = 0
    st.session_state.teste_final_svm = 0
    st.success("Histórico geral limpo com sucesso!")
