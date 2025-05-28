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
st.set_page_config(page_title="Classificador Inteligente Autopeças", layout="wide")

st.title("Classificador Inteligente Autopeças")

# Menu lateral
st.sidebar.title("Menu Principal")
menu = st.sidebar.radio("", ["Árvore de Decisão", "SVM", "Comparativo", "Limpar Histórico"])

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

# Função para limpar histórico
def limpar_historico():
    st.session_state.historico_dt.clear()
    st.session_state.historico_svm.clear()
    st.session_state.melhor = {'modelo': None, 'acuracia': 0}
    st.success("Histórico geral limpo com sucesso!")

# Navegação baseada na seleção lateral
if menu == "Árvore de Decisão":
    st.header("🌳 Árvore de Decisão")
    if st.button("Treinar Modelo"):
        modelo = DecisionTreeClassifier(random_state=42)
        modelo.fit(X_train, y_train)
        acc = accuracy_score(y_test, modelo.predict(X_test))
        st.session_state.modelo_dt = modelo
        st.session_state.historico_dt.append(acc)
        if acc > st.session_state.melhor['acuracia']:
            st.session_state.melhor = {'modelo': 'Árvore de Decisão', 'acuracia': acc}
        st.success("Classificador Treinado com Sucesso")

    st.markdown("### Histórico de Desempenho")
    if st.session_state.historico_dt:
        for i, acc in enumerate(reversed(st.session_state.historico_dt), 1):
            st.write(f"Teste {len(st.session_state.historico_dt)-i+1}: **{acc * 100:.2f}%**")
    else:
        st.info("Nenhum histórico registrado.")

    if st.button("Mostrar Árvore de Decisão"):
        if st.session_state.modelo_dt:
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_tree(st.session_state.modelo_dt, feature_names=col_auxiliares,
                      class_names=True, filled=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Treine o modelo antes de visualizar a árvore.")

elif menu == "SVM":
    st.header("🔎 SVM (Máquina de Vetores de Suporte)")
    if st.button("Treinar Modelo"):
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
        st.success("Classificador Treinado com Sucesso")

    st.markdown("### Histórico de Acurácia")
    if st.session_state.historico_svm:
        for i, acc in enumerate(reversed(st.session_state.historico_svm), 1):
            st.write(f"Teste {len(st.session_state.historico_svm)-i+1}: **{acc * 100:.2f}%**")
    else:
        st.info("Nenhum histórico registrado.")

elif menu == "Comparativo":
    st.header("📊 Comparativo de Desempenho")
    acc_dt = max(st.session_state.historico_dt) if st.session_state.historico_dt else 0
    acc_svm = max(st.session_state.historico_svm) if st.session_state.historico_svm else 0

    st.markdown(f"**Árvore de Decisão:** {acc_dt * 100:.2f}%")
    st.markdown(f"**SVM:** {acc_svm * 100:.2f}%")

    if acc_dt > acc_svm:
        st.success("🔍 Melhor desempenho: Árvore de Decisão")
    elif acc_svm > acc_dt:
        st.success("🔍 Melhor desempenho: SVM")
    else:
        st.info("🔍 Ambos os classificadores possuem desempenho igual ou ainda não foram testados.")

elif menu == "Limpar Histórico":
    limpar_historico()
