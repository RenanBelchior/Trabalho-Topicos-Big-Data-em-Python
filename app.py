import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Classificador Inteligente Autopeças", layout="wide")
st.title("Classificador Inteligente Autopeças")

# Menu lateral do projeto
menu = st.sidebar.radio("Menu Principal", ["Árvore de Decisão", "SVM", "Comparativo", "Limpar Histórico"])

# Carrega os dados e faz o pré-processamento(sem necessitar de upload)
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/RenanBelchior/Trabalho-Topicos-Big-Data-em-Python/main/historico_vendas.csv", encoding='utf-8-sig')
    le = LabelEncoder()
    for c in df.select_dtypes(include='object').columns:
        df[c] = le.fit_transform(df[c])
    X = df[['Preco', 'Quantidade']]
    y = df['Demanda']
    return train_test_split(X, y, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = load_data()

# Inicializa sessão
if 'historico_dt' not in st.session_state: st.session_state.historico_dt = []
if 'historico_svm' not in st.session_state: st.session_state.historico_svm = []
if 'melhor' not in st.session_state: st.session_state.melhor = {'modelo': None, 'acuracia': 0}
if 'modelo_dt' not in st.session_state: st.session_state.modelo_dt = None
if 'modelo_svm' not in st.session_state: st.session_state.modelo_svm = None

def treinar_dt():
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    st.session_state.modelo_dt = model
    st.session_state.historico_dt.append(acc)
    if acc > st.session_state.melhor['acuracia']:
        st.session_state.melhor = {'modelo': 'Árvore de Decisão', 'acuracia': acc}
    st.success("Classificador Treinado com Sucesso")

def treinar_svm():
    pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='linear'))])
    pipeline.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    st.session_state.modelo_svm = pipeline
    st.session_state.historico_svm.append(acc)
    if acc > st.session_state.melhor['acuracia']:
        st.session_state.melhor = {'modelo': 'SVM', 'acuracia': acc}
    st.success("Classificador Treinado com Sucesso")

def exibir_historico(lista):
    if lista:
        for i, acc in enumerate(reversed(lista), 1):
            st.write(f"Teste {len(lista)-i+1}: **{acc*100:.2f}%**")
    else:
        st.info("Nenhum histórico registrado.")

if menu == "Árvore de Decisão":
    st.header("🌳 Árvore de Decisão - Menu")
    if st.button("Treinar Modelo Árvore de Decisão"): treinar_dt()
    st.subheader("Histórico de Acurácia")
    exibir_historico(st.session_state.historico_dt)
    if st.button("Mostrar Árvore de Decisão"):
        if st.session_state.modelo_dt:
            fig, ax = plt.subplots(figsize=(12,6))
            plot_tree(st.session_state.modelo_dt, feature_names=['Preco', 'Quantidade'], class_names=True, filled=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Treine o modelo antes de visualizar a árvore.")

elif menu == "SVM":
    st.header("🔎 SVM - Menu")
    if st.button("Treinar Modelo SVM"): treinar_svm()
    st.subheader("Histórico de Acurácia")
    exibir_historico(st.session_state.historico_svm)

elif menu == "Comparativo":
    st.header("📊 Comparativo de Desempenho")
    acc_dt = max(st.session_state.historico_dt) if st.session_state.historico_dt else 0
    acc_svm = max(st.session_state.historico_svm) if st.session_state.historico_svm else 0
    st.markdown(f"**Árvore de Decisão:** {acc_dt*100:.2f}%")
    st.markdown(f"**SVM:** {acc_svm*100:.2f}%")
    if acc_dt > acc_svm: st.success("🔍 Melhor desempenho: Árvore de Decisão")
    elif acc_svm > acc_dt: st.success("🔍 Melhor desempenho: SVM")
    else: st.info("🔍 Desempenho igual ou modelos não treinados.")

elif menu == "Limpar Histórico":
    st.session_state.historico_dt.clear()
    st.session_state.historico_svm.clear()
    st.session_state.melhor = {'modelo': None, 'acuracia': 0}
    st.success("Histórico geral limpo com sucesso!")
