import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Configuração inicial da aplicação Streamlit
st.set_page_config(page_title="Classificador Inteligente Autopeças", layout="wide")
st.title("Classificador Inteligente Autopeças")

# Menu lateral para navegação entre os classificadores
menu = st.sidebar.radio("Menu Principal", ["Árvore de Decisão", "SVM", "Comparativo", "Limpar Histórico"])

# Função para carregar e pré-processar os dados (utiliza cache para eficiência)
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/RenanBelchior/Trabalho-Topicos-Big-Data-em-Python/main/historico_vendas.csv", encoding='utf-8-sig')
    # Transformação de variáveis categóricas em numéricas
    le = LabelEncoder()
    for c in df.select_dtypes(include='object').columns:
        df[c] = le.fit_transform(df[c])
    # Seleção de features e target
    X = df[['Preco', 'Quantidade']]
    y = df['Demanda']
    # Separação em treino e teste
    return train_test_split(X, y, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = load_data()

# Inicializa estados da sessão para guardar modelos e métricas
if 'historico_dt' not in st.session_state: st.session_state.historico_dt = []
if 'historico_svm' not in st.session_state: st.session_state.historico_svm = []
if 'melhor' not in st.session_state: st.session_state.melhor = {'modelo': None, 'acuracia': 0}
if 'modelo_dt' not in st.session_state: st.session_state.modelo_dt = None
if 'modelo_svm' not in st.session_state: st.session_state.modelo_svm = None

# Treinamento do classificador Árvore de Decisão
def treinar_dt():
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    st.session_state.modelo_dt = model
    st.session_state.historico_dt.append(acc)
    # Verifica se esse modelo é o melhor até agora
    if acc > st.session_state.melhor['acuracia']:
        st.session_state.melhor = {'modelo': 'Árvore de Decisão', 'acuracia': acc}
    st.success("Classificador Treinado com Sucesso")

# Treinamento do classificador SVM com pipeline de normalização
def treinar_svm():
    pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='linear'))])
    pipeline.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    st.session_state.modelo_svm = pipeline
    st.session_state.historico_svm.append(acc)
    if acc > st.session_state.melhor['acuracia']:
        st.session_state.melhor = {'modelo': 'SVM', 'acuracia': acc}
    st.success("Classificador Treinado com Sucesso")

# Exibição do histórico de acurácias registradas
def exibir_historico(lista):
    if lista:
        for i, acc in enumerate(reversed(lista), 1):
            st.write(f"Teste {len(lista)-i+1}: **{acc*100:.2f}%**")
    else:
        st.info("Nenhum histórico registrado.")

# Seção da Árvore de Decisão
if menu == "Árvore de Decisão":
    st.header("🌳 Árvore de Decisão - Menu")
    if st.button("Nova Classificação"): treinar_dt()
    if st.button("Mostrar Desempenho"):
        exibir_historico(st.session_state.historico_dt)
    if st.button("Mostrar Árvore de Decisão"):
        if st.session_state.modelo_dt:
            fig, ax = plt.subplots(figsize=(12,6))
            plot_tree(st.session_state.modelo_dt, feature_names=['Preco', 'Quantidade'], class_names=True, filled=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Treine o modelo antes de visualizar a árvore.")

# Seção do classificador SVM
elif menu == "SVM":
    st.header("🔎 SVM - Menu")
    if st.button("Nova Classificação"): treinar_svm()
    if st.button("Mostrar Desempenho"):
        exibir_historico(st.session_state.historico_svm)

# Comparativo de desempenho entre os modelos treinados
elif menu == "Comparativo":
    st.header("📊 Comparativo de Desempenho")
    acc_dt = max(st.session_state.historico_dt) if st.session_state.historico_dt else 0
    acc_svm = max(st.session_state.historico_svm) if st.session_state.historico_svm else 0
    st.markdown(f"**Árvore de Decisão:** {acc_dt*100:.2f}%")
    st.markdown(f"**SVM:** {acc_svm*100:.2f}%")
    if acc_dt > acc_svm: st.success("🔍 Melhor desempenho: Árvore de Decisão")
    elif acc_svm > acc_dt: st.success("🔍 Melhor desempenho: SVM")
    else: st.info("🔍 Desempenho igual ou modelos não treinados.")

# Limpa todo o histórico registrado
elif menu == "Limpar Histórico":
    st.session_state.historico_dt.clear()
    st.session_state.historico_svm.clear()
    st.session_state.melhor = {'modelo': None, 'acuracia': 0}
    st.success("Histórico geral limpo com sucesso!")
