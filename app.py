import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Configuração da página
st.set_page_config(page_title="Previsão de Demanda - Autopeças", layout="wide")
st.title("📦 Sistema de Previsão de Demanda de Autopeças")

# Leitura direta do arquivo do GitHub
url_dados = "https://raw.githubusercontent.com/RenanBelchior/Trabalho-Topicos-Big-Data-em-Python/main/historico_vendas.csv"
df = pd.read_csv(url_dados, encoding='utf-8-sig')

# Exibição das colunas utilizadas
col_auxiliares = [ Preco, Quantidade]
col_saida = 'Demanda'
st.info(f"**Colunas de entrada:** {col_auxiliares} | **Coluna de saída:** {col_saida}")

# Inicializa histórico e melhor resultado
if 'historico' not in st.session_state:
    st.session_state.historico = []
if 'melhor' not in st.session_state:
    st.session_state.melhor = {'modelo': None, 'acuracia': 0}

# Codificação de variáveis categóricas
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Separação em X e y
X = df[col_auxiliares]
y = df[col_saida]

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escolha do classificador
st.subheader("Classificadores")
aba = st.radio("Escolha um classificador:", ["Árvore de Decisão", "SVM"])

if aba == "Árvore de Decisão":
    if st.button("Testar Classificador Árvore de Decisão"):
        modelo_dt = DecisionTreeClassifier(random_state=42)
        modelo_dt.fit(X_train, y_train)
        y_pred_dt = modelo_dt.predict(X_test)
        acc_dt = accuracy_score(y_test, y_pred_dt)
        st.success(f"Acurácia da Árvore de Decisão: {acc_dt * 100:.2f}%")

        st.session_state.historico.append({
            'modelo': 'Árvore de Decisão',
            'acuracia': acc_dt
        })

        if acc_dt > st.session_state.melhor['acuracia']:
            st.session_state.melhor = {
                'modelo': 'Árvore de Decisão',
                'acuracia': acc_dt
            }

elif aba == "SVM":
    tipo_svm = st.radio("Tipo de SVM", ["SVM Básico", "SVM com Pipeline"])

    if tipo_svm == "SVM Básico":
        if st.button("Testar SVM Básico"):
            modelo_svm = SVC(kernel='linear')
            modelo_svm.fit(X_train, y_train)
            y_pred_svm = modelo_svm.predict(X_test)
            acc_svm = accuracy_score(y_test, y_pred_svm)
            st.success(f"Acurácia do SVM Básico: {acc_svm * 100:.2f}%")

            st.session_state.historico.append({
                'modelo': 'SVM Básico',
                'acuracia': acc_svm
            })

            if acc_svm > st.session_state.melhor['acuracia']:
                st.session_state.melhor = {
                    'modelo': 'SVM Básico',
                    'acuracia': acc_svm
                }

    elif tipo_svm == "SVM com Pipeline":
        if st.button("Testar SVM com Pipeline"):
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(kernel='linear'))
            ])
            pipeline.fit(X_train, y_train)
            y_pred_pipeline = pipeline.predict(X_test)
            acc_pipeline = accuracy_score(y_test, y_pred_pipeline)
            st.success(f"Acurácia do SVM com Pipeline: {acc_pipeline * 100:.2f}%")

            st.session_state.historico.append({
                'modelo': 'SVM com Pipeline',
                'acuracia': acc_pipeline
            })

            if acc_pipeline > st.session_state.melhor['acuracia']:
                st.session_state.melhor = {
                    'modelo': 'SVM com Pipeline',
                    'acuracia': acc_pipeline
                }

# Histórico e melhor desempenho
if st.session_state.historico:
    st.subheader("📊 Histórico de Testes")
    for i, item in enumerate(st.session_state.historico[::-1]):
        st.markdown(f"**Teste {len(st.session_state.historico)-i}:** Modelo: `{item['modelo']}` | Acurácia: `{item['acuracia'] * 100:.2f}%`")

if st.session_state.melhor['modelo']:
    st.subheader("⭐ Melhor Desempenho Atual")
    st.markdown(f"**Modelo:** `{st.session_state.melhor['modelo']}` | **Acurácia:** `{st.session_state.melhor['acuracia'] * 100:.2f}%`")
