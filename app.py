import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Previsão de Demanda - Autopeças", layout="wide")
st.title("📦 Sistema de Previsão de Demanda de Autopeças")

# Armazenar resultados para exibir depois
if 'acc_dt' not in st.session_state:
    st.session_state.acc_dt = None
if 'acc_svm' not in st.session_state:
    st.session_state.acc_svm = None

# Upload do arquivo CSV
arquivo = st.file_uploader("Faça upload do arquivo CSV", type=["csv"])

if arquivo is not None:
    df = pd.read_csv(arquivo, encoding='utf-8-sig')
    st.subheader("Pré-visualização dos Dados")
    st.dataframe(df.head())

    colunas_disponiveis = df.columns.tolist()

    col_auxiliares = st.multiselect("Escolha as colunas auxiliares (entradas):", colunas_disponiveis, default=colunas_disponiveis[:2])
    col_saida = st.selectbox("Escolha a coluna de saída (target):", colunas_disponiveis, index=len(colunas_disponiveis) - 1)

    if col_auxiliares and col_saida:
        le = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            df[col] = le.fit_transform(df[col])

        X = df[col_auxiliares]
        y = df[col_saida]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        st.subheader("Classificadores")

        aba = st.radio("Escolha um classificador:", ["Árvore de Decisão", "SVM"])

        if aba == "Árvore de Decisão":
            if st.button("Testar Classificador Árvore de Decisão"):
                modelo_dt = DecisionTreeClassifier(random_state=42)
                modelo_dt.fit(X_train, y_train)
                y_pred_dt = modelo_dt.predict(X_test)
                acc_dt = accuracy_score(y_test, y_pred_dt)
                st.session_state.acc_dt = acc_dt
                st.success(f"Acurácia da Árvore de Decisão: {acc_dt * 100:.2f}%")

        elif aba == "SVM":
            op_svm = st.radio("Tipo de SVM", ["SVM Básico", "SVM com Pipeline"])

            if st.button("Testar Classificador SVM"):
                if op_svm == "SVM Básico":
                    modelo_svm = SVC(kernel='linear')
                    modelo_svm.fit(X_train, y_train)
                    y_pred_svm = modelo_svm.predict(X_test)
                else:
                    pipeline_svm = Pipeline([
                        ('scaler', StandardScaler()),
                        ('svc', SVC(kernel='linear'))
                    ])
                    pipeline_svm.fit(X_train, y_train)
                    y_pred_svm = pipeline_svm.predict(X_test)

                acc_svm = accuracy_score(y_test, y_pred_svm)
                st.session_state.acc_svm = acc_svm
                st.success(f"Acurácia do SVM ({op_svm}): {acc_svm * 100:.2f}%")

        # Exibição apenas das acurácias se ambos foram testados
        if st.session_state.acc_dt is not None or st.session_state.acc_svm is not None:
            st.subheader("📊 Comparativo de Desempenho dos Classificadores")
            if st.session_state.acc_dt is not None:
                st.write(f"🌳 Árvore de Decisão: **{st.session_state.acc_dt * 100:.2f}%**")
            if st.session_state.acc_svm is not None:
                st.write(f"🧠 SVM: **{st.session_state.acc_svm * 100:.2f}%**")

else:
    st.info("👈 Faça upload do arquivo CSV para começar.")
