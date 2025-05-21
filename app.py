import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Previsão de Demanda - Autopeças", layout="wide")

st.title("📦 Sistema de Previsão de Demanda de Autopeças")

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
        # LabelEncoder para colunas categóricas
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
                st.success(f"Acurácia da Árvore de Decisão: {acc_dt * 100:.2f}%")
                st.text("Relatório:")
                st.text(classification_report(y_test, y_pred_dt))

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
                st.success(f"Acurácia do SVM ({op_svm}): {acc_svm * 100:.2f}%")
                st.text("Relatório:")
                st.text(classification_report(y_test, y_pred_svm))
else:
    st.info("👈 Faça upload do arquivo CSV para começar.")
