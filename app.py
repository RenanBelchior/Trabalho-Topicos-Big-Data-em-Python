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

# Upload do arquivo CSV
arquivo = st.file_uploader("Faça upload do arquivo CSV com colunas: Preco, Quantidade, Demanda", type=["csv"])

# Inicialização do histórico
if 'historico' not in st.session_state:
    st.session_state.historico = []

if 'melhor_teste' not in st.session_state:
    st.session_state.melhor_teste = {'modelo': None, 'acuracia': 0.0}

# Se um arquivo for enviado
if arquivo is not None:
    df = pd.read_csv(arquivo, encoding='utf-8-sig')

    # Verificação das colunas obrigatórias
    col_entradas = ['Preco', 'Quantidade']
    col_saida = 'Demanda'

    if not all(col in df.columns for col in col_entradas + [col_saida]):
        st.error("O arquivo precisa conter as colunas: Preco, Quantidade e Demanda")
    else:
        # Codificação se necessário
        le = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            df[col] = le.fit_transform(df[col])

        # Separação das variáveis
        X = df[col_entradas]
        y = df[col_saida]

        # Treino/teste
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

                # Atualiza melhor desempenho
                if acc_dt > st.session_state.melhor_teste['acuracia']:
                    st.session_state.melhor_teste = {
                        'modelo': 'Árvore de Decisão',
                        'acuracia': acc_dt
                    }

                st.session_state.historico.append({
                    'modelo': 'Árvore de Decisão',
                    'acuracia': acc_dt
                })

        elif aba == "SVM":
            tipo_svm = st.radio("Tipo de SVM", ["SVM Básico", "SVM com Pipeline"])

            if tipo_svm == "SVM Básico":
                if st.button("Testar SVM Básico"):
                    modelo_svm = SVC(kernel='linear')
                    modelo_svm.fit(X_train, y_train)
                    y_pred_svm = modelo_svm.predict(X_test)
                    acc_svm = accuracy_score(y_test, y_pred_svm)

                    st.success(f"Acurácia do SVM Básico: {acc_svm * 100:.2f}%")

                    if acc_svm > st.session_state.melhor_teste['acuracia']:
                        st.session_state.melhor_teste = {
                            'modelo': 'SVM Básico',
                            'acuracia': acc_svm
                        }

                    st.session_state.historico.append({
                        'modelo': 'SVM Básico',
                        'acuracia': acc_svm
                    })

            else:
                if st.button("Testar SVM com Pipeline"):
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('svc', SVC(kernel='linear'))
                    ])
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    acc_pipeline = accuracy_score(y_test, y_pred)

                    st.success(f"Acurácia do SVM com Pipeline: {acc_pipeline * 100:.2f}%")

                    if acc_pipeline > st.session_state.melhor_teste['acuracia']:
                        st.session_state.melhor_teste = {
                            'modelo': 'SVM com Pipeline',
                            'acuracia': acc_pipeline
                        }

                    st.session_state.historico.append({
                        'modelo': 'SVM com Pipeline',
                        'acuracia': acc_pipeline
                    })

        # Exibição do melhor teste
        if st.session_state.melhor_teste['modelo']:
            st.subheader("🏆 Melhor Resultado Até Agora")
            st.markdown(f"**Modelo:** {st.session_state.melhor_teste['modelo']}")
            st.markdown(f"**Acurácia:** {st.session_state.melhor_teste['acuracia'] * 100:.2f}%")

        # Exibição do histórico
        if st.session_state.historico:
            st.subheader("📊 Histórico de Testes")
            for i, item in enumerate(st.session_state.historico[::-1]):
                st.markdown(f"**Teste {len(st.session_state.historico) - i}:** Modelo: `{item['modelo']}` | Acurácia: `{item['acuracia'] * 100:.2f}%`")
else:
    st.info("👈 Faça upload do arquivo CSV para começar.")
