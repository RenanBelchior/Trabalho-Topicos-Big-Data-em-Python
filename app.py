import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import io

# Configuração da página
st.set_page_config(page_title="Previsão de Demanda - Autopeças", layout="wide")

# Título principal do aplicativo
st.title("📦 Sistema de Previsão de Demanda de Autopeças")

# Upload do arquivo CSV
arquivo = st.file_uploader("Faça upload do arquivo CSV", type=["csv"])

# Histórico de testes
if 'historico' not in st.session_state:
    st.session_state.historico = []

# Se um arquivo for enviado
if arquivo is not None:
    # Leitura do CSV
    df = pd.read_csv(arquivo, encoding='utf-8-sig')

    # Mostra os primeiros registros dos dados
    st.subheader("Pré-visualização dos Dados")
    st.dataframe(df.head())

    # Lista de colunas disponíveis
    colunas_disponiveis = df.columns.tolist()

    # Seleção de colunas auxiliares (entradas) e coluna de saída (target)
    col_auxiliares = st.multiselect("Escolha as colunas auxiliares (entradas):", colunas_disponiveis, default=colunas_disponiveis[:2])
    col_saida = st.selectbox("Escolha a coluna de saída (target):", colunas_disponiveis, index=len(colunas_disponiveis) - 1)

    # Se colunas foram selecionadas
    if col_auxiliares and col_saida:
        # Codificação de variáveis categóricas com LabelEncoder
        le = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            df[col] = le.fit_transform(df[col])

        # Separação em variáveis explicativas (X) e alvo (y)
        X = df[col_auxiliares]
        y = df[col_saida]

        # Divisão em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Título da seção de classificadores
        st.subheader("Classificadores")

        # Escolha do classificador
        aba = st.radio("Escolha um classificador:", ["Árvore de Decisão", "SVM"])

        # Classificador Árvore de Decisão
        if aba == "Árvore de Decisão":
            if st.button("Testar Classificador Árvore de Decisão"):
                modelo_dt = DecisionTreeClassifier(random_state=42)
                modelo_dt.fit(X_train, y_train)
                y_pred_dt = modelo_dt.predict(X_test)
                acc_dt = accuracy_score(y_test, y_pred_dt)
                st.success(f"Acurácia da Árvore de Decisão: {acc_dt * 100:.2f}%")

                # Salva o teste no histórico
                st.session_state.historico.append({
                    'modelo': 'Árvore de Decisão',
                    'acuracia': acc_dt,
                    'colunas': col_auxiliares,
                    'target': col_saida,
                    'modelo_obj': modelo_dt,
                    'tipo': 'arvore'
                })

        # Classificador SVM
        elif aba == "SVM":
            op_svm = st.radio("Tipo de SVM", ["SVM Básico", "SVM com Pipeline"])

            if op_svm == "SVM Básico":
                if st.button("Testar SVM Básico"):
                    modelo_svm = SVC(kernel='linear')
                    modelo_svm.fit(X_train, y_train)
                    y_pred_svm = modelo_svm.predict(X_test)
                    acc_svm_basico = accuracy_score(y_test, y_pred_svm)
                    st.success(f"Acurácia do SVM Básico: {acc_svm_basico * 100:.2f}%")

                    st.session_state.historico.append({
                        'modelo': 'SVM Básico',
                        'acuracia': acc_svm_basico,
                        'colunas': col_auxiliares,
                        'target': col_saida,
                        'tipo': 'svm'
                    })

            else:
                if st.button("Testar SVM com Pipeline"):
                    pipeline_svm = Pipeline([
                        ('scaler', StandardScaler()),
                        ('svc', SVC(kernel='linear'))
                    ])
                    pipeline_svm.fit(X_train, y_train)
                    y_pred_svm_pipeline = pipeline_svm.predict(X_test)
                    acc_svm_pipeline = accuracy_score(y_test, y_pred_svm_pipeline)
                    st.success(f"Acurácia do SVM com Pipeline: {acc_svm_pipeline * 100:.2f}%")

                    st.session_state.historico.append({
                        'modelo': 'SVM com Pipeline',
                        'acuracia': acc_svm_pipeline,
                        'colunas': col_auxiliares,
                        'target': col_saida,
                        'tipo': 'svm'
                    })

        # Exibição do histórico
        if st.session_state.historico:
            st.subheader("📊 Histórico de Testes")
            for i, item in enumerate(st.session_state.historico[::-1]):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Teste {len(st.session_state.historico)-i}:** Modelo: `{item['modelo']}` | Acurácia: `{item['acuracia'] * 100:.2f}%` | Entradas: `{', '.join(item['colunas'])}` | Saída: `{item['target']}`")
                with col2:
                    if item['tipo'] == 'arvore':
                        botao_key = f"arvore_{len(st.session_state.historico)-i}"
                        if st.button(f"Visualizar Árvore {len(st.session_state.historico)-i}", key=botao_key):
                            fig, ax = plt.subplots(figsize=(10, 6))
                            plot_tree(item['modelo_obj'], feature_names=item['colunas'], class_names=[str(cls) for cls in set(y)], filled=True, ax=ax)
                            st.pyplot(fig)
else:
    # Mensagem caso nenhum arquivo tenha sido enviado
    st.info("👈 Faça upload do arquivo CSV para começar.")
