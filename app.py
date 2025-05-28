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
st.set_page_config(page_title="Previsão de Demanda - Autopeças", layout="wide")
st.title("📦 Sistema de Previsão de Demanda de Autopeças")

# Leitura direta do arquivo do GitHub
url_dados = "https://raw.githubusercontent.com/RenanBelchior/Trabalho-Topicos-Big-Data-em-Python/main/historico_vendas.csv"
df = pd.read_csv(url_dados, encoding='utf-8-sig')

# Exibição das colunas utilizadas
col_auxiliares = ['Preco', 'Quantidade']
col_saida = 'Demanda'
st.info(f"**Colunas de entrada:** {col_auxiliares} | **Coluna de saída:** {col_saida}")

# Codificação de variáveis categóricas
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Separação em X e y
X = df[col_auxiliares]
y = df[col_saida]

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicialização dos classificadores e acurácias
modelo_dt = None
modelo_svm = None
acc_dt = None
acc_svm = None

# Menu principal
menu = st.sidebar.selectbox("Menu Principal", [
    "Árvore de Decisão",
    "SVM",
    "Exibir Desempenho dos Classificadores",
    "Encerrar"
])

# Submenu Árvore de Decisão
if menu == "Árvore de Decisão":
    submenu_dt = st.sidebar.radio("Opções - Árvore de Decisão", [
        "Treinar o Classificador Árvore de Decisão",
        "Mostrar o Desempenho",
        "Mostrar Árvore",
        "Fazer nova Classificação",
        "Retornar ao Menu Principal"
    ])

    if submenu_dt == "Treinar o Classificador Árvore de Decisão":
        modelo_dt = DecisionTreeClassifier(random_state=42)
        modelo_dt.fit(X_train, y_train)
        y_pred_dt = modelo_dt.predict(X_test)
        acc_dt = accuracy_score(y_test, y_pred_dt)
        st.success(f"Classificador Árvore de Decisão treinado com acurácia: {acc_dt * 100:.2f}%")

    elif submenu_dt == "Mostrar o Desempenho":
        if acc_dt is not None:
            st.info(f"Acurácia da Árvore de Decisão: {acc_dt * 100:.2f}%")
        else:
            st.warning("O classificador ainda não foi treinado.")

    elif submenu_dt == "Mostrar Árvore":
        if modelo_dt is not None:
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_tree(modelo_dt, feature_names=col_auxiliares, class_names=True, filled=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("O classificador ainda não foi treinado.")

    elif submenu_dt == "Fazer nova Classificação":
        if modelo_dt is not None:
            preco = st.number_input("Informe o Preço", min_value=0.0)
            quantidade = st.number_input("Informe a Quantidade", min_value=0)
            if st.button("Classificar"):
                pred = modelo_dt.predict([[preco, quantidade]])
                st.success(f"Demanda Prevista: {pred[0]}")
        else:
            st.warning("O classificador ainda não foi treinado.")

# Submenu SVM
elif menu == "SVM":
    submenu_svm = st.sidebar.radio("Opções - SVM", [
        "Treinar Classificador SVM",
        "Mostrar o desempenho",
        "Fazer nova Classificação",
        "Retornar ao Menu Principal"
    ])

    if submenu_svm == "Treinar Classificador SVM":
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='linear'))
        ])
        pipeline.fit(X_train, y_train)
        modelo_svm = pipeline
        y_pred_svm = modelo_svm.predict(X_test)
        acc_svm = accuracy_score(y_test, y_pred_svm)
        st.success(f"Classificador SVM treinado com acurácia: {acc_svm * 100:.2f}%")

    elif submenu_svm == "Mostrar o desempenho":
        if acc_svm is not None:
            st.info(f"Acurácia do SVM: {acc_svm * 100:.2f}%")
        else:
            st.warning("O classificador ainda não foi treinado.")

    elif submenu_svm == "Fazer nova Classificação":
        if modelo_svm is not None:
            preco = st.number_input("Informe o Preço", min_value=0.0, key="svm_preco")
            quantidade = st.number_input("Informe a Quantidade", min_value=0, key="svm_quantidade")
            if st.button("Classificar", key="btn_svm"):
                pred = modelo_svm.predict([[preco, quantidade]])
                st.success(f"Demanda Prevista: {pred[0]}")
        else:
            st.warning("O classificador ainda não foi treinado.")

# Exibir melhor desempenho
elif menu == "Exibir Desempenho dos Classificadores":
    if acc_dt is not None or acc_svm is not None:
        melhor_modelo = ""
        melhor_acc = 0

        if acc_dt is not None and (acc_svm is None or acc_dt > acc_svm):
            melhor_modelo = "Árvore de Decisão"
            melhor_acc = acc_dt
        elif acc_svm is not None:
            melhor_modelo = "SVM"
            melhor_acc = acc_svm

        st.success(f"Melhor modelo: {melhor_modelo} com acurácia de {melhor_acc * 100:.2f}%")
    else:
        st.warning("Nenhum classificador foi treinado ainda.")

# Encerrar
elif menu == "Encerrar":
    st.warning("Encerrando aplicação...")
