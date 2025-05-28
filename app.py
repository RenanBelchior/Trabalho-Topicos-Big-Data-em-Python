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

# Colunas de entrada e saída
col_auxiliares = ['Preco', 'Quantidade']
col_saida = 'Demanda'
st.info(f"**Colunas de entrada:** {col_auxiliares} | **Coluna de saída:** {col_saida}")

# Inicializa históricos e variáveis de controle
if 'historico' not in st.session_state:
    st.session_state.historico = []

if 'melhor' not in st.session_state:
    st.session_state.melhor = {'modelo': None, 'acuracia': 0}

if 'menu' not in st.session_state:
    st.session_state.menu = 'principal'

if 'sub_menu_dt' not in st.session_state:
    st.session_state.sub_menu_dt = None

if 'sub_menu_svm' not in st.session_state:
    st.session_state.sub_menu_svm = None

# Codificação de variáveis categóricas (se houver)
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Separação em X e y
X = df[col_auxiliares]
y = df[col_saida]

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Funções auxiliares
def testar_arvore_decisao():
    modelo_dt = DecisionTreeClassifier(random_state=42)
    modelo_dt.fit(X_train, y_train)
    y_pred = modelo_dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.session_state.historico.append({'modelo': 'Árvore de Decisão', 'acuracia': acc})
    if acc > st.session_state.melhor['acuracia']:
        st.session_state.melhor = {'modelo': 'Árvore de Decisão', 'acuracia': acc}
    return modelo_dt, acc


def testar_svm_basico():
    modelo_svm = SVC(kernel='linear')
    modelo_svm.fit(X_train, y_train)
    y_pred = modelo_svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.session_state.historico.append({'modelo': 'SVM Básico', 'acuracia': acc})
    if acc > st.session_state.melhor['acuracia']:
        st.session_state.melhor = {'modelo': 'SVM Básico', 'acuracia': acc}
    return modelo_svm, acc


def testar_svm_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='linear'))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.session_state.historico.append({'modelo': 'SVM com Pipeline', 'acuracia': acc})
    if acc > st.session_state.melhor['acuracia']:
        st.session_state.melhor = {'modelo': 'SVM com Pipeline', 'acuracia': acc}
    return pipeline, acc


# --- Menu Principal ---
if st.session_state.menu == 'principal':
    st.subheader("MENU PRINCIPAL")
    opcao = st.radio("Escolha uma opção:", [
        "Árvore de Decisão",
        "SVM",
        "Exibir Desempenhos dos Classificadores"
    ])

    if opcao == "Árvore de Decisão":
        st.session_state.menu = 'arvore_decisao'
        st.session_state.sub_menu_dt = None
    elif opcao == "SVM":
        st.session_state.menu = 'svm'
        st.session_state.sub_menu_svm = None
    elif opcao == "Exibir Desempenhos dos Classificadores":
        st.session_state.menu = 'desempenhos'


# --- Submenu Árvore de Decisão ---
elif st.session_state.menu == 'arvore_decisao':
    st.subheader("Árvore de Decisão - Menu")

    if st.session_state.sub_menu_dt is None:
        escolha = st.radio("Escolha uma ação:", [
            "Mostrar Desempenho",
            "Mostrar Árvore",
            "Fazer Nova Classificação",
            "Retornar ao Menu Principal"
        ])

        if escolha == "Mostrar Desempenho":
            st.session_state.sub_menu_dt = 'mostrar_desempenho'
        elif escolha == "Mostrar Árvore":
            st.session_state.sub_menu_dt = 'mostrar_arvore'
        elif escolha == "Fazer Nova Classificação":
            modelo_dt, acc = testar_arvore_decisao()
            st.success(f"Acurácia da Árvore de Decisão: {acc * 100:.2f}%")
        elif escolha == "Retornar ao Menu Principal":
            st.session_state.menu = 'principal'

    elif st.session_state.sub_menu_dt == 'mostrar_desempenho':
        # Mostra os resultados históricos da Árvore de Decisão
        st.write("### Histórico de testes da Árvore de Decisão:")
        historico_dt = [h for h in st.session_state.historico if h['modelo'] == 'Árvore de Decisão']
        if historico_dt:
            for i, item in enumerate(historico_dt[::-1], 1):
                st.write(f"Teste {i}: Acurácia = {item['acuracia']*100:.2f}%")
        else:
            st.info("Nenhum teste registrado para Árvore de Decisão.")
        if st.button("Voltar"):
            st.session_state.sub_menu_dt = None

    elif st.session_state.sub_menu_dt == 'mostrar_arvore':
        # Treina modelo para mostrar árvore
        modelo_dt = DecisionTreeClassifier(random_state=42)
        modelo_dt.fit(X_train, y_train)

        st.write("### Árvore de Decisão:")
        fig, ax = plt.subplots(figsize=(15, 10))
        plot_tree(modelo_dt, filled=True, feature_names=col_auxiliares, class_names=[str(c) for c in sorted(y.unique())], ax=ax)
        st.pyplot(fig)
        if st.button("Voltar"):
            st.session_state.sub_menu_dt = None


# --- Submenu SVM ---
elif st.session_state.menu == 'svm':
    st.subheader("SVM - Menu")

    if st.session_state.sub_menu_svm is None:
        escolha = st.radio("Escolha uma ação:", [
            "Mostrar Desempenho",
            "Fazer Nova Classificação",
            "Retornar ao Menu Principal"
        ])

        if escolha == "Mostrar Desempenho":
            st.session_state.sub_menu_svm = 'mostrar_desempenho'
        elif escolha == "Fazer Nova Classificação":
            tipo_svm = st.radio("Tipo de SVM", ["SVM Básico", "SVM com Pipeline"])
            if tipo_svm == "SVM Básico":
                if st.button("Testar SVM Básico"):
                    modelo_svm, acc = testar_svm_basico()
                    st.success(f"Acurácia do SVM Básico: {acc * 100:.2f}%")
            else:
                if st.button("Testar SVM com Pipeline"):
                    modelo_pipeline, acc = testar_svm_pipeline()
                    st.success(f"Acurácia do SVM com Pipeline: {acc * 100:.2f}%")
        elif escolha == "Retornar ao Menu Principal":
            st.session_state.menu = 'principal'

    elif st.session_state.sub_menu_svm == 'mostrar_desempenho':
        st.write("### Histórico de testes do SVM:")
        historico_svm = [h for h in st.session_state.historico if 'SVM' in h['modelo']]
        if historico_svm:
            for i, item in enumerate(historico_svm[::-1], 1):
                st.write(f"Teste {i}: Modelo = {item['modelo']} | Acurácia = {item['acuracia']*100:.2f}%")
        else:
            st.info("Nenhum teste registrado para SVM.")
        if st.button("Voltar"):
            st.session_state.sub_menu_svm = None


# --- Exibir Desempenhos Gerais ---
elif st.session_state.menu == 'desempenhos':
    st.subheader("📊 Histórico de Testes de Todos os Classificadores")
    if st.session_state.historico:
        for i, item in enumerate(st.session_state.historico[::-1], 1):
            st.write(f"Teste {i}: Modelo: {item['modelo']} | Acurácia: {item['acuracia'] * 100:.2f}%")
    else:
        st.info("Nenhum teste realizado ainda.")
    if st.session_state.melhor['modelo']:
        st.markdown(f"### ⭐ Melhor Desempenho Atual")
        st.markdown(f"**Modelo:** `{st.session_state.melhor['modelo']}` | **Acurácia:** `{st.session_state.melhor['acuracia'] * 100:.2f}%`")
    if st.button("Voltar ao Menu Principal"):
        st.session_state.menu = 'principal'
