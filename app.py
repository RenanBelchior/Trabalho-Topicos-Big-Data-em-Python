import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import time

# Configuração da página
st.set_page_config(page_title="Previsão de Demanda - Autopeças", layout="wide")

# --- Funções auxiliares ---

def carregar_dados():
    url_dados = "https://raw.githubusercontent.com/RenanBelchior/Trabalho-Topicos-Big-Data-em-Python/main/historico_vendas.csv"
    df = pd.read_csv(url_dados, encoding='utf-8-sig')
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    return df

def split_data(df, col_entrada, col_saida):
    X = df[col_entrada]
    y = df[col_saida]
    return train_test_split(X, y, test_size=0.3, random_state=42)

def plotar_arvore(modelo, col_entrada):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,8))
    plot_tree(modelo, feature_names=col_entrada, filled=True, ax=ax, rounded=True)
    st.pyplot(fig)

# --- Dados e variáveis fixas ---

col_auxiliares = ['Preco', 'Quantidade']
col_saida = 'Demanda'
df = carregar_dados()

# Inicializa histórico e melhor resultado
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

# --- Divisão dos dados ---

X_train, X_test, y_train, y_test = split_data(df, col_auxiliares, col_saida)

# --- Interface ---

st.title("📦 Sistema de Previsão de Demanda de Autopeças")

st.info(f"**Colunas de entrada:** {col_auxiliares} | **Coluna de saída:** {col_saida}")

# Menu Principal
if st.session_state.menu == 'principal':
    st.subheader("MENU PRINCIPAL")
    opcao = st.radio("Escolha uma opção:", 
                     ["Árvore de Decisão", "SVM", "Exibir Desempenhos dos Classificadores", "Encerrar programa"])
    
    if opcao == "Árvore de Decisão":
        st.session_state.menu = 'arvore_decisao'
        st.session_state.sub_menu_dt = None
        st.experimental_rerun()

    elif opcao == "SVM":
        st.session_state.menu = 'svm'
        st.session_state.sub_menu_svm = None
        st.experimental_rerun()

    elif opcao == "Exibir Desempenhos dos Classificadores":
        st.subheader("📊 Histórico de Testes")
        if st.session_state.historico:
            for i, item in enumerate(st.session_state.historico[::-1]):
                st.markdown(f"**Teste {len(st.session_state.historico)-i}:** Modelo: `{item['modelo']}` | Acurácia: `{item['acuracia'] * 100:.2f}%`")
        else:
            st.info("Nenhum teste realizado ainda.")

        if st.session_state.melhor['modelo']:
            st.subheader("⭐ Melhor Desempenho Atual")
            st.markdown(f"**Modelo:** `{st.session_state.melhor['modelo']}` | **Acurácia:** `{st.session_state.melhor['acuracia'] * 100:.2f}%`")
        
        if st.button("Voltar ao Menu Principal"):
            st.experimental_rerun()

    elif opcao == "Encerrar programa":
        st.warning("O programa será encerrado em alguns segundos...")
        # Script para fechar aba automaticamente
        close_script = """
        <script>
        setTimeout(() => { window.close(); }, 3000);
        </script>
        """
        st.markdown(close_script, unsafe_allow_html=True)

# Submenu Árvore de Decisão
elif st.session_state.menu == 'arvore_decisao':
    st.subheader("Árvore de Decisão - MENU")
    opcao_dt = st.radio("Escolha uma opção:", ["Mostrar Desempenho", "Mostrar Árvore", "Fazer Nova Classificação", "Retornar ao Menu Principal"])

    if opcao_dt == "Mostrar Desempenho":
        # Mostrar histórico da Árvore de Decisão
        dt_historico = [h for h in st.session_state.historico if "Árvore de Decisão" in h['modelo']]
        if dt_historico:
            for i, item in enumerate(dt_historico[::-1]):
                st.markdown(f"**Teste {len(dt_historico)-i}:** Modelo: `{item['modelo']}` | Acurácia: `{item['acuracia'] * 100:.2f}%`")
        else:
            st.info("Nenhum teste de Árvore de Decisão realizado ainda.")

    elif opcao_dt == "Mostrar Árvore":
        if 'modelo_dt' in st.session_state:
            plotar_arvore(st.session_state.modelo_dt, col_auxiliares)
        else:
            st.warning("Você precisa fazer uma classificação antes de mostrar a árvore.")

    elif opcao_dt == "Fazer Nova Classificação":
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
        st.session_state.modelo_dt = modelo_dt

    elif opcao_dt == "Retornar ao Menu Principal":
        st.session_state.menu = 'principal'
        st.experimental_rerun()

# Submenu SVM
elif st.session_state.menu == 'svm':
    st.subheader("SVM - MENU")
    opcao_svm = st.radio("Escolha uma opção:", ["Mostrar Desempenho", "Fazer Nova Classificação", "Retornar ao Menu Principal"])

    if opcao_svm == "Mostrar Desempenho":
        svm_historico = [h for h in st.session_state.historico if "SVM" in h['modelo']]
        if svm_historico:
            for i, item in enumerate(svm_historico[::-1]):
                st.markdown(f"**Teste {len(svm_historico)-i}:** Modelo: `{item['modelo']}` | Acurácia: `{item['acuracia'] * 100:.2f}%`")
        else:
            st.info("Nenhum teste de SVM realizado ainda.")

    elif opcao_svm == "Fazer Nova Classificação":
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
                st.session_state.modelo_svm = modelo_svm

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
                st.session_state.modelo_svm = pipeline

    elif opcao_svm == "Retornar ao Menu Principal":
        st.session_state.menu = 'principal'
        st.experimental_rerun()
