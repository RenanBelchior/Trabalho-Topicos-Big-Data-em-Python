# === CONFIGURAÇÕES E IMPORTAÇÕES ===
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config("Classificador MasterPeças", layout="wide")
st.title("🔧 Classificador Inteligente Master Peças")

# === CARREGAMENTO E PRÉ-PROCESSAMENTO DE DADOS ===
@st.cache_data
def carregar_dados():
    df = pd.read_csv("https://raw.githubusercontent.com/RenanBelchior/Trabalho-Topicos-Big-Data-em-Python/main/historico_vendas.csv", encoding='utf-8-sig')
    for col in df.select_dtypes('object'):
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

df = carregar_dados()
colunas = [c for c in df.columns if c != 'Demanda']

# === INICIALIZAÇÃO DE ESTADO DE SESSÃO ===
for chave in ['historico_dt', 'historico_svm']:
    st.session_state.setdefault(chave, [])
st.session_state.setdefault('melhor', {'modelo': None, 'acuracia': 0})
st.session_state.setdefault('modelos', {'dt': None, 'svm': None})
st.session_state.setdefault('testes_finais', {'dt': 0, 'svm': 0})

# === FUNÇÕES AUXILIARES GERAIS ===
def treinar_modelo(X, y, modelo):
    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)
    modelo.fit(X_train, y_train)
    acc_train = accuracy_score(y_val, modelo.predict(X_val))
    acc_final = accuracy_score(y_test_final, modelo.predict(X_test_final))
    return modelo, acc_train, acc_final

def exibir_historico(hist):
    if hist:
        for i, acc in enumerate(reversed(hist), 1):
            st.write(f"Treino {len(hist)-i+1}: **{acc*100:.2f}%**")
    else:
        st.info("Nenhum histórico registrado.")

# === ÁRVORE DE DECISÃO ===
def pagina_arvore_decisao():
    st.header("🌳 Árvore de Decisão - Menu")
    sel_cols = st.multiselect("Selecione as colunas de entrada:", colunas, default=['Preco', 'Quantidade'], key='dt')

    if st.button("Testar Nova Classificação", key='dt_treinar'):
        if sel_cols:
            X, y = df[sel_cols], df['Demanda']
            modelo = DecisionTreeClassifier(random_state=42)
            modelo, acc_train, acc_final = treinar_modelo(X, y, modelo)

            st.session_state['modelos']['dt'] = modelo
            st.session_state['historico_dt'].append(acc_train)
            st.session_state['testes_finais']['dt'] = acc_final

            if acc_train > st.session_state['melhor']['acuracia']:
                st.session_state['melhor'] = {'modelo': "Árvore de Decisão", 'acuracia': acc_train}

            st.success(f"Teste final: {acc_final*100:.2f}%")
        else:
            st.warning("Selecione ao menos uma coluna.")

    if st.button("Mostrar Desempenho", key='dt_desempenho'):
        st.subheader("Histórico de desempenho (Treino)")
        exibir_historico(st.session_state['historico_dt'])

    if st.button("Mostrar Árvore de Decisão"):
        modelo = st.session_state['modelos']['dt']
        if modelo:
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_tree(modelo, feature_names=sel_cols, filled=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Treine o modelo primeiro.")

# === SVM ===
def pagina_svm():
    st.header("🔎 SVM - Menu")
    sel_cols = st.multiselect("Selecione as colunas de entrada:", colunas, default=['Preco', 'Quantidade'], key='svm')

    if st.button("Testar Nova Classificação", key='svm_treinar'):
        if sel_cols:
            X, y = df[sel_cols], df['Demanda']
            modelo = Pipeline([('scaler', StandardScaler()), ('model', SVC(kernel='linear'))])
            modelo, acc_train, acc_final = treinar_modelo(X, y, modelo)

            st.session_state['modelos']['svm'] = modelo
            st.session_state['historico_svm'].append(acc_train)
            st.session_state['testes_finais']['svm'] = acc_final

            if acc_train > st.session_state['melhor']['acuracia']:
                st.session_state['melhor'] = {'modelo': "SVM", 'acuracia': acc_train}

            st.success(f"Teste final: {acc_final*100:.2f}%")
        else:
            st.warning("Selecione ao menos uma coluna.")

    if st.button("Mostrar Desempenho", key='svm_desempenho'):
        st.subheader("Histórico de desempenho (Treino)")
        exibir_historico(st.session_state['historico_svm'])

# === COMPARATIVO ===
def pagina_comparativo():
    st.header("📊 Comparativo de Desempenho - Teste Final")
    dt, svm = st.session_state['testes_finais'].values()
    st.markdown(f"**Árvore de Decisão:** {dt*100:.2f}%")
    st.markdown(f"**SVM:** {svm*100:.2f}%")
    if dt > svm:
        st.success("🔍 Melhor: Árvore de Decisão")
    elif svm > dt:
        st.success("🔍 Melhor: SVM")
    else:
        st.info("🔍 Desempenho igual ou não treinado.")

# === LIMPEZA ===
def pagina_limpar():
    for chave in ['historico_dt', 'historico_svm']:
        st.session_state[chave].clear()
    st.session_state['melhor'] = {'modelo': None, 'acuracia': 0}
    st.session_state['testes_finais'] = {'dt': 0, 'svm': 0}
    st.success("Histórico limpo com sucesso.")

# === NAVEGAÇÃO ===
pagina = st.sidebar.radio("Menu", ["Árvore de Decisão", "SVM", "Comparativo", "Limpar Histórico"])
if pagina == "Árvore de Decisão":
    pagina_arvore_decisao()
elif pagina == "SVM":
    pagina_svm()
elif pagina == "Comparativo":
    pagina_comparativo()
elif pagina == "Limpar Histórico":
    pagina_limpar()
