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

@st.cache_data
def carregar_dados():
    df = pd.read_csv("https://raw.githubusercontent.com/RenanBelchior/Trabalho-Topicos-Big-Data-em-Python/main/historico_vendas.csv", encoding='utf-8-sig')
    for col in df.select_dtypes('object'): df[col] = LabelEncoder().fit_transform(df[col])
    return df

df = carregar_dados()
colunas = [c for c in df.columns if c != 'Demanda']

# Inicializa sessão
for chave in ['historico_dt', 'historico_svm']: st.session_state.setdefault(chave, [])
st.session_state.setdefault('melhor', {'modelo': None, 'acuracia': 0})
st.session_state.setdefault('modelos', {'dt': None, 'svm': None})
st.session_state.setdefault('testes_finais', {'dt': 0, 'svm': 0})

# Função genérica de treino
def treinar_modelo(X, y, modelo):
    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)
    modelo.fit(X_train, y_train)
    acc_train = accuracy_score(y_val, modelo.predict(X_val))
    acc_final = accuracy_score(y_test_final, modelo.predict(X_test_final))
    return modelo, acc_train, acc_final

# Função para mostrar histórico
def exibir_historico(hist):
    if hist:
        for i, acc in enumerate(reversed(hist), 1):
            st.write(f"Treino {len(hist)-i+1}: **{acc*100:.2f}%**")
    else:
        st.info("Nenhum histórico registrado.")

# Página de modelos
def pagina_modelo(nome, chave, modelo_base, escalonar=False, mostrar_arvore=False):
    st.header(f"{'🌳' if nome == 'Árvore de Decisão' else '🔎'} {nome} - Menu")
    sel_cols = st.multiselect("Selecione as colunas de entrada:", colunas, default=['Preco', 'Quantidade'], key=chave)

    if st.button("Testar Nova Classificação", key=chave+'_treinar'):
        if sel_cols:
            X, y = df[sel_cols], df['Demanda']
            modelo = Pipeline([('scaler', StandardScaler()), ('model', modelo_base)]) if escalonar else modelo_base
            modelo, acc_train, acc_final = treinar_modelo(X, y, modelo)

            st.session_state['modelos'][chave] = modelo
            st.session_state[f'historico_{chave}'].append(acc_train)
            st.session_state['testes_finais'][chave] = acc_final

            if acc_train > st.session_state['melhor']['acuracia']:
                st.session_state['melhor'] = {'modelo': nome, 'acuracia': acc_train}

            st.success(f"Teste final: {acc_final*100:.2f}%")
        else:
            st.warning("Selecione ao menos uma coluna.")

    if st.button("Mostrar Desempenho", key=chave+'_desempenho'):
        st.subheader("Histórico de desempenho (Treino)")
        exibir_historico(st.session_state[f'historico_{chave}'])

    if mostrar_arvore and st.button("Mostrar Árvore de Decisão"):
        modelo = st.session_state['modelos'][chave]
        if modelo:
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_tree(modelo, feature_names=sel_cols, filled=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Treine o modelo primeiro.")

# Página Comparativo
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

# Página Limpar
def pagina_limpar():
    for chave in ['historico_dt', 'historico_svm']:
        st.session_state[chave].clear()
    st.session_state['melhor'] = {'modelo': None, 'acuracia': 0}
    st.session_state['testes_finais'] = {'dt': 0, 'svm': 0}
    st.success("Histórico limpo com sucesso.")

# Navegação
pagina = st.sidebar.radio("Menu", ["Árvore de Decisão", "SVM", "Comparativo", "Limpar Histórico"])
if pagina == "Árvore de Decisão":
    pagina_modelo("Árvore de Decisão", "dt", DecisionTreeClassifier(random_state=42), mostrar_arvore=True)
elif pagina == "SVM":
    pagina_modelo("SVM", "svm", SVC(kernel='linear'), escalonar=True)
elif pagina == "Comparativo":
    pagina_comparativo()
elif pagina == "Limpar Histórico":
    pagina_limpar()
