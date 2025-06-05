import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Configura o layout da página do Streamlit
st.set_page_config("Classificador MasterPeças", layout="wide")
st.title("🔧 Classificador Inteligente Master Peças")

# Função para carregar e pré-processar os dados
@st.cache_data
def carregar_dados():
    df = pd.read_csv("https://raw.githubusercontent.com/RenanBelchior/Trabalho-Topicos-Big-Data-em-Python/main/historico_vendas.csv", encoding='utf-8-sig')
    # Codifica colunas categóricas com LabelEncoder
    for col in df.select_dtypes('object'):
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

# Carrega os dados
df = carregar_dados()
# Define as colunas disponíveis para seleção, exceto a coluna de saída
colunas = [c for c in df.columns if c != 'Demanda']

# Inicializa variáveis de sessão para guardar histórico e modelos treinados
for chave in ['historico_dt', 'historico_svm']:
    st.session_state.setdefault(chave, [])
st.session_state.setdefault('melhor', {'modelo': None, 'acuracia': 0})
st.session_state.setdefault('modelos', {'dt': None, 'svm': None})
st.session_state.setdefault('testes_finais', {'dt': 0, 'svm': 0})

# Função genérica para treinar um modelo (Árvore ou SVM)
def treinar_modelo(X, y, modelo):
    # Divide os dados em treino+validação e teste final
    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(X, y, test_size=0.3, random_state=42)
    # Divide o treino+validação em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)
    # Treina o modelo com os dados de treino
    modelo.fit(X_train, y_train)
    # Avalia a acurácia nos dados de validação
    acc_train = accuracy_score(y_val, modelo.predict(X_val))
    # Avalia a acurácia nos dados de teste final
    acc_final = accuracy_score(y_test_final, modelo.predict(X_test_final))
    return modelo, acc_train, acc_final

# Função para exibir o histórico de acurácias
def exibir_historico(hist):
    if hist:
        for i, acc in enumerate(reversed(hist), 1):
            st.write(f"Treino {len(hist)-i+1}: **{acc*100:.2f}%**")
    else:
        st.info("Nenhum histórico registrado.")

# Função principal para treinar/testar um modelo
# nome: Nome do modelo para exibição
# chave: chave única usada na sessão
# modelo_base: instância do modelo (DecisionTreeClassifier ou SVC)
# escalonar: define se deve usar padronização dos dados
# mostrar_arvore: habilita botão para mostrar visualização da árvore

def pagina_modelo(nome, chave, modelo_base, escalonar=False, mostrar_arvore=False):
    st.header(f"{'🌳' if nome == 'Árvore de Decisão' else '🔎'} {nome} - Menu")
    # Seleciona colunas de entrada
    sel_cols = st.multiselect("Selecione as colunas de entrada:", colunas, default=['Preco', 'Quantidade'], key=chave)

    # Botão para treinar o modelo e testar
    if st.button("Testar Nova Classificação", key=chave+'_treinar'):
        if sel_cols:
            # Define X como colunas de entrada e y como coluna alvo
            X, y = df[sel_cols], df['Demanda']
            # Usa pipeline com padronização se necessário (para SVM)
            modelo = Pipeline([('scaler', StandardScaler()), ('model', modelo_base)]) if escalonar else modelo_base
            modelo, acc_train, acc_final = treinar_modelo(X, y, modelo)

            # Armazena modelo e histórico na sessão
            st.session_state['modelos'][chave] = modelo
            st.session_state[f'historico_{chave}'].append(acc_train)
            st.session_state['testes_finais'][chave] = acc_final

            # Atualiza o melhor modelo se necessário
            if acc_train > st.session_state['melhor']['acuracia']:
                st.session_state['melhor'] = {'modelo': nome, 'acuracia': acc_train}

            # Exibe a acurácia final
            st.success(f"Teste final: {acc_final*100:.2f}%")
        else:
            st.warning("Selecione ao menos uma coluna.")

    # Botão para mostrar histórico de desempenho (validação)
    if st.button("Mostrar Desempenho", key=chave+'_desempenho'):
        st.subheader("Histórico de desempenho (Treino)")
        exibir_historico(st.session_state[f'historico_{chave}'])

    # Botão para mostrar visualização da árvore de decisão
    if mostrar_arvore and st.button("Mostrar Árvore de Decisão"):
        modelo = st.session_state['modelos'][chave]
        if modelo:
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_tree(modelo, feature_names=sel_cols, filled=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Treine o modelo primeiro.")

# Página de comparação entre os modelos

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

# Página para limpar os dados da sessão

def pagina_limpar():
    for chave in ['historico_dt', 'historico_svm']:
        st.session_state[chave].clear()
    st.session_state['melhor'] = {'modelo': None, 'acuracia': 0}
    st.session_state['testes_finais'] = {'dt': 0, 'svm': 0}
    st.success("Histórico limpo com sucesso.")

# Menu lateral de navegação entre páginas
pagina = st.sidebar.radio("Menu", ["Árvore de Decisão", "SVM", "Comparativo", "Limpar Histórico"])

# Chama a função correspondente à página selecionada
if pagina == "Árvore de Decisão":
    pagina_modelo("Árvore de Decisão", "dt", DecisionTreeClassifier(random_state=42), mostrar_arvore=True)
elif pagina == "SVM":
    pagina_modelo("SVM", "svm", SVC(kernel='linear'), escalonar=True)
elif pagina == "Comparativo":
    pagina_comparativo()
elif pagina == "Limpar Histórico":
    pagina_limpar()
