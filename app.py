import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ----- VARIÁVEIS GLOBAIS -----
df_original = pd.read_csv('historico_vendas.csv')
le = LabelEncoder()
colunas_entrada = ['Categoria', 'TipoCliente']  # default
coluna_saida = 'Demanda'  # default

X = None
y = None
X_train = None
X_test = None
y_train = None
y_test = None

# ----- MODELOS -----
modelo_dt = DecisionTreeClassifier(random_state=42)
modelo_svm = SVC(kernel='linear')
pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='linear'))])

# ----- RESULTADOS -----
svm_testado = False
svm_pipeline_testado = False
dt_testado = False

acc_svm = None
acc_svm_pipeline = None
acc_dt = None

y_pred_dt = None
y_pred_svm = None
y_pred_svm_pipeline = None

# ----- PROCESSAMENTO DOS DADOS -----
def processar_dados():
    global X, y, X_train, X_test, y_train, y_test

    df = df_original.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    if not all(col in df.columns for col in colunas_entrada + [coluna_saida]):
        print(f"[!] Colunas inválidas. Verifique suas seleções.")
        return

    X = df[colunas_entrada]
    y = LabelEncoder().fit_transform(df[coluna_saida])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

# ----- SELEÇÃO DE COLUNAS PELO USUÁRIO -----
def escolher_colunas():
    global colunas_entrada, coluna_saida

    print("\nColunas disponíveis no CSV:")
    for i, col in enumerate(df_original.columns):
        print(f"{i + 1} - {col}")

    indices_entrada = input(
        "\nDigite os números das colunas de entrada (separados por vírgula): ")
    try:
        colunas_entrada = [df_original.columns[int(i) - 1].strip()
                           for i in indices_entrada.split(',')]
    except:
        print("[!] Entrada inválida. Usando colunas padrão.")
        colunas_entrada = ['Categoria', 'TipoCliente']

    indice_saida = input("Digite o número da coluna de saída: ")
    try:
        coluna_saida = df_original.columns[int(indice_saida) - 1].strip()
    except:
        print("[!] Entrada inválida. Usando saída padrão.")
        coluna_saida = 'Demanda'

    print(f"[✓] Entradas selecionadas: {colunas_entrada}")
    print(f"[✓] Saída selecionada: {coluna_saida}")

    processar_dados()

# ----- SUBMENU SVM -----
def submenu_svm():
    global svm_testado, svm_pipeline_testado
    global acc_svm, acc_svm_pipeline
    global y_pred_svm, y_pred_svm_pipeline

    while True:
        print("\n--- SUB-MENU SVM ---")
        print("0 - Escolher colunas de entrada e saída")
        print("1 - Testar o classificador SVM (Básico)")
        print("2 - Testar o classificador SVM (Pipeline)")
        print("3 - Mostrar o desempenho do SVM")
        print("4 - Menu Principal")
        opcao = input("Escolha uma opção: ")

        if opcao == '0':
            escolher_colunas()
        elif opcao == '1':
            modelo_svm.fit(X_train, y_train)
            y_pred_svm = modelo_svm.predict(X_test)
            acc_svm = accuracy_score(y_test, y_pred_svm)
            svm_testado = True
            print("[✓] Classificador SVM (Básico) testado com sucesso!")
        elif opcao == '2':
            pipeline_svm.fit(X_train, y_train)
            y_pred_svm_pipeline = pipeline_svm.predict(X_test)
            acc_svm_pipeline = accuracy_score(y_test, y_pred_svm_pipeline)
            svm_pipeline_testado = True
            print("[✓] Classificador SVM (Pipeline) testado com sucesso!")
        elif opcao == '3':
            if svm_testado:
                print(f"Acurácia do SVM (Básico): {acc_svm * 100:.2f}%")
            else:
                print("[!] Classificador SVM (Básico) não testado.")

            if svm_pipeline_testado:
                print(f"Acurácia do SVM (Pipeline): {acc_svm_pipeline * 100:.2f}%")
            else:
                print("[!] Classificador SVM (Pipeline) não testado.")
        elif opcao == '4':
            break
        else:
            print("[!] Opção inválida. Tente novamente.")

# ----- SUBMENU Árvore de Decisão -----
def submenu_dt():
    global dt_testado, acc_dt, y_pred_dt
    while True:
        print("\n--- SUB-MENU Árvore de Decisão ---")
        print("0 - Escolher colunas de entrada e saída")
        print("1 - Testar o classificador Árvore de Decisão")
        print("2 - Mostrar o desempenho da Árvore de Decisão")
        print("3 - Menu Principal")
        opcao = input("Escolha uma opção: ")

        if opcao == '0':
            escolher_colunas()
        elif opcao == '1':
            modelo_dt.fit(X_train, y_train)
            y_pred_dt = modelo_dt.predict(X_test)
            acc_dt = accuracy_score(y_test, y_pred_dt)
            dt_testado = True
            print("[✓] Classificador Árvore de Decisão testado com sucesso!")
        elif opcao == '2':
            if dt_testado:
                print(f"Acurácia da Árvore de Decisão: {acc_dt * 100:.2f}%")
            else:
                print("[!] Classificador Árvore de Decisão não testado.")
        elif opcao == '3':
            break
        else:
            print("[!] Opção inválida. Tente novamente.")

# ----- RELATÓRIO FINAL -----
def exibir_relatorio():
    if not dt_testado and not svm_testado and not svm_pipeline_testado:
        print("[!] Nenhum classificador foi testado ainda.")
        return

    print("\n===== RELATÓRIO DE ACURÁCIA =====")
    if dt_testado:
        print(f"Acurácia Árvore de Decisão: {acc_dt * 100:.2f}%")
    else:
        print("Acurácia Árvore de Decisão: Não testado")

    if svm_testado:
        print(f"Acurácia SVM (Básico): {acc_svm * 100:.2f}%")
    else:
        print("Acurácia SVM (Básico): Não testado")

    if svm_pipeline_testado:
        print(f"Acurácia SVM (Pipeline): {acc_svm_pipeline * 100:.2f}%")
    else:
        print("Acurácia SVM (Pipeline): Não testado")

    classificadores = {
        'Árvore de Decisão': acc_dt if dt_testado else 0,
        'SVM (Básico)': acc_svm if svm_testado else 0,
        'SVM (Pipeline)': acc_svm_pipeline if svm_pipeline_testado else 0
    }

    melhor = max(classificadores, key=classificadores.get)
    if classificadores[melhor] > 0:
        print(f"\n[✓] Melhor classificador: {melhor}")

# ----- MENU PRINCIPAL -----
def menu_principal():
    processar_dados()  # Inicialização padrão
    while True:
        print("\n===== MENU PRINCIPAL =====")
        print("1 - SVM")
        print("2 - Árvore de Decisão")
        print("3 - Exibir relatório dos classificadores")
        print("4 - Sair")
        opcao = input("Escolha uma opção: ")

        if opcao == '1':
            submenu_svm()
        elif opcao == '2':
            submenu_dt()
        elif opcao == '3':
            exibir_relatorio()
        elif opcao == '4':
            print("Programa finalizado!")
            break
        else:
            print("[!] Opção inválida. Tente novamente.")

# ----- INICIAR PROGRAMA -----
menu_principal()
