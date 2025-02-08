import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

def treinar_modelo(df: pd.DataFrame, features: list, target: str, test_size: float = 0.3, random_state: int = 42):
    """
    Divide os dados, treina um modelo de árvore de decisão e retorna o modelo treinado junto com os conjuntos de dados.

    Args:
        df (pd.DataFrame): DataFrame pré-processado.
        features (list): Lista de colunas que serão utilizadas como features.
        target (str): Nome da coluna alvo.
        test_size (float, opcional): Proporção de dados para o conjunto de teste. Padrão é 0.3.
        random_state (int, opcional): Semente para a aleatoriedade. Padrão é 42.

    Returns:
        tuple: (modelo, X_train, X_test, y_train, y_test)
    """
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    modelo = DecisionTreeClassifier(random_state=random_state, max_leaf_nodes=10)
    modelo.fit(X_train, y_train)
    print("Treinamento concluído.")
    
    return modelo, X_train, X_test, y_train, y_test

def avaliar_modelo(modelo, X_test, y_test):
    """
    Avalia o modelo treinado utilizando métricas de classificação e imprime os resultados.

    Args:
        modelo: Modelo treinado.
        X_test (pd.DataFrame): Conjunto de dados de teste (features).
        y_test (pd.Series): Conjunto de dados de teste (target).
    """
    y_pred = modelo.predict(X_test)
    
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

def salvar_modelo(modelo, caminho_saida: str):
    """
    Salva o modelo treinado em um arquivo utilizando joblib.

    Args:
        modelo: Modelo treinado.
        caminho_saida (str): Caminho para salvar o modelo (ex.: 'models/modelo_arvore_decisao.pkl').
    """
    joblib.dump(modelo, caminho_saida)
    print(f"Modelo salvo em {caminho_saida}")
