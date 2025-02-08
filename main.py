from src.data_preprocessing import carregar_dados, preprocessar_dados
from src.model_training import treinar_modelo, avaliar_modelo
from src.visualization import salvar_arvore

def main():
    caminho_dados = 'data/raw/INFLUD21-01-05-2023.csv'
    df = carregar_dados(caminho_dados)
    df_processado = preprocessar_dados(df)
    
    modelo, X_test, y_test = treinar_modelo(df_processado)
    avaliar_modelo(modelo, X_test, y_test)
    
    salvar_arvore(modelo, 'reports/figures/arvore_decisao')

if __name__ == '__main__':
    main()
