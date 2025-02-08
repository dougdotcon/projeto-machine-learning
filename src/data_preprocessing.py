import pandas as pd
import numpy as np

def carregar_dados(caminho: str) -> pd.DataFrame:
    """
    Carrega os dados de um arquivo CSV.

    Args:
        caminho (str): Caminho para o arquivo CSV.

    Returns:
        pd.DataFrame: DataFrame com os dados carregados.
    """
    try:
        df = pd.read_csv(caminho, sep=';', encoding='utf-8')
        print("Dados carregados com sucesso!")
    except Exception as e:
        print("Erro ao carregar os dados. Verifique o caminho ou a URL.")
        df = pd.DataFrame()
    return df

def mapear_resposta(valor):
    """
    Mapeia os valores das respostas:
      - Se o valor for NaN ou 9, retorna np.nan.
      - Se o valor for 1, retorna 1; caso contrário, retorna 0.
    
    Args:
        valor: Valor a ser mapeado.

    Returns:
        int ou np.nan: Resultado do mapeamento.
    """
    if pd.isna(valor):
        return np.nan
    if valor == 9:
        return np.nan
    return 1 if valor == 1 else 0

def preprocessar_dados(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Realiza o pré-processamento dos dados:
      - Seleciona as colunas relevantes.
      - Converte colunas de datas para o formato datetime.
      - Aplica mapeamento para as colunas binárias.
      - Cria a coluna 'idade' com base em 'nu_idade_n' ou 'dt_nasc'.
      - Cria a coluna 'target' a partir da coluna 'CLASSI_FIN'.
      - Mapeia a coluna 'cs_sexo' para valores numéricos.

    Args:
        df (pd.DataFrame): DataFrame original.
        verbose (bool): Se True, imprime informações durante o processo.

    Returns:
        pd.DataFrame: DataFrame pré-processado.
    """
    # Seleção das colunas de interesse
    colunas_selecionadas = [
        'dt_notific', 'dt_sin_pri', 'dt_nasc', 'nu_idade_n', 'cs_sexo',
        'FEBRE', 'TOSSE', 'GARGANTA', 'DISPNEIA', 'DESC_RESP', 'SATURACAO',
        'DIARREIA', 'VOMITO', 'DOR_ABD', 'FADIGA', 'PERD_OLFT', 'PERD_PALA',
        'FATOR_RISC', 'PUERPERA', 'CARDIOPATI', 'HEMATOLOGI', 'SIND_DOWN',
        'HEPATICA', 'ASMA', 'DIABETES', 'NEUROLOGIC', 'PNEUMOPATI', 'IMUNODEPRE',
        'RENAL', 'OBESIDADE', 'OBES_IMC', 'OUT_MORBI', 'CS_GESTANT',
        'VACINA_COV', 'UTI', 'ANTIVIRAL', 'CLASSI_FIN', 'EVOLUCAO', 'TRAT_COV'
    ]
    colunas_existentes = [col for col in colunas_selecionadas if col in df.columns]
    if verbose:
        print(f"Colunas selecionadas disponíveis: {colunas_existentes}")
    df = df[colunas_existentes].copy()

    # Converter colunas de datas para datetime
    for coluna in ['dt_notific', 'dt_sin_pri', 'dt_nasc']:
        if coluna in df.columns:
            df[coluna] = pd.to_datetime(df[coluna], errors='coerce')

    # Aplicar mapeamento para as colunas binárias
    colunas_binarias = [
        'FEBRE', 'TOSSE', 'GARGANTA', 'DISPNEIA', 'DESC_RESP', 'SATURACAO',
        'DIARREIA', 'VOMITO', 'DOR_ABD', 'FADIGA', 'PERD_OLFT', 'PERD_PALA',
        'FATOR_RISC', 'PUERPERA', 'CARDIOPATI', 'HEMATOLOGI', 'SIND_DOWN',
        'HEPATICA', 'ASMA', 'DIABETES', 'NEUROLOGIC', 'PNEUMOPATI', 'IMUNODEPRE',
        'RENAL', 'OBESIDADE', 'OBES_IMC', 'OUT_MORBI', 'CS_GESTANT'
    ]
    for coluna in colunas_binarias:
        if coluna in df.columns:
            df[coluna] = df[coluna].apply(mapear_resposta)

    # Criação da coluna 'idade'
    if 'nu_idade_n' in df.columns:
        df['idade'] = df['nu_idade_n']
    elif 'dt_nasc' in df.columns:
        df['idade'] = pd.to_datetime('today').year - df['dt_nasc'].dt.year
    else:
        if verbose:
            print("Colunas para cálculo de idade não encontradas. A coluna 'idade' não será criada.")
        df['idade'] = np.nan

    # Criação da coluna 'target' a partir de 'CLASSI_FIN'
    if 'CLASSI_FIN' in df.columns:
        df['target'] = df['CLASSI_FIN'].apply(lambda x: 1 if x == 5 else 0)
    else:
        if verbose:
            print("Coluna CLASSI_FIN não encontrada. A coluna 'target' não será criada.")

    # Mapeamento da coluna 'cs_sexo'
    if 'cs_sexo' in df.columns:
        df['cs_sexo'] = df['cs_sexo'].map({'M': 0, 'F': 1})

    if verbose:
        print("Resumo do DataFrame pré-processado:")
        print(df.info())
        print(df.head())

    return df

def preparar_modelo(df: pd.DataFrame, verbose: bool = True):
    """
    Prepara o DataFrame para o treinamento do modelo:
      - Define as features que serão utilizadas (incluindo 'idade' se disponível e as colunas binárias).
      - Remove as linhas com valores faltantes nas features ou na coluna 'target'.

    Args:
        df (pd.DataFrame): DataFrame pré-processado.
        verbose (bool): Se True, imprime informações durante o processo.

    Returns:
        tuple: (df_model, features, target)
            df_model (pd.DataFrame): DataFrame pronto para treinamento.
            features (list): Lista de colunas utilizadas como features.
            target (str): Nome da coluna alvo.
    """
    features = []
    if 'idade' in df.columns and df['idade'].notna().sum() > 0:
        features.append('idade')
    else:
        if verbose:
            print("Coluna 'idade' não possui dados válidos e será desconsiderada nas features.")

    colunas_binarias = [
        'FEBRE', 'TOSSE', 'GARGANTA', 'DISPNEIA', 'DESC_RESP', 'SATURACAO',
        'DIARREIA', 'VOMITO', 'DOR_ABD', 'FADIGA', 'PERD_OLFT', 'PERD_PALA',
        'FATOR_RISC', 'PUERPERA', 'CARDIOPATI', 'HEMATOLOGI', 'SIND_DOWN',
        'HEPATICA', 'ASMA', 'DIABETES', 'NEUROLOGIC', 'PNEUMOPATI', 'IMUNODEPRE',
        'RENAL', 'OBESIDADE', 'OBES_IMC', 'OUT_MORBI', 'CS_GESTANT'
    ]
    for coluna in colunas_binarias:
        if coluna in df.columns:
            features.append(coluna)

    if verbose:
        print("Features utilizadas:", features)

    if 'target' not in df.columns:
        if verbose:
            print("Coluna 'target' não encontrada.")
        return df, features, None

    df_model = df.dropna(subset=features + ['target']).copy()
    return df_model, features, 'target'
