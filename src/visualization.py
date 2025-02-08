from sklearn.tree import export_graphviz
import graphviz

def salvar_arvore_png(modelo, feature_names: list, class_names: list, caminho_saida: str):
    """
    Gera e salva a visualização da árvore de decisão em formato PNG.

    Args:
        modelo: Modelo treinado da árvore de decisão.
        feature_names (list): Lista de nomes das features.
        class_names (list): Lista de nomes das classes.
        caminho_saida (str): Caminho de saída para salvar o arquivo PNG (ex.: 'reports/figures/arvore_decisao').
    """
    dot_data = export_graphviz(
        modelo,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render(caminho_saida, format='png', cleanup=True)
    print(f"Visualização salva em {caminho_saida}.png")

def salvar_arvore_svg(modelo, feature_names: list, class_names: list, caminho_saida: str):
    """
    Gera e salva a visualização da árvore de decisão em formato SVG.

    Args:
        modelo: Modelo treinado da árvore de decisão.
        feature_names (list): Lista de nomes das features.
        class_names (list): Lista de nomes das classes.
        caminho_saida (str): Caminho de saída para salvar o arquivo SVG (ex.: 'reports/figures/arvore_decisao').
    """
    dot_data = export_graphviz(
        modelo,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data, format="svg")
    svg_data = graph.pipe().decode("utf-8")
    with open(f"{caminho_saida}.svg", "w", encoding="utf-8") as f:
        f.write(svg_data)
    print(f"Visualização salva em {caminho_saida}.svg")
