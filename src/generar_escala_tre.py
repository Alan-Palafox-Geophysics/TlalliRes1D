import matplotlib.colors as mcolors


def generar_escala_tre():
    """
    Genera un mapa de color personalizado pasando exactamente por la 
    secuencia de colores solicitada para datos de TRE.
    """
    
    # Lista de colores en orden (de menor a mayor valor)
    colores = [
        "#400080",  # 1. Azul casi morado
        "#0000AA",  # 2. Azul RGB normal
        "#0000ff",  # 2. Azul RGB normal
        "#0064FF",  # 3. Azul
        "#00ffff",  # 4. Cian
        "#00FFAA",  # 5. Cian-Verde
        "#00FF46",  # 6. Verde-Cian
        "#00ff00",  # 7. Verde RGB normal
        "#00AA00",  # 8. Verde opaco
        "#55AA0A",  # 9. Verde-Amarillento
        "#AAFF0A",  # 10. Amarillo-Verdoso
        "#FFFF0A",  # 11. Amarillo
        "#FF7D0A",  # 12. Anaranjado
        "#AA550A",  # 13. Anaranjado Oscuro
        "#FF000A",  # 14. Rojo RGB normal
        "#800040"   # 15. Rojo oscuro
    ]
    
    # Crear el colormap a partir de la lista
    # N=256 asegura que la interpolación entre los colores sea muy suave
    cmap_tre = mcolors.LinearSegmentedColormap.from_list("Escala_TRE_Detallada", colores, N=256)
    
    return cmap_tre