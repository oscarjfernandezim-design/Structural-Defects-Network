"""
04_metricas.py - calcula metricas del daño con el mejor operador
cpr, longitud de grieta, componentes conectados, clasificacion
"""

import cv2
import os
import numpy as np
import pandas as pd

dir_masks = "results/masks"
csv_salida = "results_summary.csv"

# umbrales de clasificacion
umbral_leve = 5.0
umbral_moderado = 20.0


def esqueletizar_manual(mascara):
    """
    esqueletizacion simple: calcula el numero de pixeles conectados
    sin usar skimage, solo opencv y numpy
    """
    # conectar componentes pequeños
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # kernel de dilatacion
    dilatada = cv2.dilate(mascara, kernel, iterations=1) # dilatar para conectar componentes pequeños
    
    # contar pixeles en la estructura final
    return np.sum(dilatada > 0)


def calc_metricas(mascara):
    """calcula todas las metricas de una mascara"""
    total_pixeles = mascara.size
    pixeles_grieta = np.count_nonzero(mascara)
    cpr = (pixeles_grieta / total_pixeles) * 100

    # longitud de grieta (aproximacion)
    longitud_grieta = esqueletizar_manual(mascara)

    # componentes conectados
    num_etiquetas, _, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    num_componentes = num_etiquetas - 1
    areas = stats[1:, cv2.CC_STAT_AREA] if num_componentes > 0 else [0]
    area_max = int(np.max(areas)) if len(areas) > 0 else 0

    return {
        "cpr": round(cpr, 4),
        "longitud_grieta_px": int(longitud_grieta),
        "num_componentes": num_componentes,
        "area_max_componente": area_max,
    }


def clasificar_daño(cpr):
    """clasifica el daño segun el cpr"""
    if cpr < umbral_leve:
        return "leve"
    elif cpr < umbral_moderado:
        return "moderado"
    else:
        return "severo"


def ejecutar(mejor_op="canny"):
    """calcula metricas con el mejor operador con validación mejorada"""
    dir_op = os.path.join(dir_masks, mejor_op)

    if not os.path.exists(dir_op):
        print(f"  error: no encontrada carpeta de mascaras para {mejor_op}")
        return None

    imgs = sorted([f for f in os.listdir(dir_op) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not imgs:
        print(f"  error: no hay mascaras para el operador {mejor_op}")
        return None

    print(f"  calculando metricas de {len(imgs)} imagenes con {mejor_op}")

    registros = []
    for nombre in imgs:
        ruta_mascara = os.path.join(dir_op, nombre)
        mascara = cv2.imread(ruta_mascara, cv2.IMREAD_GRAYSCALE)
        if mascara is None:
            continue

        try:
            metricas = calc_metricas(mascara)
            daño = clasificar_daño(metricas["cpr"])

            registros.append({
                "imagen": nombre,
                "cpr": metricas["cpr"],
                "longitud_px": metricas["longitud_grieta_px"],
                "num_componentes": metricas["num_componentes"],
                "area_max": metricas["area_max_componente"],
                "grado_daño": daño,
                "operador": mejor_op,
            })
        except Exception as e:
            print(f"  [!] error procesando {nombre}: {e}")

    if not registros:
        print(f"  error: no se procesaron metricas para ninguna imagen")
        return None

    df = pd.DataFrame(registros)
    df.to_csv(csv_salida, index=False)

    # resumen
    print(f"\n  resumen de daño:")
    print(df["grado_daño"].value_counts().to_string())
    print(f"\n  estadísticas CPR:")
    print(f"    - Media: {df['cpr'].mean():.3f}%")
    print(f"    - Mínimo: {df['cpr'].min():.3f}%")
    print(f"    - Máximo: {df['cpr'].max():.3f}%")
    print(f"\n  [OK] {csv_salida} guardado con {len(df)} registros")
    return df


if __name__ == "__main__":
    ejecutar()