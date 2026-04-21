"""
01_preproceso.py - convierte imagenes a gris, redimensiona y mejora contraste
"""

import cv2
import os
import numpy as np

dir_raw = "data/raw"
dir_proc = "data/processed"
tamaño = (256, 256)


def filtro_media(img, ksize=3):
    """aplica filtro de media usando OpenCV (más eficiente que manual)"""
    return cv2.medianBlur(img, ksize)


def procesar_imagen(ruta_in, ruta_out):
    """lee imagen, aplica gris, redimensiona y filtro de media"""
    img = cv2.imread(ruta_in)
    if img is None:
        print(f"  advertencia: no se pudo leer {ruta_in}")
        return None

    # gris
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # redimensionar
    redim = cv2.resize(gris, tamaño)

    # filtro de media manual para suavizar y reducir ruido
    suavizada = filtro_media(redim, ksize=3)

    cv2.imwrite(ruta_out, suavizada)
    return suavizada


def ejecutar():
    """procesa todas las imagenes del directorio raw con validación"""
    os.makedirs(dir_proc, exist_ok=True)

    if not os.path.exists(dir_raw):
        print(f"  error: directorio {dir_raw} no existe")
        return False

    imgs = [f for f in os.listdir(dir_raw) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not imgs:
        print(f"  error: no hay imagenes en {dir_raw}")
        return False

    print(f"  procesando {len(imgs)} imagenes...")

    procesadas = 0
    for nombre in imgs:
        ruta_in = os.path.join(dir_raw, nombre)
        ruta_out = os.path.join(dir_proc, nombre)
        try:
            resultado = procesar_imagen(ruta_in, ruta_out)
            if resultado is not None:
                procesadas += 1
        except Exception as e:
            print(f"  [!] error procesando {nombre}: {e}")

    print(f"  [OK] {procesadas}/{len(imgs)} imagenes guardadas en {dir_proc}/")
    return procesadas > 0


if __name__ == "__main__":
    ejecutar()