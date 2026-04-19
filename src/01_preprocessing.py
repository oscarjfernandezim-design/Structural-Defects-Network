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
    """aplica filtro de media(kernel 3x3)"""
    h, w = img.shape
    pad = ksize // 2
    resultado = np.zeros_like(img, dtype=np.uint8)
    
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            ventana = img[i-pad:i+pad+1, j-pad:j+pad+1]
            resultado[i, j] = np.median(ventana)
    
    return resultado


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
    """procesa todas las imagenes del directorio raw"""
    os.makedirs(dir_proc, exist_ok=True)
    
    if not os.path.exists(dir_raw):
        print(f"  error: directorio {dir_raw} no existe")
        return
    
    imgs = [f for f in os.listdir(dir_raw) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"  procesando {len(imgs)} imagenes...")
    
    for nombre in imgs:
        ruta_in = os.path.join(dir_raw, nombre)
        ruta_out = os.path.join(dir_proc, nombre)
        procesar_imagen(ruta_in, ruta_out)

    print(f"  ✓ imagenes guardadas en {dir_proc}/")


if __name__ == "__main__":
    ejecutar()