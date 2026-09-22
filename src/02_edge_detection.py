"""
02_bordes.py - deteccion de bordes con 6 operadores
operadores: canny, laplaciano, sobel, prewitt, roberts, fft
"""

import cv2
import os
import numpy as np


dir_proc = "data/processed"
dir_masks = "results/masks"
operadores = ["canny", "laplaciano", "sobel", "prewitt", "roberts", "fft"]


def detectar_canny(img):
    """canny: gradiente + non-maximum suppression + histéresis"""
    # Aplicar un poco de suavizado primero para reducir ruido
    img_suavizado = cv2.GaussianBlur(img, (5, 5), 1.5)
    edges = cv2.Canny(img_suavizado, 50, 150)
    return edges


def detectar_laplaciano(img):
    """laplaciano: segunda derivada"""
    lap = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F)
    magnitud = np.abs(lap)
    umbral = np.percentile(magnitud, 98.5)
    binaria = np.where(magnitud >= max(umbral, 1.0), 255, 0).astype(np.uint8)
    return binaria


def detectar_sobel(img):
    """sobel: derivadas en x e y"""
    sobelx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitud = np.sqrt(sobelx**2 + sobely**2)
    umbral = np.percentile(magnitud, 97.0)
    binaria = np.where(magnitud >= max(umbral, 1.0), 255, 0).astype(np.uint8)
    return binaria


def detectar_prewitt(img):
    """prewitt: derivadas con kernels specificos"""
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    gx = cv2.filter2D(img.astype(np.float32), cv2.CV_32F, kx)
    gy = cv2.filter2D(img.astype(np.float32), cv2.CV_32F, ky)
    magnitud = np.sqrt(gx**2 + gy**2)
    umbral = np.percentile(magnitud, 97.0)
    binaria = np.where(magnitud >= max(umbral, 1.0), 255, 0).astype(np.uint8)
    return binaria


def detectar_roberts(img):
    """roberts: derivadas diagonales"""
    kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gx = cv2.filter2D(img.astype(np.float32), cv2.CV_32F, kx)
    gy = cv2.filter2D(img.astype(np.float32), cv2.CV_32F, ky)
    magnitud = np.sqrt(gx**2 + gy**2)
    umbral = np.percentile(magnitud, 97.0)
    binaria = np.where(magnitud >= max(umbral, 1.0), 255, 0).astype(np.uint8)
    return binaria


def detectar_fft(img):
    """
    fft: filtro paso-altas en dominio de frecuencias
    parámetros optimizados para detección de grietas
    """
    f = np.fft.fft2(img.astype(np.float64))
    fshift = np.fft.fftshift(f)

    h, w = img.shape
    cy, cx = h // 2, w // 2
    radio = min(h, w) * 0.12

    # mascara: deja pasar altas frecuencias (bordes)
    # usando gaussian falloff para transición suave
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mascara = 1.0 - np.exp(-(dist**2) / (2 * radio**2))

    fshift_filtrada = fshift * mascara
    f_ishift = np.fft.ifftshift(fshift_filtrada)
    img_reconstruida = np.fft.ifft2(f_ishift)
    img_reconstruida = np.abs(img_reconstruida)

    umbral = np.percentile(img_reconstruida, 99.0)
    binaria = np.where(img_reconstruida >= max(umbral, 1e-8), 255, 0).astype(np.uint8)
    return binaria


mapa_detectores = {
    "canny": detectar_canny,
    "laplaciano": detectar_laplaciano,
    "sobel": detectar_sobel,
    "prewitt": detectar_prewitt,
    "roberts": detectar_roberts,
    "fft": detectar_fft,
}


def limpiar_ruido(binaria):
    """Conecta discontinuidades sin eliminar grietas de un pixel."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=1)


def ejecutar():
    """aplica todos los operadores a todas las imagenes con validación"""
    if not os.path.exists(dir_proc):
        print(f"  error: directorio {dir_proc} no existe. ejecuta 01_preprocessing.py primero")
        return

    imgs = [f for f in os.listdir(dir_proc) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not imgs:
        print(f"  error: no hay imagenes procesadas en {dir_proc}")
        return

    print(f"  detectando bordes en {len(imgs)} imagenes con {len(operadores)} operadores...")

    for nom_op in operadores:
        os.makedirs(os.path.join(dir_masks, nom_op), exist_ok=True)

    procesadas = 0
    for nombre in imgs:
        ruta_img = os.path.join(dir_proc, nombre)
        img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [!] no se pudo leer {nombre}")
            continue

        try:
            for nom_op, detector in mapa_detectores.items():
                mascara = detector(img)
                mascara_limpia = limpiar_ruido(mascara)
                ruta_out = os.path.join(dir_masks, nom_op, nombre)
                cv2.imwrite(ruta_out, mascara_limpia)
            procesadas += 1
        except Exception as e:
            print(f"  [!] error detectando bordes en {nombre}: {e}")

    print(f"  [OK] {procesadas}/{len(imgs)} mascaras guardadas en {dir_masks}/<operador>/")


if __name__ == "__main__":
    ejecutar()