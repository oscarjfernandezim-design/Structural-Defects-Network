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


def convolucionar(img, kernel):
    """aplicar convolucion manual """
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    resultado = np.zeros_like(img, dtype=np.float64)
    
    for i in range(pad_h, h - pad_h):
        for j in range(pad_w, w - pad_w):
            ventana = img[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1]
            resultado[i, j] = np.sum(ventana * kernel)
    
    return resultado


def detectar_canny(img):
    """canny: gradiente + non-maximum suppression + histéresis"""
    base = img.astype(np.float64)
    
    # kernels de Sobel para derivadas
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    
    gx = convolucionar(base, kx)
    gy = convolucionar(base, ky)
    
    magnitud = np.sqrt(gx**2 + gy**2)
    angulo = np.arctan2(gy, gx)
    
    # normalizar magnitud
    magnitud_norm = np.uint8(np.clip(magnitud / (magnitud.max() + 1e-8) * 255, 0, 255))
    
    # umbralizado con histéresis (simulado)
    _, binaria = cv2.threshold(magnitud_norm, 50, 255, cv2.THRESH_BINARY)
    return binaria


def detectar_laplaciano(img):
    """laplaciano: segunda derivada (kernel 3x3)"""
    base = img.astype(np.float64)
    
    # kernel laplaciano
    kernel_lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    
    lap = convolucionar(base, kernel_lap)
    lap = np.uint8(np.clip(np.abs(lap), 0, 255))
    _, binaria = cv2.threshold(lap, 20, 255, cv2.THRESH_BINARY)
    return binaria


def detectar_sobel(img):
    """sobel: derivadas en x e y"""
    base = img.astype(np.float64)
    
    # kernels sobel
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    
    gx = convolucionar(base, kx)
    gy = convolucionar(base, ky)
    magnitud = np.sqrt(gx**2 + gy**2)
    magnitud = np.uint8(np.clip(magnitud / (magnitud.max() + 1e-8) * 255, 0, 255))
    _, binaria = cv2.threshold(magnitud, 30, 255, cv2.THRESH_BINARY)
    return binaria


def detectar_prewitt(img):
    """prewitt: derivadas con kernels specificos"""
    base = img.astype(np.float64)
    
    # kernels prewitt
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    
    gx = convolucionar(base, kx)
    gy = convolucionar(base, ky)
    magnitud = np.sqrt(gx**2 + gy**2)
    magnitud = np.uint8(np.clip(magnitud / (magnitud.max() + 1e-8) * 255, 0, 255))
    _, binaria = cv2.threshold(magnitud, 30, 255, cv2.THRESH_BINARY)
    return binaria


def detectar_roberts(img):
    """roberts: derivadas diagonales (kernels 2x2)"""
    base = img.astype(np.float64)
    
    # kernels roberts expandidos a 3x3
    kx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float64)
    ky = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], dtype=np.float64)
    
    gx = convolucionar(base, kx)
    gy = convolucionar(base, ky)
    magnitud = np.sqrt(gx**2 + gy**2)
    magnitud = np.uint8(np.clip(magnitud / (magnitud.max() + 1e-8) * 255, 0, 255))
    _, binaria = cv2.threshold(magnitud, 20, 255, cv2.THRESH_BINARY)
    return binaria


def detectar_fft(img):
    """fft: filtro paso-altas en dominio de frecuencias"""
    f = np.fft.fft2(img.astype(np.float64))
    fshift = np.fft.fftshift(f)
    
    h, w = img.shape
    cy, cx = h // 2, w // 2
    radio = 30
    
    # mascara: deja pasar altas frecuencias (bordes)
    mascara = np.ones((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            dist = np.sqrt((j - cx)**2 + (i - cy)**2)
            if dist <= radio:
                mascara[i, j] = 0
    
    fshift_filtrada = fshift * mascara
    f_ishift = np.fft.ifftshift(fshift_filtrada)
    img_reconstruida = np.fft.ifft2(f_ishift)
    img_reconstruida = np.abs(img_reconstruida)
    
    img_reconstruida = np.uint8(np.clip(img_reconstruida / (img_reconstruida.max() + 1e-8) * 255, 0, 255))
    _, binaria = cv2.threshold(img_reconstruida, 25, 255, cv2.THRESH_BINARY)
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
    """abre y cierra para limpiar ruido"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    abierta = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)
    cerrada = cv2.morphologyEx(abierta, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cerrada


def ejecutar():
    """aplica todos los operadores a todas las imagenes"""
    imgs = [f for f in os.listdir(dir_proc) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"  detectando bordes en {len(imgs)} imagenes con {len(operadores)} operadores...")

    for nom_op in operadores:
        os.makedirs(os.path.join(dir_masks, nom_op), exist_ok=True)

    for nombre in imgs:
        ruta_img = os.path.join(dir_proc, nombre)
        img = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        for nom_op, detector in mapa_detectores.items():
            mascara = detector(img)
            mascara_limpia = limpiar_ruido(mascara)
            ruta_out = os.path.join(dir_masks, nom_op, nombre)
            cv2.imwrite(ruta_out, mascara_limpia)

    print(f"  ✓ mascaras guardadas en {dir_masks}/<operador>/")


if __name__ == "__main__":
    ejecutar()