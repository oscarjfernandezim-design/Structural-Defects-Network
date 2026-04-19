# 🔍 Structural Damage Detection Pipeline | Pipeline de Detección de Daño Estructural

> **Advanced Computer Vision & Image Processing System** for automated crack and structural damage analysis in infrastructure
>
> **Sistema Avanzado de Visión Computacional** para análisis automatizado de grietas y daño estructural

---

## 📋 Table of Contents | Índice de Contenidos

- [🎯 Overview | Descripción General](#-overview--descripción-general)
- [✨ Key Features | Características Principales](#-key-features--características-principales)
- [🛠️ Tech Stack | Stack Tecnológico](#️-tech-stack--stack-tecnológico)
- [📁 Project Structure | Estructura del Proyecto](#-project-structure--estructura-del-proyecto)
- [🚀 Quick Start | Inicio Rápido](#-quick-start--inicio-rápido)
- [📊 Pipeline Details | Detalles del Pipeline](#-pipeline-details--detalles-del-pipeline)
- [🔬 Edge Detection Operators | Operadores de Detección de Bordes](#-edge-detection-operators--operadores-de-detección-de-bordes)
- [📈 Metrics & Classification | Métricas y Clasificación](#-metrics--classification--métricas-y-clasificación)
- [🎓 Academic Implementation | Implementación Académica](#-academic-implementation--implementación-académica)

---

## 🎯 Overview | Descripción General

### English
This project implements a **complete pipeline for structural damage detection** using computer vision and image processing techniques. It analyzes infrastructure images to automatically detect, measure, and classify cracks and other structural damage.

The system:
- ✅ Processes raw images with preprocessing techniques (grayscale, resizing, median filtering)
- ✅ Applies 6 different edge detection operators (Canny, Laplacian, Sobel, Prewitt, Roberts, FFT)
- ✅ Compares operators using Signal-to-Noise Ratio (SNR) and Crack Pixel Ratio (CPR)
- ✅ Classifies damage into severity levels: **Mild (CPR < 5%) | Moderate (5% ≤ CPR < 20%) | Severe (CPR ≥ 20%)**
- ✅ Generates comprehensive visualizations and statistical reports

### Español
Este proyecto implementa un **pipeline completo para detección de daño estructural** utilizando técnicas de visión computacional y procesamiento de imágenes. Analiza imágenes de infraestructura para detectar, medir y clasificar automáticamente grietas y otro daño estructural.

El sistema:
- ✅ Procesa imágenes con técnicas de preprocesamiento (escala de grises, redimensionamiento, filtro de media)
- ✅ Aplica 6 operadores diferentes de detección de bordes (Canny, Laplaciano, Sobel, Prewitt, Roberts, FFT)
- ✅ Compara operadores usando Relación Señal/Ruido (SNR) y Proporción de Píxeles de Grieta (CPR)
- ✅ Clasifica daño en niveles de severidad: **Leve (CPR < 5%) | Moderado (5% ≤ CPR < 20%) | Severo (CPR ≥ 20%)**
- ✅ Genera visualizaciones y reportes estadísticos exhaustivos

---

## ✨ Key Features | Características Principales

| Feature | Description | Característica | Descripción |
|---------|-------------|-----------------|-------------|
| **Manual Image Processing** | All algorithms implemented from scratch using NumPy | **Procesamiento Manual** | Todos los algoritmos implementados desde cero con NumPy |
| **6 Edge Detectors** | Canny, Laplacian, Sobel, Prewitt, Roberts, FFT | **6 Detectores de Bordes** | Canny, Laplaciano, Sobel, Prewitt, Roberts, FFT |
| **Custom Convolution** | Manual convolution kernels for each operator | **Convolución Personalizada** | Kernels de convolución manual para cada operador |
| **Adaptive Median Filter** | Reduces noise while preserving edge information | **Filtro de Media Adaptativo** | Reduce ruido preservando información de bordes |
| **Automated Metrics** | CPR, SNR, Connected Components, Crack Length | **Métricas Automatizadas** | CPR, SNR, Componentes Conectados, Longitud de Grieta |
| **Multi-Level Classification** | 3-tier damage severity system | **Clasificación Multinivel** | Sistema de severidad de daño de 3 niveles |
| **Visual Analytics** | Histograms, comparisons, spectral analysis | **Análisis Visual** | Histogramas, comparaciones, análisis espectral |
| **Batch Processing** | Process entire image directories automatically | **Procesamiento por Lotes** | Procesar directorios completos automáticamente |

---

## 🛠️ Tech Stack | Stack Tecnológico

```
Python 3.8+
├── OpenCV (cv2) - Image processing & morphological operations
├── NumPy - Matrix operations & numerical computations
├── Pandas - Data analysis & CSV export
├── Matplotlib - Scientific visualization
└── FFT - Frequency domain analysis
```

**No external ML frameworks** - Pure image processing and computer vision algorithms

---

## 📁 Project Structure | Estructura del Proyecto

```
.
├── ejecutar.py                      # Main orchestration script | Script principal
├── requirements.txt                 # Project dependencies | Dependencias
├── README.md                        # This file | Este archivo
│
├── src/                            # Source code | Código fuente
│   ├── 01_preprocessing.py         # Image preprocessing (grayscale, median filter)
│   ├── 02_edge_detection.py        # Edge detection operators (6 methods)
│   ├── 03_fft_filter.py            # Comparison & metrics calculation
│   ├── 04_comparison.py            # Damage metrics & classification
│   ├── 05_metrics.py               # Statistical graphics (legacy)
│   ├── 06_visualization.py         # Final visualizations & reports
│   └── main.py                     # Legacy entry point
│
├── data/
│   ├── raw/                        # Input images | Imágenes de entrada
│   └── processed/                  # Preprocessed images | Imágenes preprocesadas
│
└── results/
    ├── masks/                      # Detection masks per operator
    │   ├── canny/
    │   ├── laplaciano/
    │   ├── sobel/
    │   ├── prewitt/
    │   ├── roberts/
    │   └── fft/
    ├── graphs/                     # Analysis charts & comparisons
    │   ├── comparacion_operadores.png
    │   ├── 01_histograma_cpr.png
    │   └── ...
    ├── visualizations/             # Final visualizations
    └── results_summary.csv          # Final metrics table

```

---

## 🚀 Quick Start | Inicio Rápido

### Prerequisites | Requisitos Previos
```bash
python --version  # Python 3.8 or higher | 3.8 o superior
pip install -r requirements.txt
```

### Installation | Instalación
```bash
git clone <repository-url>
cd parcial
pip install -r requirements.txt
```

### Usage | Uso

#### 1️⃣ Place Your Images | Coloca tus Imágenes
```
data/raw/
├── image1.jpg
├── image2.jpg
└── image3.jpg
```

#### 2️⃣ Run the Pipeline | Ejecuta el Pipeline
```bash
python ejecutar.py
```

#### 3️⃣ Check Results | Verifica Resultados
```
results/
├── results_summary.csv              # Metrics table
├── graphs/
│   └── comparacion_operadores.png   # Operator comparison
└── visualizations/
    └── 03_mosaico_comparacion.png   # Detection results
```

---

## 📊 Pipeline Details | Detalles del Pipeline

```
INPUT IMAGES (raw)
        ↓
[1] PREPROCESSING
    ├── Convert to Grayscale
    ├── Resize to 256x256
    └── Apply Median Filter (3x3)
        ↓
    PREPROCESSED IMAGES
        ↓
[2] EDGE DETECTION (6 Operators)
    ├── Canny (Gradient + Hystheresis)
    ├── Laplacian (2nd Derivative)
    ├── Sobel (Directional Gradients)
    ├── Prewitt (Edge Emphasis)
    ├── Roberts (Diagonal Gradients)
    └── FFT (Frequency Domain)
        ↓
    EDGE MASKS (per operator)
        ↓
[3] OPERATOR COMPARISON
    ├── Calculate CPR (Crack Pixel Ratio)
    ├── Calculate SNR (Signal-to-Noise Ratio)
    └── Select Best Operator
        ↓
[4] DAMAGE METRICS
    ├── CPR Calculation
    ├── Connected Components
    ├── Crack Length Approximation
    └── Severity Classification
        ↓
[5] VISUALIZATION
    ├── CPR Histogram
    ├── Damage Category Distribution
    ├── Detection Mosaics
    └── FFT Spectrum Analysis
        ↓
OUTPUT: results_summary.csv + graphs/
```

---

## 🔬 Edge Detection Operators | Operadores de Detección de Bordes

### Implementation Details | Detalles de Implementación

#### **Canny Edge Detector**
```python
# Sobel kernels (custom implementation)
Gx = [-1, 0, 1]    Gy = [-1, -2, -1]
     [-2, 0, 2]         [ 0,  0,  0]
     [-1, 0, 1]         [ 1,  2,  1]

Magnitude = √(Gx² + Gy²)
Direction = atan2(Gy, Gx)
# Hysteresis thresholding (threshold: 50-150)
```

#### **Laplacian Operator**
```python
Kernel = [0, -1, 0]
         [-1, 4, -1]
         [0, -1, 0]
# 2nd derivative (threshold: 20)
```

#### **Sobel Operator**
```python
Gx = [-1, 0, 1]    Gy = [-1, -2, -1]
     [-2, 0, 2]         [ 0,  0,  0]
     [-1, 0, 1]         [ 1,  2,  1]
# Magnitude (threshold: 30)
```

#### **Prewitt Operator**
```python
Gx = [-1, 0, 1]    Gy = [-1, -1, -1]
     [-1, 0, 1]         [ 0,  0,  0]
     [-1, 0, 1]         [ 1,  1,  1]
# Magnitude (threshold: 30)
```

#### **Roberts Operator**
```python
Gx = [1, 0]        Gy = [0, 1]
     [0, -1]            [-1, 0]
# Diagonal gradients (threshold: 20)
```

#### **FFT Operator**
```python
1. FFT2(image)
2. High-pass filter (remove center frequency components)
3. Inverse FFT
4. Magnitude thresholding (threshold: 25)
```

**All operators use custom convolution kernels - no OpenCV functions!** | **¡Todos los operadores usan kernels de convolución personalizados!**

---

## 📈 Metrics & Classification | Métricas y Clasificación

### Damage Severity Levels | Niveles de Severidad del Daño

| Level | CPR Range | Interpretation | Color |
|-------|-----------|-----------------|-------|
| 🟢 **Mild** | CPR < 5% | Minimal structural impact | Green |
| 🟡 **Moderate** | 5% ≤ CPR < 20% | Significant damage, repair recommended | Yellow |
| 🔴 **Severe** | CPR ≥ 20% | Critical damage, immediate action required | Red |

### Key Metrics | Métricas Clave

1. **CPR (Crack Pixel Ratio)** = (Crack Pixels / Total Pixels) × 100
   - Measures percentage of image covered by cracks | Mide el porcentaje de imagen cubierto por grietas

2. **SNR (Signal-to-Noise Ratio)** = Signal / Noise
   - Compares largest component (signal) vs small components (noise) | Compara componentes grandes vs ruido

3. **Connected Components** = Number of distinct crack regions
   - Indicates fragmentation level | Indica nivel de fragmentación

4. **Crack Length (px)** = Approximation via dilation
   - Estimated crack length in pixels | Longitud de grieta estimada

---

## 🎓 Academic Implementation | Implementación Académica

This project demonstrates **core image processing concepts** without relying on high-level ML frameworks:

✅ **Manual Convolution** - Understand kernel operations pixel-by-pixel
✅ **Edge Detection Algorithms** - Implement 6 different mathematical approaches
✅ **Frequency Domain Analysis** - FFT-based filtering
✅ **Morphological Operations** - Dilation, erosion, opening, closing
✅ **Connected Component Labeling** - Graph-based component identification
✅ **Statistical Analysis** - CPR, SNR, classification metrics

**Perfect for:** Students, engineers, and practitioners learning computer vision fundamentals

---

## 📊 Sample Output | Salida de Ejemplo

### Operator Comparison
```
       operador  cpr_promedio  snr_promedio
0          Canny         4.25          8.32
1      Laplaciano         6.18          5.94
2          Sobel         5.87          7.21
3         Prewitt         5.92          7.15
4        Roberts         4.98          6.54
5            Fft         3.45          4.28
```

### Damage Classification
```
Mild:       45 images (🟢)
Moderate:   32 images (🟡)
Severe:      8 images (🔴)
```

---

## 🔧 Configuration | Configuración

Edit thresholds in `src/04_comparison.py`:
```python
umbral_leve = 5.0        # Mild threshold
umbral_moderado = 20.0   # Moderate threshold
```

Edit filter parameters in `src/01_preprocessing.py`:
```python
tamaño = (256, 256)      # Image size
ksize = 3                # Median filter kernel size
```

---

## 📝 License | Licencia

This project is part of an academic course (Semester 7 - Signal Processing)

---

## 👨‍💻 Development Notes | Notas de Desarrollo

- **Language**: Python 3.8+
- **Code Style**: PEP 8 compliant
- **Documentation**: Bilingual (English/Spanish)
- **Testing**: Batch processing on ~85 infrastructure images

### Key Files Overview | Descripción de Archivos Clave

| File | Purpose | Estado |
|------|---------|--------|
| `ejecutar.py` | Pipeline orchestrator | ✅ Active |
| `01_preprocessing.py` | Image preprocessing | ✅ Manual median filter |
| `02_edge_detection.py` | Edge detection (6 operators) | ✅ Manual kernels |
| `03_fft_filter.py` | Operator comparison | ✅ SNR/CPR metrics |
| `04_comparison.py` | Damage classification | ✅ 3-tier system |
| `06_visualization.py` | Final reports | ✅ Charts & analysis |

---

## 🚀 Future Enhancements | Mejoras Futuras

- [ ] Deep Learning integration (CNN for crack detection)
- [ ] Real-time video analysis
- [ ] 3D reconstruction from multiple images
- [ ] Mobile app deployment
- [ ] Cloud processing infrastructure
- [ ] Advanced ML model comparison

---

## 💡 Insights for Recruiters | Información para Reclutadores

### What Makes This Project Stand Out?

✨ **Complete Implementation from Scratch**
- No reliance on scikit-image or other ML frameworks
- Every algorithm implemented with NumPy and OpenCV
- Demonstrates deep understanding of computer vision fundamentals

✨ **Production-Ready Code**
- Clean architecture with modular components
- Error handling and logging
- Batch processing capabilities
- Comprehensive documentation

✨ **Practical Application**
- Real infrastructure inspection use case
- Scalable pipeline processing 85+ images
- Comparative analysis of 6 different algorithms
- Statistical validation and reporting

✨ **Mathematical Rigor**
- Custom convolution kernels
- FFT frequency domain analysis
- Signal-to-Noise Ratio calculations
- Connected component analysis

---

## 📞 Contact | Contacto

For questions about this project, please refer to the documentation above or contact the development team.

---

**Made with ❤️ for Infrastructure Analysis | Hecho con ❤️ para Análisis de Infraestructura**

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green)
![NumPy](https://img.shields.io/badge/NumPy-1.19+-orange)
![License](https://img.shields.io/badge/License-Academic-red)
