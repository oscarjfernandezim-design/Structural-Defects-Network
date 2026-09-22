# Structural Defects Network

Pipeline reproducible de visión computacional para explorar detección de
grietas en imágenes de infraestructura. El repositorio contiene un baseline
clásico; **no es un sistema certificado de inspección estructural**.

## Qué hace

1. Convierte las imágenes a escala de grises, las normaliza a `256x256` y
   aplica un filtro de mediana.
2. Genera máscaras candidatas con Canny, Laplaciano, Sobel, Prewitt, Roberts
   y un filtro de altas frecuencias FFT.
3. Calcula CPR (porcentaje de píxeles activos), componentes conectados y
   severidad descriptiva.
4. Si existen etiquetas en `data/annotated/Cracked` y
   `data/annotated/No-Cracked`, calibra una regla CPR en train y reporta
   clasificación en un conjunto test separado.

CPR es una señal de textura/bordes, no una medición física de una grieta.
Para segmentación real se necesitan máscaras anotadas píxel a píxel.

## Instalación

Requiere Python 3.10 o superior.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Estructura

```text
.
├── ejecutar.py
├── requirements.txt
├── src/
│   ├── 01_preprocessing.py
│   ├── 02_edge_detection.py
│   ├── 03_fft_filter.py
│   ├── 04_comparison.py
│   ├── 06_visualization.py
│   └── 07_evaluate.py
├── data/
│   ├── raw/                  # imágenes de entrada
│   ├── processed/            # generado
│   └── annotated/
│       ├── Cracked/          # etiqueta positiva
│       └── No-Cracked/       # etiqueta negativa
└── results/                  # máscaras, gráficos y evaluación generados
```

Los datos y resultados generados están excluidos de Git por `.gitignore`.

## Uso

Coloca imágenes en `data/raw/` y ejecuta:

```powershell
python ejecutar.py
```

El pipeline falla con código distinto de cero si una etapa no puede producir
salida. El orquestador resuelve la raíz del proyecto y se puede invocar desde
cualquier carpeta:

```powershell
python C:\ruta\Structural-Defects-Network\ejecutar.py
```

Para evaluar etiquetas por carpeta:

```powershell
python src\07_evaluate.py --folder-labels
```

Se generan:

- `results/evaluation/classification_by_image.csv`
- `results/evaluation/classification_summary.csv`
- `results/evaluation/classification_report.md`

La evaluación usa solamente imágenes presentes simultáneamente en las
carpetas anotadas y en `results/masks`. Las imágenes sin predicción se
reportan explícitamente como `MISSING`.

## Interpretación de la evaluación

La calibración es determinista: por cada clase, el 70% de los nombres
ordenados se usa para elegir dirección (`CPR >= umbral` o `CPR <= umbral`) y
umbral maximizando F1; el 30% restante se usa como test. Esto evita presentar
como precisión objetiva un umbral fijo escogido mirando todo el dataset.

La evaluación de carpetas mide clasificación de imagen. `IoU` y `Dice` solo
son válidos cuando se proporcionan máscaras humanas en
`data/annotated/images` y `data/annotated/masks`.

## Calidad y limitaciones

- Los operadores clásicos detectan bordes, textura y ruido; no entienden la
  semántica de una grieta.
- El resize fijo puede perder escala física y grietas muy finas.
- El conjunto etiquetado actual no proporciona máscaras píxel a píxel.
- Las métricas con un dataset pequeño tienen alta incertidumbre.
- No debe utilizarse el resultado para decisiones de seguridad sin inspección
  experta y validación independiente.

El siguiente paso científico es comparar este baseline con un clasificador
supervisado y separar formalmente train, validación y test.

## Tests

Las pruebas no requieren dependencias adicionales:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
