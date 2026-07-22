# VERIDICT IQ

**Sistema predictivo de resultados judiciales para litigios civiles y laborales**  
Proyecto de titulación — Maestría en Inteligencia Artificial Aplicada.

## Estado del repositorio

Esta versión contiene una ejecución reproducible de los controles **B0** y del baseline **B1 (regresión logística regularizada)**.

> [!IMPORTANT]
> Las métricas publicadas en esta versión fueron obtenidas con **360 registros sintéticos de contingencia**. El conjunto reproduce el esquema técnico previsto, pero no contiene causas, clientes ni decisiones judiciales reales. Los resultados validan el pipeline; **no permiten concluir el desempeño real de VERIDICT IQ en GRUND Abogados LLP®**.

## Contenido principal

- `data/synthetic/`: dataset sintético utilizado en la demostración.
- `notebooks/`: notebook ejecutable en Google Colab.
- `src/veridict_iq/`: código del pipeline de datos, entrenamiento y evaluación.
- `config/`: configuración metodológica del baseline.
- `reports/`: métricas, figuras y model card de la ejecución documentada.
- `models/baseline/`: baseline B1 serializado.
- `docs/`: informe S5, backlog, bitácora y matriz de cumplimiento.
- `tests/`: pruebas mínimas del contrato de datos y reproducibilidad.

## Modelos evaluados

| Identificador | Modelo | Función |
|---|---|---|
| B0 most frequent | `DummyClassifier(strategy="most_frequent")` | Piso de no habilidad basado en la clase mayoritaria. |
| B0 stratified | `DummyClassifier(strategy="stratified")` | Piso aleatorio según la distribución histórica. |
| B1 | Regresión logística L2 | Baseline técnico probabilístico e interpretable. |

## Resultados de demostración en test temporal

| Modelo | ROC-AUC | PR-AUC | F1 macro | Balanced accuracy | Brier |
|---|---:|---:|---:|---:|---:|
| B0 most frequent | 0.500 | 0.556 | 0.357 | 0.500 | 0.444 |
| B0 stratified | 0.463 | 0.538 | 0.462 | 0.463 | 0.528 |
| B1 regresión logística | **0.747** | **0.826** | **0.625** | **0.625** | **0.207** |

El umbral de B1 (`0.485`) fue seleccionado con predicciones *out-of-fold* del conjunto de desarrollo. El test temporal contiene 72 registros y fue reservado para una evaluación final única.

## Ejecución rápida en Google Colab

1. Abra `notebooks/VERIDICT_IQ_Resultados_Preliminares_Colab.ipynb`.
2. Mantenga `MODO_DATOS = "sintetico"` para reproducir la demostración.
3. Ejecute todas las celdas en orden.
4. Descargue el ZIP de artefactos generado al final.

Para cargar posteriormente un snapshot real anonimizado, cambie a `MODO_DATOS = "subir"`. Antes de hacerlo deben existir autorización institucional, protocolo de anonimización y validación jurídica del *ground truth*.

## Ejecución local

### 1. Crear el entorno

```bash
python -m venv .venv
```

Activación en Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activación en Linux o macOS:

```bash
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Ejecutar el pipeline

```bash
python run_demo.py
```

Los artefactos nuevos se guardan en `artifacts/latest/`, carpeta excluida de Git para no modificar las evidencias congeladas del informe.

### 4. Ejecutar pruebas

```bash
pytest -q
```

## Contrato mínimo del dataset

| Columna | Tipo esperado | Descripción |
|---|---|---|
| `case_id` | texto | Identificador seudonimizado o sintético. |
| `fecha_cierre` | fecha | Fecha usada para el corte temporal. |
| `materia` | categórica | Civil o laboral. |
| `tipo_proceso` | categórica | Tipo de procedimiento. |
| `cuantia_inicial_usd` | numérica | Cuantía conocida en el instante de predicción. |
| `rol_cliente` | categórica | Actor o demandado. |
| `instancia_en_t0` | categórica | Instancia conocida en `t0`. |
| `jurisdiccion` | categórica | Unidad territorial agregada. |
| `tipo_contraparte` | categórica | Persona natural, jurídica o entidad pública. |
| `medida_cautelar_solicitada` | binaria | Información disponible en `t0`. |
| `prueba_pericial_prevista` | binaria | Planificación inicial, no resultado posterior. |
| `resultado_favorable` | binaria | Variable objetivo: 1 favorable, 0 no favorable. |

## Reglas de confidencialidad

No deben subirse a GitHub nombres, cédulas, teléfonos, correos, direcciones, números reales de causa, expedientes, sentencias no autorizadas, tablas de correspondencia ni credenciales. Las carpetas destinadas a datos reales están excluidas en `.gitignore`.

## Reproducibilidad y trazabilidad

- Semilla global: `42`.
- Separación principal: 80 % de causas más antiguas para desarrollo y 20 % más recientes para test.
- Validación: `RepeatedStratifiedKFold`, 5 folds × 5 repeticiones.
- Preprocesamiento encapsulado en `Pipeline` y `ColumnTransformer`.
- Dataset sintético y modelo acompañados por hashes SHA-256.
- Métricas disponibles en CSV y JSON.
- Resultados documentados en model card, bitácora y backlog.

## Uso permitido

Este repositorio se destina a evaluación académica, auditoría metodológica y desarrollo técnico. El modelo incluido no debe utilizarse para aceptar, rechazar, valorar, priorizar ni recomendar estrategias sobre litigios reales.

## Equipo

- Alejandro Sarzosa Larrea
- Juan Francisco Soria Maldonado
- Solange Estefanía Swoboda Gamboa

## Documentación

- Informe S5: `docs/informe/S5_Informe_Resultados_Preliminares_VERIDICT_IQ_DEMO.pdf`
- Guía para publicar el repositorio: `docs/GUIA_SUBIDA_GITHUB.md`
- Model card: `reports/model_card/MODEL_CARD.md`
