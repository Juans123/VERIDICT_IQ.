# Matriz de cumplimiento del informe de resultados preliminares

| Requisito de la actividad | Evidencia principal |
|---|---|
| Descripción del prototipo y baseline | Secciones 2 y 3 del informe; `notebooks/VERIDICT_IQ_Resultados_Preliminares_Colab.ipynb` |
| Conjunto de datos y contexto de evaluación | Sección 3.2; `data/dataset_sintetico_veridict_iq.csv`; advertencia de contingencia sintética |
| Métricas de entrenamiento y validación | Sección 4; `reports/metrics/resultados_ajuste_desarrollo.json` y `resultados_cv.csv` |
| Resultados de prueba y evidencias visuales | Sección 4; `reports/metrics/resultados_test.csv`; `reports/figures/` |
| Análisis de errores y comportamientos inesperados | Sección 5; `reports/metrics/analisis_errores.csv` |
| Fortalezas, limitaciones y relación organizacional | Secciones 5.3–5.5 del informe |
| Ajustes técnicos y mejoras metodológicas | Sección 6.1–6.2 |
| Backlog actualizado y priorizado | Sección 6.3; `docs/backlog_actualizado.csv` |
| Repositorio, README y bitácora | Sección 7; `README.md`; `docs/bitacora_cambios.csv` |
| Reproducibilidad y trazabilidad | `config/baseline.yaml`, `requirements.txt`, hashes y metadatos en `reports/metrics/run_metadata.json` |

## Regla de uso
La ejecución incluida es una prueba de ingeniería con datos sintéticos. Para convertirla en evidencia
organizacional, debe ejecutarse nuevamente con el snapshot anonimizado y autorizado, conservar el test
temporal sellado y sustituir en el informe todas las tablas y figuras de demostración por los artefactos reales.
