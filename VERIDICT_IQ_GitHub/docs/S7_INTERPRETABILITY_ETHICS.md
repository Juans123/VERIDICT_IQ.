# Semana 7 — Interpretabilidad y ética

Esta extensión conserva intactos los artefactos de S6 e incorpora análisis reproducibles para el modelo B2 ajustado.

## Alcance

- SHAP global y local sobre el test temporal sellado.
- Sustituto lineal local inspirado en LIME, implementado sin dependencia externa.
- Estabilidad ante perturbación gaussiana de la cuantía.
- Comparación entre `class_weight="balanced"` y SMOTE aplicado dentro del pipeline.
- Bootstrap estratificado con 2.000 réplicas.
- Métricas de equidad por rol del cliente y equalized odds.
- Prueba exploratoria 5x2cv de Dietterich sobre balanced accuracy.

## Ejecución

```bash
python -m src.explainability.run_s7_analysis
```

Los resultados se almacenan en:

- `reports/figures/s7/`
- `reports/metrics/s7/`

## Advertencia

Todos los resultados corresponden a datos sintéticos. Son evidencia de validación de ingeniería y no deben interpretarse como reglas jurídicas, relaciones causales ni desempeño operativo real.
