# Model Card - VERIDICT IQ Baseline B1

## Estado
Validación técnica preliminar con datos sintéticos. No apto para decisiones reales.

## Uso previsto
Verificar la ejecución reproducible del pipeline de clasificación binaria y preparar la sustitución por el snapshot anonimizado autorizado.

## Modelo
Regresión logística L2, C=1.0, max_iter=1000, preprocesamiento encapsulado.

## Datos
360 registros sintéticos; desarrollo=288 y test temporal=72. SHA-256: `c1a96f1adf37f9b1e85f1c8fd819e6983f7add4b5ed09042eaa1f4a018cf917c`.

## Métricas de demostración
ROC-AUC test: 0.747; PR-AUC: 0.826; macro-F1: 0.625; Brier: 0.207.

## Limitaciones
- Los datos no provienen de GRUND Abogados LLP®.
- Las relaciones fueron simuladas y no sustentan conclusiones jurídicas u organizacionales.
- La calibración y los intervalos son inestables por el tamaño del test.
- Requiere validación jurídica de etiquetas, variables t0 y sesgos antes de cualquier piloto real.

## Usos no permitidos
No usar para aceptar, rechazar, valorar, priorizar o aconsejar litigios reales.
