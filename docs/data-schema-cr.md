# Feature schema v2 — Credit Intelligence (Costa Rica)

Este documento define el esquema canónico de datos usado para entrenar y servir
los modelos de **elegibilidad** ("sujeto a crédito") y **riesgo de default**.

## 1. Por qué no usamos datos reales del CIC

El Centro de Información Crediticia (CIC) de SUGEF consolida el historial
crediticio real de deudores costarricenses, pero el acceso está restringido a
"funcionarios de entidades supervisadas" con firma digital
(Reglamento del CIC, Art. 3-6, `pgrweb.go.cr/scij` nValor1=1&nValor2=57386).
finu-saas no es una entidad supervisada, así que **no tenemos ni podemos
obtener acceso a registros reales del CIC**.

En su lugar:
- Replicamos la **estructura/esquema de variables** que el CIC define para sus
  3 reportes (dominio público, entidad autorizada, deudor).
- Calibramos las **distribuciones agregadas** del generador sintético con
  estadísticas públicas (INEC, BCCR, SUGEF) — nunca con registros individuales
  reales.
- Todo dataset generado se marca con `data_source: "synthetic_v1"` y debe
  citarse como tal ante cualquier auditoría SUGEF; no debe presentarse como
  dato real de buró.

## 2. Variables del esquema CIC replicadas (estructura, no datos reales)

Fuente: Reglamento del CIC (SUGEF), Anexo de variables por tipo de reporte.

**Reporte de dominio público** (las usamos todas):
- `moneda` (CRC / USD)
- `tipo_operacion` (consumo, vivienda, comercial, tarjeta, etc.)
- `condicion_deudor` (al día, moroso, cobro judicial, castigado)
- `historial_pago_12m` (string de 12 posiciones, días de atraso por mes)
- `dias_atraso` (`worst_delay_days` en nuestro esquema)

**Reporte para entidad con autorización** (subset relevante a scoring):
- `entidad_acreedora`, `identificador_operacion` — no usados como features (PII/leakage), solo para trazabilidad de auditoría si hubiera datos reales.
- `saldo_operacion` por moneda → `active_debts`, `credit_utilization_pct`
- `monto_no_desembolsado`
- `cuota_principal`, `cuota_intereses` → `payment_to_income`
- `tasa_interes_nominal`, `tipo_tasa` (fija/ajustable), `parametro_referencia` (Libor/TBP/TPM)
- `fecha_vencimiento`
- `dias_atraso` por operación

**Reporte para el deudor**: agrega metadatos de auditoría (quién consultó el
archivo) — no es relevante para features de scoring, se omite.

## 3. Variables CCSS (empleo formal)

- `patrono` (empleador) — no se usa como feature directa (alta cardinalidad,
  riesgo de leakage), pero sí en reglas de elegibilidad (lista de patronos
  inactivos/sancionados).
- `salario_reportado_ccss` → contrastado contra `monthly_income` declarado
  (regla `employment_ccss` ya existente en `apps/fintech-saas/lib/credit/rules/employment.ts`).
- `antiguedad_laboral_meses` → `employment_months`.
- `condicion_patronal` (activo/inactivo) → gate duro de elegibilidad.

## 4. Esquema canónico: `PERSONAL_CREDIT_V1` (pipeline/schemas.py)

El esquema de features en producción vive en
[pipeline/schemas.py](../pipeline/schemas.py) (`PERSONAL_CREDIT_V1` /
`CORPORATE_CREDIT_V1`) y es consumido por el champion model registry
([models/registry.py](../models/registry.py)) y por
[pipeline/features.py](../pipeline/features.py) (`compute_features`). El
modelo de elegibilidad ([models/eligibility.py](../models/eligibility.py))
y el generador sintético ([synthetic/generator.py](../synthetic/generator.py))
reutilizan ese MISMO esquema (`PERSONAL_CREDIT_V1.features`) para que
train/serve nunca diverjan.

`PERSONAL_CREDIT_V1.features` (14 continuas, con signo de monotonicidad
SUGEF 1-05 ya declarado en el propio `pipeline/schemas.py`):

`age`, `monthly_income`, `employment_months`, `employment_type_encoded`,
`equifax_score`, `active_debts`, `worst_delay_days`,
`credit_utilization_pct`, `avg_balance_3m`, `transaction_count_3m`,
`dti_ratio`, `payment_to_income`, `debt_service_coverage`, `loan_to_income`.

Mapeo a variables del CIC/CCSS descritas en las secciones 2-3 de este
documento: `equifax_score` ≈ score de buró (CIC, dominio público/entidad
autorizada), `active_debts` ≈ conteo de operaciones crediticias activas,
`worst_delay_days` ≈ `dias_atraso` del CIC, `employment_months` ≈
antigüedad CCSS.

Nota histórica: una versión anterior de este documento proponía un
`feature_schema_v2` propio (con `income_ccss_ratio`, `credit_history_months`,
variables corporativas, etc.) en paralelo al esquema productivo. Se descartó
para evitar dos fuentes de verdad — cualquier feature nueva (p. ej. variables
corporativas o categóricas adicionales del CIC) debe agregarse directamente
a `pipeline/schemas.py`.

### Labels (no parte de `PERSONAL_CREDIT_V1`, específicas del dataset sintético)

- `elegible` (bool) — gate de elegibilidad, consumido por `POST /eligibility`.
- `default_12m` (0/1) — probabilidad de incumplimiento a 12 meses, usado para
  entrenar el champion model que sirve `POST /score`.

## 5. Calibración del generador sintético con estadísticas públicas (2025-2026)

| Parámetro | Valor usado | Fuente |
|---|---|---|
| Ingreso promedio hogar quintil 1 | ¢275,771/mes | INEC ENAHO 2025 |
| Ingreso promedio hogar quintil 2 | ¢560,500/mes | INEC ENAHO 2025 |
| Ingreso promedio hogar quintil 3 | ¢925,093/mes | INEC ENAHO 2025 |
| Ingreso promedio hogar quintil 4 | ¢1,391,137/mes | INEC ENAHO 2025 |
| Ingreso promedio hogar quintil 5 | ¢2,897,190/mes | INEC ENAHO 2025 |
| Ingreso per cápita nacional | ¢485,792/mes | INEC ENAHO 2025 |
| Mora >90 días + cobro judicial / cartera total | ~2.07% | BCCR Informe Estabilidad Financiera 2025 |
| Mora amplia del sistema (SIFR) | ~11% | BCCR / SUGEF 2025 |
| Mora consumo/tarjetas (deterioro reciente) | 3.0%-3.3% | BCCR IAEF 2025 |
| Crecimiento cartera crediticia | 4.87% anual | SUGEF, presentación Balance Sistema Financiero 2025 |

Estas cifras son **agregados a nivel de sistema/país**, no registros
individuales — se usan únicamente para calibrar las distribuciones marginales
del generador (p. ej. asignar el ingreso simulado a un quintil con la
probabilidad correcta, y fijar la tasa base de default sintético cerca del
2-3% observado en consumo, antes de aplicar el modelo causal por segmento).

## 6. Limitaciones y advertencia de auditoría

- El generador NO modela correlaciones individuales reales (p. ej. la relación
  exacta entre patrono específico y riesgo) — solo relaciones causales
  razonables definidas a mano (DTI alto → más riesgo, etc.) más el ruido
  estructural típico de un generador SCM.
- Cualquier modelo entrenado solo con `synthetic_v1` debe tratarse como
  **prototipo de validación de arquitectura**, no como modelo de producción
  apto para decisiones reales hasta que se reentrene con datos reales de
  producción (vía `credit_ml_feature_snapshots` y `ml_feedback`) una vez la
  plataforma tenga volumen real de aplicaciones con desenlace conocido.
