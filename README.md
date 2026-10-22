# Examen Progreso 1 - Aprendizaje Automático (IIAR0004)
**Universidad de Las Américas (UDLA) - Facultad de Ingeniería y Ciencias Aplicadas**  
**Docente:** Raúl Paredes, M. Sc.  
**Estudiante:** Carlos Alberto Angulo Pizarro (`carlos.angulo.pizarro@udla.edu.ec`)  
**Grupo:** Group 4 (Actividad Grupal / Entrega Individual)  
**Semestre:** 2026 – 2027 (2027-10)

---

## 🚀 Acceso Directo e Interactivo en Google Colab

Haga clic en la insignia inferior para abrir y ejecutar interactivamente el cuaderno del examen en Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Elnuget/examen-progreso-1-aprendizaje-automatico/blob/main/examen_Aprendizaje_Automatico_Progreso_1_.ipynb)

---

## 📌 Resumen Ejecutivo del Examen de Aplicación

El objetivo de esta evaluación práctica consiste en diagnosticar, sanear y preparar un conjunto de datos empresariales (`clientes_financieros_examen.csv`, 6.035 registros) para la predicción de **abandono de clientes (*customer churn*)** en una institución financiera, justificando rigurosamente cada decisión técnica y previniendo la fuga de información (*data leakage*).

### Estructura de la Solución
1. **Sección A: Fundamentos Conceptuales:**
   - Formulación del problema de retención y definición formal del vector objetivo $y = \text{abandono} \in \{0, 1\}$.
   - Caracterización como aprendizaje supervisado de clasificación binaria.
   - Posicionamiento de la ingeniería de datos en el ciclo de vida de ML y justificación teórica contra la evaluación en datos de entrenamiento.
2. **Sección B: Auditoría Exploratoria de Datos (EDA):**
   - Detección de tipos erróneos (`edad`, `ingresos_anuales`, `saldo_promedio` leídos como `object` por cadenas `"desconocido"`).
   - Identificación de 35 registros duplicados exactos por `id_cliente`.
   - Diagnóstico del desbalance de clases (86,4% permanencia vs 13,6% abandono, ratio 6,37:1).
   - Identificación de fuga de información crítica en `motivo_baja` y valores centinela `-999999` en saldos.
3. **Sección C: Limpieza y Saneamiento de Datos (`df_clean`):**
   - Coerción numérica con `pd.to_numeric(errors='coerce')`.
   - Homogeneización de cadenas textuales (`.str.strip().str.capitalize()`) y mapeo de marcadores `"?"` y `"nan"` a `NaN`.
   - Eliminación justificada de los 35 duplicados (6.000 registros únicos consolidados).
   - Aplicación de reglas de dominio: sustitución de edades $> 100$ por `NaN`, conversión de centinelas `-999999` a `NaN`, y conservación deliberada de sobregiros financieros legítimos ($saldo < 0$).
4. **Sección D: Preparación de Características y Pipelines Scikit-Learn:**
   - Exclusión de `id_cliente` (identificador) y `motivo_baja` (fuga de información).
   - Partición estratificada 80/20 (`random_state=42`, $4.800$ train / $1.200$ test).
   - `ColumnTransformer` hermético:
     * Pipeline Numérico: `SimpleImputer(strategy='median')` + `StandardScaler()`.
     * Pipeline Categórico: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
   - Ajuste estricto sobre `X_train` (`fit_transform`) y transformación de `X_test` (`transform`), generando matrices continuas libres de `NaN` e infinitos.
5. **Sección E: Diseño Experimental, Ética y Conclusiones:**
   - Análisis de sesgo por truncamiento de valores atípicos.
   - Explicación del mecanismo causal de fuga en variables *post-hoc*.
   - Evaluación de riesgos éticos (*proxy discrimination / redlining* y privacidad).
   - Propuesta de controles de equidad (*Equalized Odds*, *Disparate Impact*) y monitoreo de deriva.
