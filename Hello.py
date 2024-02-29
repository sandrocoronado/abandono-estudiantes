import streamlit as st

def main():
    st.title("Bienvenido a la Aplicación de Evaluación de Modelos de Machine Learning Para Predecir el Abandono de Estudiantes")

    st.header("Introducción")
    st.write("""
    Esta aplicación interactiva está diseñada para facilitar la exploración de datos, 
    la realización de un análisis exploratorio de datos (EDA), y la evaluación de diferentes 
    modelos de machine learning para clasificación. Los usuarios pueden cargar sus propios 
    conjuntos de datos, realizar limpieza de datos, visualizar estadísticas descriptivas, 
    y entrenar modelos para comparar su rendimiento.
    """)

    st.header("¿Qué puede hacer esta aplicación?")
    st.write("""
    - **Carga de Datos**: Suba su conjunto de datos en formato CSV para comenzar el análisis.
    - **Análisis Exploratorio de Datos (EDA)**: Obtenga visualizaciones automáticas para las 
      variables categóricas y numéricas, y explore la correlación entre las características.
    - **Preprocesamiento de Datos**: Aplique transformaciones estándar a sus datos para 
      prepararlos para el modelado.
    - **Entrenamiento de Modelos**: Seleccione entre varios algoritmos de clasificación, 
      como Regresión Logística, Random Forest, SVM y Gradient Boosting.
    - **Evaluación de Modelos**: Evalúe el rendimiento del modelo seleccionado utilizando 
      métricas como la precisión y el informe de clasificación.
    """)

    st.header("Instrucciones")
    st.write("""
    1. Comience cargando su archivo CSV utilizando la opción de carga de archivos.
    2. Limpie sus datos si es necesario, y seleccione 'Show EDA' para visualizar el EDA.
    3. Elija el modelo de clasificación que desee utilizar desde el menú desplegable.
    4. Revise los resultados del rendimiento del modelo en la sección de resultados.
    """)

    st.header("Requisitos del Conjunto de Datos")
    st.write("""
    Para que esta aplicación funcione correctamente, tu archivo CSV debe contener las siguientes columnas:
    
    - Estado Civil (`Marital status`)
    - Modo de Aplicación (`Application mode`)
    - Orden de Aplicación (`Application order`)
    - Curso (`Course`)
    - Asistencia Diurna/Nocturna (`Daytime/evening attendance`)
    - Cualificación Previa (`Previous qualification`)
    - Nacionalidad (`Nacionality`)
    - Cualificación de la Madre (`Mother's qualification`)
    - Cualificación del Padre (`Father's qualification`)
    - Ocupación de la Madre (`Mother's occupation`)
    - Ocupación del Padre (`Father's occupation`)
    - Desplazado (`Displaced`)
    - Necesidades Especiales Educativas (`Educational special needs`)
    - Deudor (`Debtor`)
    - Cuotas de Matrícula al Día (`Tuition fees up to date`)
    - Género (`Gender`)
    - Becario (`Scholarship holder`)
    - Edad al Inscribirse (`Age at enrollment`)
    - Internacional (`International`)
    - Unidades Curriculares 1er Semestre (Créditos) (`Curricular units 1st sem (credited)`)
    - Unidades Curriculares 1er Semestre (Inscrito) (`Curricular units 1st sem (enrolled)`)
    - Unidades Curriculares 1er Semestre (Evaluaciones) (`Curricular units 1st sem (evaluations)`)
    - Unidades Curriculares 1er Semestre (Aprobadas) (`Curricular units 1st sem (approved)`)
    - Unidades Curriculares 1er Semestre (Nota) (`Curricular units 1st sem (grade)`)
    - Unidades Curriculares 1er Semestre (Sin Evaluaciones) (`Curricular units 1st sem (without evaluations)`)
    - Unidades Curriculares 2do Semestre (Créditos) (`Curricular units 2nd sem (credited)`)
    - Unidades Curriculares 2do Semestre (Inscrito) (`Curricular units 2nd sem (enrolled)`)
    - Unidades Curriculares 2do Semestre (Evaluaciones) (`Curricular units 2nd sem (evaluations)`)
    - Unidades Curriculares 2do Semestre (Aprobadas) (`Curricular units 2nd sem (approved)`)
    - Unidades Curriculares 2do Semestre (Nota) (`Curricular units 2nd sem (grade)`)
    - Unidades Curriculares 2do Semestre (Sin Evaluaciones) (`Curricular units 2nd sem (without evaluations)`)
    - Tasa de Desempleo (`Unemployment rate`)
    - Tasa de Inflación (`Inflation rate`)
    - PIB (`GDP`)
    - Variable Objetivo (`Target`)
    
    Asegúrate de que tu conjunto de datos contenga estas columnas antes de cargarlo para garantizar que la aplicación funcione como se espera.
    """)

if __name__ == "__main__":
    main()

