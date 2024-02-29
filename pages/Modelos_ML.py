import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

def load_data(filepath):
    return pd.read_csv(filepath)

def treat_data(data):
    data = data.drop_duplicates()
    data = data.dropna()
    return data

def perform_eda(data):
    # Para columnas categóricas
    cat_columns = data.select_dtypes(include=['object']).columns
    for col in cat_columns:
        fig, ax = plt.subplots()
        data[col].value_counts().plot(kind='bar', title=col, ax=ax)
        st.pyplot(fig)  # Se muestra cada gráfico de barras en Streamlit

    # Para datos numéricos
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    fig, ax = plt.subplots(figsize=(10, 10))
    numeric_data.hist(ax=ax)
    st.pyplot(fig)  # Se usa st.pyplot para mostrar la figura en Streamlit
    
    # Graficar matriz de correlación
    corr_matrix = numeric_data.corr()
    sns.set_theme(style="white")  # Actualizado a set_theme
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
    
    # Obtener las 5 variables más correlacionadas
    corr_values = corr_matrix.unstack()
    sorted_corr_values = corr_values.sort_values(kind="quicksort", ascending=False)
    top_corr_pairs = sorted_corr_values[sorted_corr_values != 1].head(5)
    top_corr_variables = list(set([pair[0] for pair in top_corr_pairs.index] + [pair[1] for pair in top_corr_pairs.index]))

    # Graficar las 5 variables más correlacionadas
    if len(top_corr_variables) > 1:
        top_corr_data = data[top_corr_variables]
        pairplot = sns.pairplot(data=top_corr_data, kind="scatter", diag_kind="kde", markers="o", plot_kws=dict(s=50, edgecolor="b", linewidth=0.5), diag_kws=dict(shade=True))
        st.pyplot(pairplot.fig)


def preprocess_data(data):
    num_features = data.select_dtypes(include=['int64', 'float64']).columns
    cat_features = data.select_dtypes(include=['object']).columns.drop('Target')
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)])
    
    return preprocessor

def divide_data(data, target_name):
    X = data.drop(target_name, axis=1)
    y = data[target_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def model_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"{name}:")
        print(f"Accuracy on test set: {accuracy}")
        print(report)
        print("\n" + "-"*80 + "\n")

def main():
    st.title("Machine Learning Model Evaluation")

    # Carga de datos
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())

        # Tratamiento de datos y EDA
        data_clean = treat_data(data)
        if st.checkbox('Show EDA'):
            perform_eda(data_clean)
        
        # Preprocesamiento y división del conjunto de datos
        preprocessor = preprocess_data(data_clean)
        X_train, X_test, y_train, y_test = divide_data(data_clean, 'Target')
        
        # Selección de modelos
        option = st.selectbox('Which model would you like to use?',
                              ('Logistic Regression', 'Random Forest', 'SVM', 'Gradient Boosting'))
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        # Modelado y evaluación
        model = models[option]
        model_name, model_report = model_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, model)
        st.write(f"Results for {model_name}:")
        st.text(model_report)

# Esta función se ha ajustado para trabajar dentro de Streamlit
def model_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, model):
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return model.__class__.__name__, "Accuracy on test set: " + str(accuracy) + "\n" + report

if __name__ == "__main__":
    main()
