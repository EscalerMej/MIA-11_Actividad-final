import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

img = Image.open("GOU.png")

st.set_page_config(page_title="Análisis Interactivo para determinar la muestra maestra de acuerdo a los reultados obtenidos en la prueba de BCI", layout="wide", page_icon="img")
st.title("Análisis de los resultados de las muestras en Prueba de BCI")

# Sidebar de navegación
menu = st.sidebar.radio(
    "Navegación",
    [
        "Cargar Datos",
        "Análisis de Promedios",
        "Análisis Descriptivo",
        "Filtrar por Prueba",
        "Histogramas",
        "PCA 2D",
        "PCA 3D"
    ]
)

# Subida de archivos
uploaded_files = st.sidebar.file_uploader("Carga uno o varios archivos CSV", type=["csv"], accept_multiple_files=True)

# Variables globales
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}
if 'variables_a_graficar' not in st.session_state:
    st.session_state.variables_a_graficar = []

if uploaded_files:
    st.session_state.dataframes = {}
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            st.session_state.dataframes[file.name] = df
        except Exception as e:
            st.error(f"Error al leer {file.name}: {e}")

    if menu == "Cargar Datos":
        st.subheader("Archivos Cargados")
        for name, df in st.session_state.dataframes.items():
            st.write(f"📄 **{name}**")
            st.dataframe(df.head())

    if menu == "Análisis de Promedios":
        resumen = []
        st.subheader("Promedios por Archivo")
        for name, df in st.session_state.dataframes.items():
            try:
                mean_vals = df.select_dtypes(include=[np.number]).mean()
                resumen.append((name, mean_vals.mean()))
                st.markdown(f"**{name}**")
                st.dataframe(mean_vals)
            except:
                st.warning(f"⚠️ No se pudieron calcular promedios en {name}.")

        if resumen:
            mejor = min(resumen, key=lambda x: x[1])
            st.success(f"✅ Archivo con mejor desempeño promedio: **{mejor[0]}** (promedio: {mejor[1]:.2f})")

    if menu in ["Análisis Descriptivo", "Filtrar por Prueba", "Histogramas", "PCA 2D", "PCA 3D"]:
        archivo_seleccionado = st.selectbox("Selecciona un archivo para el análisis", list(st.session_state.dataframes.keys()))
        df = st.session_state.dataframes[archivo_seleccionado]

        target_column = st.selectbox("Selecciona la columna objetivo (etiqueta):", df.columns)
        try:
            X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
            y = df[target_column]
        except:
            st.error("Asegúrate de que las columnas (excepto la etiqueta) sean numéricas.")
            st.stop()

        mean_values = X.mean()

        if menu == "Análisis Descriptivo":
            st.subheader("Análisis Descriptivo de las Pruebas")
            st.write("### Estadísticas Generales:")
            st.dataframe(df.describe().transpose())

            st.write("### Correlación entre prueba y prueba:")
            corr_matrix = df.corr()
            st.dataframe(corr_matrix)

            st.write("### Gráfico de Correlación entre las pruebas:")
            fig, ax = plt.subplots(figsize=(10, 6))
            cax = ax.matshow(corr_matrix, cmap='coolwarm')
            fig.colorbar(cax)
            ax.set_xticks(np.arange(len(corr_matrix.columns)))
            ax.set_yticks(np.arange(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=90)
            ax.set_yticklabels(corr_matrix.columns)
            st.pyplot(fig)

        if menu == "Filtrar por Prueba":
            st.subheader("Filtrado por Prueba de Valor Promedio")
            opciones_rango = {
                "Malo (>20)": (20, np.inf),
                "Regular (10-20)": (10, 20),
                "Bueno (5-10)": (5, 10),
                "Muy bueno (2-5)": (2, 5),
                "Excelente (<2)": (0, 2)
            }

            categoria = st.select_slider(
                "Selecciona una Categoria a evaluar",
                options=list(opciones_rango.keys()),
                value="Bueno (5-10)"
            )

            rango = opciones_rango[categoria]
            variables_filtradas = mean_values[(mean_values > rango[0]) & (mean_values <= rango[1])]

            if not variables_filtradas.empty:
                st.success(f"Pruebas en categoría **{categoria}**:")
                st.dataframe(variables_filtradas.sort_values())
            else:
                st.warning(f"No se encontraron Pruebas en la categoría: **{categoria}**")

        if menu == "Histogramas":
            st.subheader("Histogramas Interactivos")

            opciones_hist = mean_values.index.tolist()
            seleccionadas = st.multiselect(
                "Selecciona la o las pruebas para mostrar su histograma:",
                options=opciones_hist,
                default=opciones_hist[:3]
            )

            st.session_state.variables_a_graficar = seleccionadas

            if seleccionadas:
                for var in seleccionadas:
                    fig, ax = plt.subplots()
                    ax.hist(X[var], bins=20, color='skyblue', edgecolor='black')
                    ax.set_title(f"Histograma de {var}")
                    ax.set_xlabel("Valor")
                    ax.set_ylabel("Frecuencia")
                    st.pyplot(fig)

        if menu == "PCA 2D" or menu == "PCA 3D":
            X_pca_data = X[st.session_state.variables_a_graficar] if st.session_state.variables_a_graficar else X

            X_train, X_test, y_train, y_test = train_test_split(X_pca_data, y, test_size=0.3, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            if menu == "PCA 2D":
                st.subheader("Visualización PCA en 2D")
                pca_2c = PCA(n_components=2)
                X_pca_2c = pca_2c.fit_transform(X_train_scaled)

                fig, ax = plt.subplots()
                scatter = ax.scatter(X_pca_2c[:, 0], X_pca_2c[:, 1], c=y_train, cmap='viridis')
                ax.set_xlabel('Componente Principal 1')
                ax.set_ylabel('Componente Principal 2')
                fig.colorbar(scatter, label='Relación de VSWR')
                st.pyplot(fig)

            if menu == "PCA 3D":
                st.subheader("Visualización PCA en 3D")
                pca_3c = PCA(n_components=3)
                X_pca_3c = pca_3c.fit_transform(X_train_scaled)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(X_pca_3c[:, 0], X_pca_3c[:, 1], X_pca_3c[:, 2], c=y_train, cmap='viridis')
                ax.set_xlabel('CP 1')
                ax.set_ylabel('CP 2')
                ax.set_zlabel('CP 3')
                fig.colorbar(scatter, label='Relación de VSWR')
                st.pyplot(fig)
else:
    st.info("Por favor, sube uno o varios archivos CSV para comenzar.")
