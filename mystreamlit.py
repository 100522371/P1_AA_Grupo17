import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Título y descripción
st.title("Predicción de Contratación de Depósito (Bank Marketing)")
st.write("""
Esta aplicación permite predecir si un cliente suscribirá un depósito a plazo fijo, 
utilizando un modelo de Regresión Logística con regularización L1 previamente entrenado.
""")

# 2. Cargar el modelo entrenado
@st.cache_resource
def load_model():
    return joblib.load('modelo_final.joblib')

try:
    modelo = load_model()
    st.success("Modelo cargado correctamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# 3. Crear el formulario de entrada de datos en la barra lateral
st.sidebar.header("Datos del Cliente")

def user_input_features():
    # Variables numéricas
    age = st.sidebar.slider("Edad (age)", 18, 100, 35)
    balance = st.sidebar.number_input("Saldo medio anual (balance) en €", value=1500)
    day = st.sidebar.slider("Día del mes (day)", 1, 31, 15)
    duration = st.sidebar.slider("Duración del contacto (duration) en segundos", 0, 4000, 200)
    campaign = st.sidebar.slider("Contactos en esta campaña (campaign)", 1, 50, 2)
    pdays_raw = st.sidebar.number_input("Días desde el contacto anterior (pdays, -1 si no hubo)", value=-1)
    previous = st.sidebar.slider("Contactos previos a esta campaña (previous)", 0, 50, 0)
    
    # Diccionarios de traducción
    trad_sino = {'no': 'No', 'yes': 'Sí'}
    trad_job = {'admin.': 'Administrativo', 'blue-collar': 'Obrero', 'entrepreneur': 'Emprendedor', 'housemaid': 'Empleada del hogar', 'management': 'Gerente', 'retired': 'Jubilado', 'self-employed': 'Autónomo', 'services': 'Servicios', 'student': 'Estudiante', 'technician': 'Técnico', 'unemployed': 'Desempleado', 'unknown': 'Desconocido'}
    trad_marital = {'divorced': 'Divorciado', 'married': 'Casado', 'single': 'Soltero', 'unknown': 'Desconocido'}
    trad_edu = {'primary': 'Primaria', 'secondary': 'Secundaria', 'tertiary': 'Universitaria', 'unknown': 'Desconocido'}
    trad_contact = {'cellular': 'Móvil', 'telephone': 'Teléfono fijo', 'unknown': 'Desconocido'}
    trad_month = {'jan': 'Enero', 'feb': 'Febrero', 'mar': 'Marzo', 'apr': 'Abril', 'may': 'Mayo', 'jun': 'Junio', 'jul': 'Julio', 'aug': 'Agosto', 'sep': 'Septiembre', 'oct': 'Octubre', 'nov': 'Noviembre', 'dec': 'Diciembre'}
    trad_poutcome = {'failure': 'Fracaso', 'other': 'Otro', 'success': 'Éxito', 'unknown': 'Desconocido'}

    # Variables categóricas
    job = st.sidebar.selectbox("Trabajo (job)", list(trad_job.keys()), format_func=lambda x: trad_job[x])
    marital = st.sidebar.selectbox("Estado civil (marital)", list(trad_marital.keys()), format_func=lambda x: trad_marital[x])
    education = st.sidebar.selectbox("Educación (education)", list(trad_edu.keys()), format_func=lambda x: trad_edu[x])
    default = st.sidebar.selectbox("¿Tiene crédito en mora? (default)", list(trad_sino.keys()), format_func=lambda x: trad_sino[x])
    housing = st.sidebar.selectbox("¿Tiene préstamo hipotecario? (housing)", list(trad_sino.keys()), format_func=lambda x: trad_sino[x])
    loan = st.sidebar.selectbox("¿Tiene préstamo personal? (loan)", list(trad_sino.keys()), format_func=lambda x: trad_sino[x])
    contact = st.sidebar.selectbox("Tipo de contacto (contact)", list(trad_contact.keys()), format_func=lambda x: trad_contact[x])
    month = st.sidebar.selectbox("Mes del último contacto (month)", list(trad_month.keys()), format_func=lambda x: trad_month[x])
    poutcome = st.sidebar.selectbox("Resultado campaña anterior (poutcome)", list(trad_poutcome.keys()), format_func=lambda x: trad_poutcome[x])

    # Preprocesado manual de pdays tal y como se hizo en el EDA para entrenar
    if pdays_raw == -1:
        pdays = 'no_contact'
    elif pdays_raw <= 200:
        pdays = 'recent'
    elif pdays_raw <= 400:
        pdays = 'intermediate'
    else:
        pdays = 'old'

    # Guardar en diccionario
    data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays, # el categorizado
        'previous': previous,
        'poutcome': poutcome
    }
    
    return pd.DataFrame(data, index=[0])

# Obtenemos el DataFrame con los inputs del usuario
input_df = user_input_features()

# 4. Mostrar los datos ingresados en la parte principal
st.subheader("Datos Ingresados")
st.write("A continuación se muestra el perfil del cliente en base a lo configurado (nota: 'pdays' ha sido categorizado internamente para el modelo):")
st.dataframe(input_df)

# 5. Predicción
if st.button("Predecir Suscripción de Depósito"):
    # Como el pipeline tiene OneHotEncoder y StandardScaler, acepta el DataFrame crudo
    prediccion = modelo.predict(input_df)[0]
    probabilidades = modelo.predict_proba(input_df)[0]
    
    st.subheader("Resultado de la Predicción")
    
    if prediccion == 'yes':
        st.success(f"**El modelo predice: SÍ contratará el depósito.** (Probabilidad: {probabilidades[1]*100:.2f}%)")
    else:
        st.warning(f"**El modelo predice: NO contratará el depósito.** (Probabilidad: {probabilidades[0]*100:.2f}%)")

