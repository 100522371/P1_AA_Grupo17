# Práctica 1: Predicción de subscripción a un producto bancario
## Archivos
- `modelo_final.joblib`: El modelo final exportado y listo para ser consumido.
- `mystreamlit.py`: Aplicación web interactiva programada en Streamlit.
- `p1_aa.ipynb`: Notebook donde se describe todo el proceso de trabajo y se entrena el modelo final
- `predicciones.csv`: Archivo que contiene las predicciones de los datos de `bank_competition.pkl`.
- `predicciones.ipynb`: Notebook que carga el modelo final y realiza predicciones sobre `bank_competition.pkl`
- `capturas-streamlit`: Carpeta con dos imagenes que prueban que streamlit nos funciona bien

---

## Instalación y Uso
1. **Instalar dependencias**:
   Abre una terminal en esta misma carpeta y ejecuta:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar la aplicación**:
   Levanta la interfaz web de predicción en tiempo real simplemente escribiendo:
   ```bash
   streamlit run mystreamlit.py
   ```
