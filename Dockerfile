# Dockerfile para NeuroNet-Fusion CDSS (Producción)
# Apunta exclusivamente a Analytical_Biomarker_Project/
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar solo el código fuente y los modelos necesarios
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Exponer el puerto de Streamlit
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Lanzar la aplicación
ENTRYPOINT ["streamlit", "run", "src/app_diagnostics.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
