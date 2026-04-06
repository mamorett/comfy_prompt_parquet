# Use a slim Python image for efficiency
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and .streamlit config
COPY streamlit_viewer.py .
COPY .streamlit/ ./.streamlit/

# Expose the default Streamlit port
EXPOSE 8501

# Add a healthcheck to ensure the container is healthy
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "streamlit_viewer.py", "--server.port=8501", "--server.address=0.0.0.0"]
