# Use a stable Python version with pre‑built wheels for OpenCV and Ultralytics
FROM python:3.10-slim

WORKDIR /app

# Install all system libraries required by OpenCV (headless and GTK support)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    libgtk2.0-0 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

EXPOSE 8501

# Start the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
