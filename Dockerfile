FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

# Expose the port Streamlit runs on
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
