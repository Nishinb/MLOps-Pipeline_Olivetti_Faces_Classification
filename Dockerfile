FROM python:3.10-slim

WORKDIR /app

# Install explicit system performance dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy explicitly required model artifacts and runtimes
COPY savedmodel.pth /app/savedmodel.pth
COPY app.py /app/app.py

EXPOSE 5000

ENV FLASK_ENV=production

CMD ["python", "app.py"]