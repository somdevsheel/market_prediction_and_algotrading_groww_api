FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TZ=Asia/Kolkata
COPY pyproject.toml .
RUN pip install --no-cache-dir growwapi==1.5.0 "pandas>=2.2" "numpy>=1.26" "scipy>=1.11" "pyarrow>=15" "duckdb>=1.0" \
    "pyyaml>=6" "fastapi>=0.110" "uvicorn>=0.29" "requests>=2.31" "pyotp>=2.9" "pytest>=8" "httpx>=0.27"
COPY . .
CMD ["python", "-m", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
