FROM node:20-slim AS web
WORKDIR /web
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY apps/web/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TZ=Asia/Kolkata
RUN pip install --no-cache-dir growwapi==1.5.0 "pandas>=2.2" "numpy>=1.26" "scipy>=1.11" "pyarrow>=15" "duckdb>=1.0" \
    "pyyaml>=6" "fastapi>=0.110" "uvicorn>=0.29" "requests>=2.31" "pyotp>=2.9" "scikit-learn>=1.4" "pytest>=8" "httpx>=0.27"
COPY . .
COPY --from=web /web/dist /app/apps/web/dist
CMD ["python", "-m", "uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
