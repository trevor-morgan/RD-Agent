FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy code
COPY api_server.py .
COPY production_semantic_trader.py .
COPY semantic_space_network.py .
COPY semantic_space_data_loader.py .
COPY semantic_network_best.pt .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
