FROM python:3.11-slim

WORKDIR /app

# System deps for bleak (BLE) and zeroconf
RUN apt-get update && apt-get install -y --no-install-recommends \
    bluez \
    libbluetooth-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 18789

CMD ["python", "-m", "dashboard.server"]
