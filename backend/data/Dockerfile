# Dockerfile for data_update service
FROM python:3.9-slim

WORKDIR /app

COPY . /app
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script and any required files
COPY data_updater.py .
COPY .env .  # For environment variables

# Run the script
CMD ["python", "data_updater.py"]