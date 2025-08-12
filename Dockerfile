# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY src/ ./src/
COPY data/ ./data/

# Expose port 8080
EXPOSE 8080

# Run the FastAPI app
CMD ["python", "./src/App.py"]
