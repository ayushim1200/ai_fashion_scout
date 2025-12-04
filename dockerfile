# Use the official Python base image
FROM python:3.11-slim

# Set environment variable for the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
# Use --no-cache-dir to keep the image size small
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY app.py .

# Expose the port Uvicorn will run on
EXPOSE 8000

# Command to run the application using Uvicorn
# 'app:app' means look for the 'app' variable in the 'app.py' file
# We use Gunicorn to manage Uvicorn workers for production stability
CMD ["gunicorn", "app:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
