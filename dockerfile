# Use the official Python base image
FROM python:3.11-slim

# Set environment variable for the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (backend and frontend HTML)
COPY app.py .
COPY index.html .

# Expose the port Uvicorn will run on
EXPOSE 8000

# Command to run the application using Uvicorn directly (The Fix for the ASGI/WSGI issue)
# This uses the standard Uvicorn command to launch the ASGI application.
# It binds to 0.0.0.0:$PORT (which Render passes as 8000).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
