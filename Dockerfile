# Use a Python slim base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API and Model folders into the container
COPY api/ /app/api/
COPY models/ /app/models/

# Set PYTHONPATH to include /app (so Python can find 'api')
ENV PYTHONPATH=/app

# Expose port 5000 for Flask
EXPOSE 5000

# Define the command to run the Flask app
CMD ["python", "api/app.py"]
