# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt to install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY templates .

# Expose the port Flask runs on
EXPOSE 5001

# Run the app with Gunicorn (better for production)
CMD ["gunicorn", "-b", "0.0.0.0:5001", "app:app"]
