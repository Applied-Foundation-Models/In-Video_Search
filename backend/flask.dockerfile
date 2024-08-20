FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Install required packages including PostgreSQL client
RUN apt-get update && apt-get install -y \
    libpq-dev \
    postgresql-client \
    && apt-get clean

# Copy the requirements file to the working directory
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the working directory
COPY . /app/

# Expose the Flask port
EXPOSE 4000

# Set the Flask app environment variable to point to the correct module
ENV FLASK_APP=backend.run

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=4000"]
