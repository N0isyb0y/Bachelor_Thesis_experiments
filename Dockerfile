# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock first to optimize Docker cache
COPY Pipfile Pipfile.lock ./

# Install Pipenv
RUN pip install --no-cache-dir pipenv

# Ensure Pipenv uses Python 3.12 inside the container
ENV PIPENV_PYTHON=/usr/local/bin/python3.12

# Install dependencies (use --ignore-pipfile to avoid strict dependency issues)
RUN pipenv install --ignore-pipfile

# Copy the rest of the application files
COPY . .

# Expose the application port (adjust if needed)
EXPOSE 5000

# Command to start the application inside the container
CMD ["pipenv", "run", "python", "main.py"]
