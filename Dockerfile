# 1. Use a lightweight Python image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the necessary files for the API to run
COPY requirements.txt .
COPY model.joblib .
COPY app.py .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose the port FastAPI runs on
EXPOSE 8000

# 6. Run the API when the container starts
# We use --host 0.0.0.0 to allow connections from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]