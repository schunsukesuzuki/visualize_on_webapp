FROM python:3.11-slim

WORKDIR /app

# dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade scipy

# copy app code
COPY . .

# making folder for data
RUN mkdir -p ../data

# port exposure
EXPOSE 8000

# boot app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# FROM python:3.11

# WORKDIR /app

# # for SciPy / NumPyro / JAX 
# RUN apt-get update && apt-get install -y \
#     gfortran \
#     libopenblas-dev \
#     liblapack-dev \
#     && rm -rf /var/lib/apt/lits/*

# COPY requirements.txt .
d
# RUN pip install --upgrade pip setuptools wheel
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# EXPOSE 8000

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
