FROM python:3.8

# Set working directory
WORKDIR /pa_recognition

ENV PYTHONPATH /pa_recognition

# Upgrade pip
RUN pip install --upgrade pip

# Clear pip cache
RUN pip cache purge

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code directory
COPY . /pa_recognition

# Set the default command to bash
ENTRYPOINT [ "/bin/bash" ]