# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV KAGGLE_USERNAME=peixuanli1107
ENV KAGGLE_KEY=e3288a92bce400d18f4bc3bd3d94a140

# Set the default command to run the training script
CMD ["python", "main.py"]