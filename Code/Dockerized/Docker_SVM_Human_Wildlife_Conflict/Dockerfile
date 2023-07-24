FROM python:3.9

# Set working directory
WORKDIR /src

# Copy directories into the container
COPY Datasets /src/Datasets

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Run the application
CMD ["python","Support Vector Machine SVMs ; Human Wildlife Conflict - Exercise 2.py"]