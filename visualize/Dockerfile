FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# tylko biblioteka standardowa - bez requirements.txt
COPY visualize.py .

EXPOSE 8080
CMD ["python", "visualize.py"]
