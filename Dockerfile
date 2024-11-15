FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

WORKDIR $APP_HOME

RUN apt-get update && \
    apt-get install -y poppler-utils && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit","run","app.py"]