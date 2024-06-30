FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000


CMD ["chainlit", "run", "chatbot.py", "-w" ,"--port", "8000"]



