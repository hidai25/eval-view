FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir evalview

EXPOSE 8000

CMD ["evalview", "mcp", "serve"]
