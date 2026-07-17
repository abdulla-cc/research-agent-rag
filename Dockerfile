# start from a slim Python 3.12 base image (matches your local version)
FROM python:3.12-slim

# HF Spaces convention: run as a non-root user, app lives in /app
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

# install the CPU-only version of torch FIRST, explicitly — this avoids
# pulling the giant GPU build (HF free tier is CPU-only). Big size saving.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

# copy just requirements first, install them — this caches the layer so
# rebuilds are fast when only your code changes (Docker best practice)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# now copy your application code and the pre-built vector DB
COPY --chown=user src/ ./src/
COPY --chown=user data/chroma/ ./data/chroma/

# HF Spaces expects the app on port 7860
EXPOSE 7860

# pre-download the embedding model at BUILD time so the container doesn't
# have to fetch it on first request (faster cold start, works offline)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# start the server — note: NO --reload in production, and bind to 0.0.0.0
# so it's reachable from outside the container. cd into src so imports work.
CMD ["sh", "-c", "cd src && uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]
