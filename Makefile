.PHONY: ingest app dash eval-retr


setup:
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt


ingest:
python ingest/crawl_docs.py && python ingest/chunk_and_embed.py


app:
streamlit run app/ui_app.py


docker:
docker compose -f docker/docker-compose.yml up --build