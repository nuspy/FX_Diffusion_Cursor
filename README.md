MagicForex: MVP forecasting FX con diffusion latente e GUI.

Setup rapido (Python 3.12):
- Configura `ALPHAVANTAGE_API_KEY` env.
- Installa: `pip install -e .` (usa `pyproject.toml`).
- Migrazioni: `alembic upgrade head` (SQLite in `data/forex.db`).
- Avvio API: `forex-api` (FastAPI su 127.0.0.1:8080).
- Avvio GUI: `forex-gui` (placeholder, verrà aggiunta).

Note:
- Config in `configs/default.yaml` (strumenti, provider, training, GUI).
- Log rotanti in `data/MagicForex.log`.
