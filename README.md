# Smart Leader Loan Prediction

A simple Flask web app for loan approval prediction using a trained model.

## Structure

- `app.py` - Flask application entry point
- `templates/` - HTML templates for pages
- `static/` - optional static assets (CSS / JS)
- `data_sets/` - source data files
- `model.pkl` / `scaler.pkl` - trained model and scaler pickle files

## Run locally

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it and install dependencies:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open `http://127.0.0.1:5000/` in your browser.
