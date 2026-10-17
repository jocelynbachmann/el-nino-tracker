# Time → Value

A small Flask app that takes a time and returns a value.

## Run

```
pip install flask
python app.py
```

Open http://127.0.0.1:5000 to use the input page.

## API

`GET /api/value?time=13:30` or `POST /api/value` with `{"time": "13:30"}`

```json
{"time": "13:30", "value": "13:30"}
```

It accepts `HH:MM`, `HH:MM:SS` and `1:30 PM`. Invalid input returns a 400 with an error message.

## Customize

To change what the value is, edit `compute_value()` in `app.py`.