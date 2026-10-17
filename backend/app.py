from datetime import datetime

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


def compute_value(t):
    """Turn a time into a value.
    Currently just returns the time itself (e.g. 13:30 -> "13:30").
    """
    return t.strftime("%H:%M")


def parse_time(raw):
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p"):
        try:
            return datetime.strptime(raw.strip(), fmt).time()
        except ValueError:
            continue
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/value", methods=["GET", "POST"])
def api_value():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        raw = data.get("time")
    else:
        raw = request.args.get("time")

    if not raw:
        return jsonify(error="Missing 'time' (e.g. 13:30 or 1:30 PM)"), 400

    t = parse_time(raw)
    if t is None:
        return jsonify(error=f"Could not parse time: {raw!r}"), 400

    return jsonify(time=t.strftime("%H:%M"), value=compute_value(t))


if __name__ == "__main__":
    app.run(debug=True)