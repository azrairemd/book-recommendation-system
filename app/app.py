from flask import Flask, render_template
from recommender_service import get_recommendation

app = Flask(__name__)

@app.route("/")
def index():
    user_id, books = get_recommendation(top_n=5)
    return render_template(
        "index.html",
        user_id=user_id,
        books=books.to_dict(orient="records")
    )

@app.route("/recommend")
def recommend():
    user_id, books = get_recommendation(top_n=5)
    return render_template(
        "index.html",
        user_id=user_id,
        books=books.to_dict(orient="records")
    )

if __name__ == "__main__":
    app.run(debug=True)
