from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# -----------------------------------------------
# Load trained KMeans++ model and scaler
# -----------------------------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Friendly cluster descriptions based on typical KMeans++ results
# on Mall Customers dataset (5 clusters)
CLUSTER_LABELS = {
    0: ("💰 High Income, Low Spender", "Careful with money. Earns well but shops selectively."),
    1: ("🛍️ High Income, High Spender", "Premium customer! Loves to shop and can afford it."),
    2: ("📊 Average Income, Average Spender", "Typical middle-ground shopper."),
    3: ("⚠️ Low Income, High Spender", "Impulsive buyer. Spends beyond their income level."),
    4: ("💤 Low Income, Low Spender", "Budget-conscious. Low income and low spending."),
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        annual_income = float(request.form.get("annual_income"))
        spending_score = float(request.form.get("spending_score"))

        # Validate ranges
        if not (0 <= annual_income <= 200):
            raise ValueError("Annual income should be between 0 and 200 (k$)")
        if not (1 <= spending_score <= 100):
            raise ValueError("Spending score should be between 1 and 100")

        # Scale input the same way training data was scaled
        input_data = np.array([[annual_income, spending_score]])
        input_scaled = scaler.transform(input_data)

        # Predict cluster
        cluster = model.predict(input_scaled)[0]

        label, description = CLUSTER_LABELS.get(cluster, ("Unknown Cluster", ""))

        return render_template(
            "index.html",
            prediction=True,
            cluster_number=int(cluster) + 1,
            cluster_label=label,
            cluster_description=description,
            income=annual_income,
            score=spending_score
        )

    except ValueError as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
