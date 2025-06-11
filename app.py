from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename
from topsis import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Linguistic dictionaries
CRITERIA_LINGUISTIC_MAP = {
    "wi": wi, "ei": ei, "si": si, "vsi": vsi, "ai": ai
}

ALTERNATIVE_LINGUISTIC_MAP = {
    "vb": vb, "b": b, "mb": mb, "m": m, "mg": mg, "g": g, "vg": vg
}

EXPERT_COUNT = 10 

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        alt_matrix = []
        file = request.files.get("excel_file")
        if file and file.filename != "":
            # Excel input
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            df = pd.read_excel(filepath)
            alternatives = df["Alternative"].tolist()
            linguistic_df = df.drop(columns=["Alternative"])

            num_criteria = int(linguistic_df.shape[1] / EXPERT_COUNT)
            num_alternatives = len(alternatives)

            for expert in range(EXPERT_COUNT):
                for alt_index in range(num_alternatives):
                    row = []
                    for crit_index in range(num_criteria):
                        cell_val = linguistic_df.iloc[alt_index, expert * num_criteria + crit_index]
                        cell_val = str(cell_val).strip()
                        if cell_val not in ALTERNATIVE_LINGUISTIC_MAP:
                            return f"ERROR: '{cell_val}' is an invalid value (Excel cell)"
                        row.append(ALTERNATIVE_LINGUISTIC_MAP[cell_val])
                    alt_matrix.append(row)

            alt_matrix = np.array(alt_matrix)

            weights_matrix = [ai] * num_criteria 
        else:
            #Manual form
            criteria = request.form.getlist("criteria[]")
            alternatives = request.form.getlist("alternatives[]")

            num_criteria = len(criteria)
            num_alts = len(alternatives)

            #Collect weights for 10 experts
            weights_matrix = []
            for i in range(num_criteria):
                try:
                    partials = []
                    for e in range(1, EXPERT_COUNT + 1):
                        w = CRITERIA_LINGUISTIC_MAP[request.form.getlist(f"weights_e{e}[]")[i]]
                        partials.append(w)
                    sum_w = partials[0]
                    for pw in partials[1:]:
                        sum_w = t2nn_add(sum_w, pw)
                    agg = t2nn_scalar_mult(sum_w, 1 / EXPERT_COUNT)
                    weights_matrix.append(agg)
                except Exception as e:
                    return f"ERROR: Could not process weights — {str(e)}"

            #Alternative inputs
            raw_matrix = []
            for i in range(num_alts * EXPERT_COUNT):
                row = []
                for j in range(num_criteria):
                    val = request.form.get(f"cell_{i}_{j}")
                    if val is None or val.strip() == "":
                        return f"ERROR: Missing input — cell_{i}_{j} is left empty."
                    if val not in ALTERNATIVE_LINGUISTIC_MAP:
                        return f"ERROR: '{val}' is not valid (cell_{i}_{j})"
                    row.append(ALTERNATIVE_LINGUISTIC_MAP[val])
                raw_matrix.append(row)

            alt_matrix = np.array(raw_matrix)

        #TOPSIS calculations
        scored_opinion = np.array([score(w) for w in weights_matrix])
        norm_opinion = scored_opinion / np.linalg.norm(scored_opinion)

        agg_alt = get_aggregated_alternatives(alt_matrix, EXPERT_COUNT)
        scored_alt = get_scored_alternatives(agg_alt)
        norm_alt = get_normalized_alternatives(scored_alt)
        weighted = get_weighted_alternative_matrix(norm_alt, norm_opinion)

        num_criteria = weighted.shape[1]
        benefit_idx = list(range(num_criteria))
        cost_idx = []

        dp = calculate_dp(weighted, calculate_npis(weighted, benefit_idx, cost_idx))
        dn = calculate_dn(weighted, calculate_nnis(weighted, benefit_idx, cost_idx))
        scores = calculate_proximity_coeffs(dp, dn)

        results = list(zip(alternatives, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        return render_template("result.html", results=enumerate(results, start=1))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
