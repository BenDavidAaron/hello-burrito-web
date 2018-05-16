from flask import Flask, render_template, request
from flask_wtf import Form
from wtforms import SelectField, SubmitField
import numpy as np
import pandas as pd
import pickle


def titleize_burrito_row(i, pd_row):
    b_type = pd_row[0].strip().title()
    b_vendor = pd_row[1].strip()
    b_cost = pd_row[8]
    title = "#{i} {b_vendor} Burrito @ {b_type}(${b_cost})"
    if len(title) > 20:
        split = title.split("@")
        title = "\n".join(split)
    return f"#{i} {b_vendor} Burrito @ {b_type}(${b_cost})"


app = Flask(__name__)
app.vars = {}
app.secret_key = "development key"

app.vars["df_complete"] = pickle.load(open("clean-burrito-pandas.pkl", "rb"))
features = [
    "Chips_1h",
    "Beef_1h",
    "Pico_1h",
    "Guac_1h",
    "Cheese_1h",
    "Fries_1h",
    "Sour cream_1h",
    "Pork_1h",
    "Chicken_1h",
    "Shrimp_1h",
    "Fish_1h",
    "Rice_1h",
    "Beans_1h",
    "Lettuce_1h",
    "Tomato_1h",
    "Bell peper_1h",
    "Carrots_1h",
    "Cabbage_1h",
    "Sauce_1h",
    "Salsa.1_1h",
    "Cilantro_1h",
    "Onion_1h",
    "Taquito_1h",
    "Pineapple_1h",
    "Ham_1h",
    "Chile relleno_1h",
    "Nopales_1h",
    "Lobster_1h",
    "Queso_1h",
    "Egg_1h",
    "Mushroom_1h",
    "Bacon_1h",
    "Sushi_1h",
    "Avocado_1h",
    "Corn_1h",
    "Zucchini_1h",
    "Cost",
    "Yelp",
    "Google",
]
app.vars["df_features"] = app.vars["df_complete"][features]
app.vars["knn"] = pickle.load(open("k-nearest-burritos.pkl", "rb"))

app.vars["burrito_titles"] = [
    titleize_burrito_row(idx, row) for idx, row in app.vars["df_complete"].iterrows()
]


class Burrito_Picker(Form):
    pick = SelectField(
        "user_choice",
        choices=[(idx, title) for idx, title in enumerate(app.vars["burrito_titles"])],
    )
    submit = SubmitField("Confirm")


@app.route("/", methods=["GET", "POST"])
def index():
    burrito_picker = Burrito_Picker()
    if request.method == "GET":
        return render_template(
            "index.html",
            burrito_picker=burrito_picker,
            burrito_list=app.vars["burrito_titles"],
        )

    else:
        query_index = request.values["pick"]
        #print(query_index)
        #print(app.vars["df_features"].iloc[[query_index]]) #why the heck is the index a list within a list? Whatever it'll work
        predictions = app.vars["knn"].kneighbors(
            app.vars["df_features"].iloc[[query_index]]
        )
        scores, neighbors = predictions[0][0], predictions[1][0]#unpack the goofy predictions data structure
        return render_template(
            "index.html",
            burrito_picker=burrito_picker,
            burrito_list=app.vars["burrito_titles"],
            neighbors={'n':neighbors, 's':scores}
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=33507, debug=True)
