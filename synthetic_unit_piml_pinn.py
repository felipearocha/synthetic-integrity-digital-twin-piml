
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random

random.seed(42)
np.random.seed(42)

def generate_equipment(n=20):
    rows = []
    for i in range(1, n+1):
        rows.append({
            "Tag": f"EQ-{i:04d}",
            "Type": random.choice(["TANK","COLUMN","VESSEL","PUMP","EXCHANGER"]),
            "X": random.uniform(0,200),
            "Y": random.uniform(0,200),
            "Height": random.uniform(5,30)
        })
    return pd.DataFrame(rows)

def generate_piping(n=40):
    rows = []
    for i in range(1, n+1):
        rows.append({
            "Line": f"LN-{i:04d}",
            "Service": random.choice(["Sour Water","Hydrocarbon","Amine","Steam"]),
            "Operating Temp C": random.uniform(20,180),
            "Operating Pressure kPag": random.uniform(50,2000),
            "Chloride ppm": random.randint(0,500),
            "pH": round(random.uniform(3,9),2)
        })
    return pd.DataFrame(rows)

def create_3d_visual(equipment):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=equipment["X"],
        y=equipment["Y"],
        z=equipment["Height"],
        mode="markers+text",
        text=equipment["Tag"],
        marker=dict(size=5)
    ))
    fig.update_layout(scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Height"
    ))
    fig.write_html("outputs/synthetic_refinery_unit_3d.html")

if __name__ == "__main__":
    eq = generate_equipment()
    pipe = generate_piping()
    eq.to_csv("data/equipment_anonymized.tsv", sep="\t", index=False)
    pipe.to_csv("data/piping_attributes_anonymized.tsv", sep="\t", index=False)
    create_3d_visual(eq)
    print("Synthetic integrity digital twin generated successfully.")
