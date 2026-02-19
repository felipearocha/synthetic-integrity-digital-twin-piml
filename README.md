Synthetic Integrity Digital Twin (PIML + PINN Framework)
Overview

This project generates a fully synthetic refinery-style digital twin including:

3D equipment visualization (tanks, vessels, columns, pumps, exchangers)

Routed piping topology

Corrosion and chemical attributes

RBI-style integrity metadata

100 synthetic SCC alternative scenarios

Optional Physics-Informed Machine Learning (PIML / PINN) demo

All data is synthetic and anonymized.

This repository is intended for:

Research

Digital twin prototyping

Integrity analytics architecture design

AI experimentation for corrosion and cracking

It is NOT a certified engineering calculator and does not replace API 579 or ASME FFS assessments.

Engineering Context

Conceptually aligned with:

API 570 (Piping Inspection)

API 571 (Damage Mechanisms)

API 580 / 581 (RBI)

API 579-1 / ASME FFS-1 (Fitness-for-Service)

ASME B31.3 (Process Piping)

These references are conceptual only.

Features
1. Synthetic Equipment Model

Tagged equipment

Design and operating conditions

Material specification

Insulation and coating metadata

2. Piping Attributes

Service type

Chlorides

pH

H2S

Corrosion rate (MPY and mm/y)

SCC susceptibility score (0–1)

3. 3D Interactive Visualization

Generates:
synthetic_refinery_unit_3d.html

Open in browser to rotate, zoom and inspect.

4. PIML / PINN Demo

Optional physics-informed neural network to:

Predict SCC susceptibility

Apply Paris-type crack growth constraint

Demonstrate hybrid physics + ML architecture

Installation

pip install -r requirements.txt

Optional (for ML):
pip install torch

Run

python synthetic_unit_piml_pinn.py

Outputs

data/equipment_anonymized.tsv

data/piping_attributes_anonymized.tsv

outputs/synthetic_refinery_unit_3d.html

Future Extensions

Real P&ID parser integration

Crack growth time simulation

Remaining life forecasting

Monte Carlo reliability modeling

Integration with Power BI dashboards

Risk prioritization engine
