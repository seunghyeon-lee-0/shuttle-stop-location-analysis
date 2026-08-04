# Location-Allocation-Based Shuttle Bus Route and Headway Optimization for a Regional Festival

<br>

🏆 **Excellence Award , 2024 BIG CONTEST**

<br>
<br>

This project develops a data-driven shuttle bus planning framework for a **regional festival** experiencing temporary traffic congestion and parking shortages.

The framework integrates **shuttle stop location optimization**, **route clustering**, and **time-based passenger demand forecasting** to determine where shuttle stops should be placed, how they should be organized into routes, and how frequently buses should operate.

<br>

[📊 Analysis Results](research_results.md)  
[📑 Presentation Slides](bigcontest-presentation.pdf)

<br>

---

## 💡 Project Motivation

Regional festivals attract large numbers of visitors within a limited time and area, often causing traffic congestion, parking shortages, and reduced accessibility.

Exploratory analysis showed a sharp increase in tourism-related mobility during the festival period and a high proportion of private vehicle use. The results indicated the need for a temporary shuttle bus network designed around actual visitor movement and transportation demand.

<br>


<br>

<p align="center">
  <img src="images/research procedure.png" alt="motivation" width="900">
</p>


---

## 📚 Data Sources

- **SKT Mobility Data**: Administrative-district OD data and stay-population data used to analyze visitor movement and time-based demand
- **Local Government Open Data**: Population, bus stop, administrative-area, and tourist-attraction data used to evaluate candidate locations
- **Transportation Card Big Data System**: Stop-level boarding and alighting data used to measure public transportation demand
- **Ministry of Land, Infrastructure and Transport**: Administrative boundaries and cadastral data used for GIS-based distance and coverage analysis
- **Ministry of the Interior and Safety**: Administrative-district codes used to integrate datasets
- **Regional Economy Portal**: Festival-related social text data used to analyze perceptions of shuttle and parking services

<br>

---

### MCLP

The **Maximal Covering Location Problem** selects a limited number of shuttle stops while maximizing weighted demand covered within a **400 m service radius**.

It was used to identify locations that could serve the largest possible number of potential passengers.

<br>

### P-Median

The **P-Median model** selects a fixed number of stops while minimizing the total weighted distance between demand points and their assigned stops.

The objective function was modified to consider both access distance and AHP-based location importance.


<br>

<p align="center">
  <img src="images/final-results.png" alt="Final Shuttle Bus Operation Plan" width="900">
</p>

<br>

The final framework selected 22 stops with limited spatial overlap and organized them into two routes serving different areas around the festival venue.

Rather than applying a fixed timetable, the project proposed flexible headways based on predicted passenger demand throughout the day.

<br>

---

## 📁 Repository Structure

```text
regional-festival-shuttle-optimization/
├── README.md
├── research_results.md
├── bigcontest-presentation.pdf
├── requirements.txt
├── LICENSE
│
├── images/
│   ├── analysis-pipeline.png
│   ├── location-optimization.png
│   ├── route-clustering.png
│   └── final-results.png
│
├── scripts/
│   ├── 01_preprocess_mobility_data.py
│   ├── 02_calculate_ahp_weights.py
│   ├── 03_run_mclp.py
│   ├── 04_run_pmedian.py
│   ├── 05_select_final_stops.py
│   ├── 06_cluster_routes.py
│   ├── 07_forecast_demand.py
│   └── run_pipeline.py
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_location_optimization.ipynb
│   ├── 03_route_clustering.ipynb
│   └── 04_demand_forecasting.ipynb
│
├── data/
│   ├── raw/
│   ├── reference/
│   └── processed/
│
└── results/
    ├── final_shuttle_stops.csv
    ├── shuttle_routes.csv
    ├── headway_plan.csv
    └── figures/
```

<br>

---
