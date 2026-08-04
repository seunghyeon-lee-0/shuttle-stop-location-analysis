# Location-Allocation-Based Shuttle Bus Route and Headway Optimization for a Regional Festival

🏆 **Excellence Award, 2024 BIG CONTEST**

<br>

This project develops a data-driven shuttle bus planning framework to address traffic congestion, parking shortages, and limited accessibility during a regional festival.

The framework integrates location-allocation models, route clustering, and passenger demand forecasting to optimize shuttle stop locations, operating routes, and time-based headways.

<br>

[📊 Analysis Results](research_results.md)  
[📑 Presentation Slides](bigcontest-presentation.pdf)

<br>

---

## 💡 Project Motivation

Regional festivals attract large numbers of visitors within a limited time and area, often causing traffic congestion and parking shortages.

Exploratory analysis showed that visitor mobility increased sharply during the festival period and that private vehicles accounted for a high proportion of travel. These findings indicated the need for a temporary shuttle bus network based on actual mobility patterns and transportation demand.

<br>

<p align="center">
  <img src="images/research procedure.png" alt="Project Motivation" width="900">
</p>

<br>

---

## 📚 Data Sources

- **SKT Mobility Data**: Administrative-district OD data and stay-population data used to analyze visitor movement and time-based demand
- **Local Government Open Data**: Population, bus stop, tourist-attraction, and administrative-area data used to evaluate candidate locations
- **Transportation Card Big Data System**: Stop-level boarding and alighting data used to measure public transportation demand
- **Ministry of Land, Infrastructure and Transport**: Administrative boundaries and cadastral data used for GIS-based distance and coverage analysis
- **Ministry of the Interior and Safety**: Administrative-district codes used to integrate datasets
- **Regional Economy Portal**: Festival-related social text data used to examine perceptions of shuttle and parking services

<br>

---

## 🗂️ Analysis Procedure

The analysis combines location weighting, stop selection, route clustering, and demand forecasting to develop an integrated shuttle bus operation plan for a regional festival.

<br>

<p align="center">
  <img src="images/analysis-pipeline.png" alt="Analysis Pipeline" width="900">
</p>

<br>



### MCLP

Selects shuttle stops that maximize passenger-demand coverage within a **400 m service radius**.

<br>



### P-Median

Selects shuttle stops that minimize access distance while reflecting AHP-based location importance.

<br>


---

## 📊 Results

The final framework selected **22 shuttle stops** and organized them into **two routes** serving different areas around the festival venue.

It also proposed time-based headways based on predicted passenger demand.


<br>

<p align="center">
  <img src="images/final-results-2.png" alt="final-results" width="900">
</p>
