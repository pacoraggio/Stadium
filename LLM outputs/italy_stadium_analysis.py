"""
Statistical Analysis: Is Italy an outlier in new stadium construction (2005–2025)?

Approach:
- Dataset: new/fully rebuilt stadiums (2005–2025) per UEFA nation, from our StadiumDB
  data collected earlier (no seat constraint), normalized by top-division club count
- Model: Poisson regression — stadium counts are non-negative integers, classic Poisson territory
- Null hypothesis (H0): Italy's stadium count is consistent with the expected count
  given its number of top-division clubs
- We fit a simple Poisson GLM: E[stadiums_i] = exp(alpha + beta * log(clubs_i))
  i.e. stadium count scales with number of top-division clubs
- Then compute the p-value for Italy's observed count under the fitted model

Data sources:
- Stadium counts: StadiumDB.com (our earlier research, no seat constraint)
- Top-division clubs: Wikipedia "List of top-division football clubs in UEFA countries"
  (current top-flight league sizes, used as a stable proxy for football infrastructure)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# ── 1. DATA ──────────────────────────────────────────────────────────────────
# Columns: country, new_stadiums (2005-2025, StadiumDB, no capacity filter),
#          top_div_clubs (current top-flight league size)
#
# Stadium counts come from our earlier StadiumDB research.
# Countries with 0 confirmed new builds in the period are included.
# Top-division clubs: current league sizes from Wikipedia.

data = {
    # Major nations (well-researched in our earlier work)
    "Turkey":          {"new_stadiums": 21, "top_div_clubs": 19},
    "Russia":          {"new_stadiums": 13, "top_div_clubs": 16},
    "England":         {"new_stadiums":  7, "top_div_clubs": 20},
    "Germany":         {"new_stadiums":  6, "top_div_clubs": 18},
    "Poland":          {"new_stadiums":  5, "top_div_clubs": 18},
    "Ukraine":         {"new_stadiums":  5, "top_div_clubs": 16},
    "Hungary":         {"new_stadiums":  4, "top_div_clubs": 12},
    "Romania":         {"new_stadiums":  4, "top_div_clubs": 16},
    "France":          {"new_stadiums":  4, "top_div_clubs": 18},
    "Spain":           {"new_stadiums":  4, "top_div_clubs": 20},  # incl. partial rebuilds
    "Sweden":          {"new_stadiums":  3, "top_div_clubs": 16},
    "Denmark":         {"new_stadiums":  2, "top_div_clubs": 14},
    "Belgium":         {"new_stadiums":  2, "top_div_clubs": 16},
    "Switzerland":     {"new_stadiums":  2, "top_div_clubs": 12},
    "Austria":         {"new_stadiums":  2, "top_div_clubs": 12},
    "Israel":          {"new_stadiums":  2, "top_div_clubs": 14},
    "Italy":           {"new_stadiums":  6, "top_div_clubs": 20},  # our verified count
    "Netherlands":     {"new_stadiums":  1, "top_div_clubs": 18},
    "Norway":          {"new_stadiums":  1, "top_div_clubs": 16},
    "Wales":           {"new_stadiums":  1, "top_div_clubs": 12},
    "Ireland":         {"new_stadiums":  1, "top_div_clubs": 10},
    "Slovakia":        {"new_stadiums":  1, "top_div_clubs": 12},
    "Czech Republic":  {"new_stadiums":  1, "top_div_clubs": 16},
    "Albania":         {"new_stadiums":  1, "top_div_clubs": 10},
    "North Macedonia": {"new_stadiums":  1, "top_div_clubs": 10},
    "Kazakhstan":      {"new_stadiums":  1, "top_div_clubs": 16},
    "Azerbaijan":      {"new_stadiums":  1, "top_div_clubs": 14},
    "Georgia":         {"new_stadiums":  1, "top_div_clubs": 10},
    "Belarus":         {"new_stadiums":  1, "top_div_clubs": 16},
    # Nations with 0 confirmed new builds
    "Portugal":        {"new_stadiums":  0, "top_div_clubs": 18},
    "Scotland":        {"new_stadiums":  0, "top_div_clubs": 12},
    "Greece":          {"new_stadiums":  1, "top_div_clubs": 16},  # OPAP Arena 2022
    "Croatia":         {"new_stadiums":  0, "top_div_clubs": 10},
    "Serbia":          {"new_stadiums":  0, "top_div_clubs": 16},
    "Bulgaria":        {"new_stadiums":  0, "top_div_clubs": 14},
    "Finland":         {"new_stadiums":  0, "top_div_clubs": 12},
    "Bosnia-Herz.":    {"new_stadiums":  0, "top_div_clubs": 12},
    "Cyprus":          {"new_stadiums":  0, "top_div_clubs": 14},
    "Northern Ireland":{"new_stadiums":  0, "top_div_clubs": 12},
    "Lithuania":       {"new_stadiums":  0, "top_div_clubs": 10},
    "Latvia":          {"new_stadiums":  0, "top_div_clubs": 10},
    "Estonia":         {"new_stadiums":  0, "top_div_clubs": 10},
    "Iceland":         {"new_stadiums":  0, "top_div_clubs": 12},
    "Luxembourg":      {"new_stadiums":  0, "top_div_clubs": 14},
    "Moldova":         {"new_stadiums":  0, "top_div_clubs": 10},
    "Armenia":         {"new_stadiums":  0, "top_div_clubs": 10},
    "Kosovo":          {"new_stadiums":  0, "top_div_clubs": 10},
    "Slovenia":        {"new_stadiums":  0, "top_div_clubs": 10},
    "Montenegro":      {"new_stadiums":  0, "top_div_clubs": 10},
}

df = pd.DataFrame(data).T.reset_index()
df.columns = ["country", "new_stadiums", "top_div_clubs"]
df = df.astype({"new_stadiums": int, "top_div_clubs": int})

# ── 2. EXPLORATORY STATS ─────────────────────────────────────────────────────
print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(f"Countries in dataset     : {len(df)}")
print(f"Total new stadiums       : {df['new_stadiums'].sum()}")
print(f"Mean stadiums / country  : {df['new_stadiums'].mean():.2f}")
print(f"Median stadiums/country  : {df['new_stadiums'].median():.1f}")
print(f"\nTop 10 builders:")
print(df.nlargest(10, "new_stadiums")[["country","new_stadiums","top_div_clubs"]].to_string(index=False))

italy = df[df["country"] == "Italy"].iloc[0]
print(f"\nItaly: {italy['new_stadiums']} new stadiums, {italy['top_div_clubs']} top-div clubs")

# ── 3. NAIVE (EQUAL PROBABILITY) TEST ────────────────────────────────────────
# H0: each country is equally likely to build any given stadium
# Under H0, total T stadiums distributed uniformly across N countries
# => each country gets Binomial(T, 1/N) ~ Poisson(T/N) stadiums

print("\n" + "=" * 60)
print("MODEL 1: NAIVE EQUAL-PROBABILITY (Poisson, lambda = T/N)")
print("=" * 60)
T = df["new_stadiums"].sum()
N = len(df)
lam_naive = T / N
print(f"Total stadiums T={T}, Countries N={N}, lambda = {lam_naive:.2f}")

italy_obs = int(italy["new_stadiums"])
# Two-tailed p-value: P(X <= obs) or P(X >= obs), take 2 * min
p_lower = stats.poisson.cdf(italy_obs, lam_naive)
p_upper = 1 - stats.poisson.cdf(italy_obs - 1, lam_naive)
p_naive = 2 * min(p_lower, p_upper)
print(f"Italy observed = {italy_obs}, E[X] under H0 = {lam_naive:.2f}")
print(f"P(X >= {italy_obs}) = {p_upper:.4f}")
print(f"Two-tailed p-value     = {p_naive:.4f}")
conclusion_naive = "SIGNIFICANT (Italy is an outlier)" if p_naive < 0.05 else "NOT significant (Italy is NOT an outlier)"
print(f"Conclusion (alpha=0.05): {conclusion_naive}")

# ── 4. POISSON GLM NORMALIZED BY CLUB COUNT ──────────────────────────────────
# More principled: E[stadiums_i] = mu_i, where log(mu_i) = alpha + beta*log(clubs_i)
# This is a log-linear Poisson GLM — fit by MLE

print("\n" + "=" * 60)
print("MODEL 2: POISSON GLM (log-linear, covariate = log(top_div_clubs))")
print("=" * 60)

y = df["new_stadiums"].values
x = np.log(df["top_div_clubs"].values)
X = np.column_stack([np.ones(len(x)), x])  # design matrix [intercept, log(clubs)]

def neg_log_likelihood(params):
    alpha, beta = params
    mu = np.exp(alpha + beta * x)
    # Poisson log-likelihood: sum(y*log(mu) - mu - log(y!))
    ll = np.sum(y * np.log(mu + 1e-12) - mu)
    return -ll

result = minimize(neg_log_likelihood, x0=[0.0, 1.0], method="Nelder-Mead",
                  options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 10000})
alpha_hat, beta_hat = result.x

print(f"Fitted alpha (intercept) : {alpha_hat:.4f}")
print(f"Fitted beta  (log-clubs) : {beta_hat:.4f}")

# Fitted mu for Italy
italy_clubs = int(italy["top_div_clubs"])
mu_italy = np.exp(alpha_hat + beta_hat * np.log(italy_clubs))
print(f"\nItaly top-div clubs = {italy_clubs}")
print(f"Fitted E[stadiums | Italy] = {mu_italy:.2f}")
print(f"Italy observed             = {italy_obs}")

# P-value under fitted Poisson(mu_italy)
p_lower_glm = stats.poisson.cdf(italy_obs, mu_italy)
p_upper_glm = 1 - stats.poisson.cdf(italy_obs - 1, mu_italy)
p_glm = 2 * min(p_lower_glm, p_upper_glm)
print(f"P(X >= {italy_obs} | mu={mu_italy:.2f}) = {p_upper_glm:.4f}")
print(f"Two-tailed p-value         = {p_glm:.4f}")
conclusion_glm = "SIGNIFICANT (Italy is an outlier)" if p_glm < 0.05 else "NOT significant (Italy is NOT an outlier)"
print(f"Conclusion (alpha=0.05)    : {conclusion_glm}")

# ── 5. FULL COUNTRY COMPARISON ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 2 — FITTED VALUES & RESIDUALS FOR ALL COUNTRIES")
print("=" * 60)

df["log_clubs"] = np.log(df["top_div_clubs"])
df["mu_fitted"] = np.exp(alpha_hat + beta_hat * df["log_clubs"])
df["residual"] = df["new_stadiums"] - df["mu_fitted"]
df["ratio_obs_exp"] = df["new_stadiums"] / df["mu_fitted"]

# p-value for each country (two-tailed)
def two_tailed_poisson_p(obs, mu):
    p_lo = stats.poisson.cdf(obs, mu)
    p_hi = 1 - stats.poisson.cdf(obs - 1, mu)
    return 2 * min(p_lo, p_hi)

df["p_value"] = df.apply(lambda r: two_tailed_poisson_p(r["new_stadiums"], r["mu_fitted"]), axis=1)

display_cols = ["country", "new_stadiums", "top_div_clubs", "mu_fitted", "residual", "ratio_obs_exp", "p_value"]
print(df[display_cols].sort_values("p_value").to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ── 6. BONFERRONI CORRECTION ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BONFERRONI-CORRECTED SIGNIFICANCE (alpha = 0.05 / N)")
print("=" * 60)
alpha_bonf = 0.05 / N
print(f"Bonferroni threshold: {alpha_bonf:.4f}")
outliers = df[df["p_value"] < alpha_bonf].sort_values("p_value")
print(f"Significant outliers: {len(outliers)}")
print(outliers[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

italy_p = df[df["country"] == "Italy"]["p_value"].values[0]
print(f"\nItaly p-value = {italy_p:.4f} | Bonferroni threshold = {alpha_bonf:.4f}")
print(f"Italy IS {'a significant' if italy_p < alpha_bonf else 'NOT a significant'} outlier after Bonferroni correction")

# ── 7. SUMMARY ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
Italy observed new stadiums  : {italy_obs}
Italy top-division clubs     : {italy_clubs}

Model 1 (naive equal prob)
  Expected under H0          : {lam_naive:.2f}
  Two-tailed p-value         : {p_naive:.4f}
  Conclusion                 : {conclusion_naive}

Model 2 (Poisson GLM, normalized by club count)
  Expected under H0          : {mu_italy:.2f}
  Two-tailed p-value         : {p_glm:.4f}
  Conclusion                 : {conclusion_glm}
  After Bonferroni correction: Italy IS {'a significant' if italy_p < alpha_bonf else 'NOT a significant'} outlier

The original claim that Italy 'stands out' is therefore {'SUPPORTED' if p_glm < 0.05 else 'NOT SUPPORTED'} by the data.
""")
