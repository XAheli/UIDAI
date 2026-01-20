# Statistical Inferences from UIDAI Aadhaar Data Analysis

Generated: 2026-01-20 20:11:04.525163

Total Records Analyzed: 4,345,469

---


### Temporal Analysis - Biometric Dataset

**Day of Week Patterns:**
- Peak enrollment day: **Tuesday** (mean = 75.99 per record)
- Lowest enrollment day: **Wednesday** (mean = 15.26 per record)
- Difference: 397.9% higher on Tuesday

**Statistical Significance:**
- Kruskal-Wallis H-test: H = 18448.03, p = 0.00e+00
- Interpretation: Significant differences exist between days (p < 0.05)

**Weekend vs Weekday:**
- Weekend mean: 47.20
- Weekday mean: 35.55
- t-statistic: 41.16, p = 0.00e+00
- Inference: Weekend enrollments are significantly higher, possibly due to working population availability

---

### Geographic Analysis - Biometric Dataset

**Regional Distribution:**
- Central: 26.0% of total
- South: 20.7% of total
- West: 18.0% of total
- East: 17.1% of total
- North: 15.4% of total
- Northeast: 2.8% of total

**Top 5 States:**
- Uttar Pradesh: 9,367,083 (13.7%)
- Maharashtra: 9,020,710 (13.2%)
- Madhya Pradesh: 5,819,736 (8.5%)
- Bihar: 4,778,968 (7.0%)
- Tamil Nadu: 4,572,152 (6.7%)

**Inequality Metrics:**
- Gini Coefficient: **0.654** (High inequality)
- Top 5 states account for **49.2%** of all enrollments
- Top 10 states account for **72.4%** of all enrollments

**Statistical Test:**
- Regional Kruskal-Wallis: H = 109418.97, p = 0.00e+00
- Interpretation: Significant regional differences exist

**Key Insight:** The high Gini coefficient indicates concentrated enrollment in a few populous states, suggesting need for targeted outreach in smaller states.

---

### Socioeconomic Analysis - Biometric Dataset

**HDI Correlation:**
- Pearson r = -0.365
- p-value = 5.12e-02
- Interpretation: Negative correlation - higher enrollment in lower HDI states indicates successful penetration in underdeveloped regions

**Literacy Rate Correlation:**
- Pearson r = -0.338

**HDI Category ANOVA:**
- F-statistic = 1429.76
- p-value = 0.00e+00

**Policy Implication:** The inverse relationship between HDI and enrollment volume suggests that Aadhaar enrollment drives have successfully targeted less-developed states, contributing to financial inclusion goals.

---

### Climate Analysis - Biometric Dataset

**Rainfall Zone Distribution:**
- High: 8.8% (mean: 22.26)
- Low: 11.4% (mean: 46.98)
- Moderate: 70.1% (mean: 43.25)
- Very High: 3.9% (mean: 18.36)
- Very Low: 5.8% (mean: 51.21)

**ANOVA Results:**
- F-statistic = 1629.94
- p-value = 0.00e+00

**Interpretation:** No significant climate-based differences

---

### Temporal Analysis - Demographic Dataset

**Day of Week Patterns:**
- Peak enrollment day: **Saturday** (mean = 48.84 per record)
- Lowest enrollment day: **Monday** (mean = 15.54 per record)
- Difference: 214.3% higher on Saturday

**Statistical Significance:**
- Kruskal-Wallis H-test: H = 16314.98, p = 0.00e+00
- Interpretation: Significant differences exist between days (p < 0.05)

**Weekend vs Weekday:**
- Weekend mean: 35.59
- Weekday mean: 18.70
- t-statistic: 71.27, p = 0.00e+00
- Inference: Weekend enrollments are significantly higher, possibly due to working population availability

---

### Geographic Analysis - Demographic Dataset

**Regional Distribution:**
- Central: 27.3% of total
- East: 23.0% of total
- South: 17.5% of total
- North: 14.5% of total
- West: 14.3% of total
- Northeast: 3.4% of total

**Top 5 States:**
- Uttar Pradesh: 6,460,511 (17.7%)
- Maharashtra: 3,824,891 (10.5%)
- Bihar: 3,638,841 (9.9%)
- West Bengal: 2,844,316 (7.8%)
- Madhya Pradesh: 2,104,635 (5.8%)

**Inequality Metrics:**
- Gini Coefficient: **0.707** (High inequality)
- Top 5 states account for **51.6%** of all enrollments
- Top 10 states account for **73.9%** of all enrollments

**Statistical Test:**
- Regional Kruskal-Wallis: H = 131524.04, p = 0.00e+00
- Interpretation: Significant regional differences exist

**Key Insight:** The high Gini coefficient indicates concentrated enrollment in a few populous states, suggesting need for targeted outreach in smaller states.

---

### Socioeconomic Analysis - Demographic Dataset

**HDI Correlation:**
- Pearson r = -0.451
- p-value = 1.40e-02
- Interpretation: Negative correlation - higher enrollment in lower HDI states indicates successful penetration in underdeveloped regions

**Literacy Rate Correlation:**
- Pearson r = -0.406

**HDI Category ANOVA:**
- F-statistic = 1703.44
- p-value = 0.00e+00

**Policy Implication:** The inverse relationship between HDI and enrollment volume suggests that Aadhaar enrollment drives have successfully targeted less-developed states, contributing to financial inclusion goals.

---

### Climate Analysis - Demographic Dataset

**Rainfall Zone Distribution:**
- High: 11.3% (mean: 16.83)
- Low: 10.4% (mean: 26.42)
- Moderate: 68.8% (mean: 25.14)
- Very High: 3.9% (mean: 10.34)
- Very Low: 5.6% (mean: 30.08)

**ANOVA Results:**
- F-statistic = 610.48
- p-value = 0.00e+00

**Interpretation:** No significant climate-based differences

---

### Temporal Analysis - Enrollment Dataset

**Day of Week Patterns:**
- Peak enrollment day: **Tuesday** (mean = 10.04 per record)
- Lowest enrollment day: **Saturday** (mean = 4.01 per record)
- Difference: 150.1% higher on Tuesday

**Statistical Significance:**
- Kruskal-Wallis H-test: H = 1583.61, p = 0.00e+00
- Interpretation: Significant differences exist between days (p < 0.05)

**Weekend vs Weekday:**
- Weekend mean: 4.80
- Weekday mean: 5.62
- t-statistic: -10.90, p = 1.10e-27
- Inference: Weekday enrollments dominate, indicating office-hour-based enrollment centers

---

### Geographic Analysis - Enrollment Dataset

**Regional Distribution:**
- Central: 29.8% of total
- East: 23.2% of total
- South: 14.4% of total
- North: 13.3% of total
- West: 12.0% of total
- Northeast: 7.2% of total

**Top 5 States:**
- Uttar Pradesh: 1,002,631 (18.8%)
- Bihar: 593,753 (11.1%)
- Madhya Pradesh: 487,892 (9.2%)
- West Bengal: 369,242 (6.9%)
- Maharashtra: 363,446 (6.8%)

**Inequality Metrics:**
- Gini Coefficient: **0.664** (High inequality)
- Top 5 states account for **52.8%** of all enrollments
- Top 10 states account for **76.8%** of all enrollments

**Statistical Test:**
- Regional Kruskal-Wallis: H = 90275.90, p = 0.00e+00
- Interpretation: Significant regional differences exist

**Key Insight:** The high Gini coefficient indicates concentrated enrollment in a few populous states, suggesting need for targeted outreach in smaller states.

---

### Socioeconomic Analysis - Enrollment Dataset

**HDI Correlation:**
- Pearson r = -0.534
- p-value = 2.82e-03
- Interpretation: Negative correlation - higher enrollment in lower HDI states indicates successful penetration in underdeveloped regions

**Literacy Rate Correlation:**
- Pearson r = -0.448

**HDI Category ANOVA:**
- F-statistic = 918.23
- p-value = 0.00e+00

**Policy Implication:** The inverse relationship between HDI and enrollment volume suggests that Aadhaar enrollment drives have successfully targeted less-developed states, contributing to financial inclusion goals.

---

### Climate Analysis - Enrollment Dataset

**Rainfall Zone Distribution:**
- High: 10.2% (mean: 3.86)
- Low: 10.1% (mean: 6.16)
- Moderate: 65.6% (mean: 5.59)
- Very High: 7.7% (mean: 5.54)
- Very Low: 6.4% (mean: 6.21)

**ANOVA Results:**
- F-statistic = 109.08
- p-value = 4.14e-93

**Interpretation:** Significant differences exist across climate zones

---
