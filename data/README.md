# Asthma Exacerbation Risk Prediction System  
**AstraZeneca Data Scientist Case Study**  
*Leveraging Synthetic Patient Data & Real-World Environmental Factors*

[![AWS HealthLake](https://img.shields.io/badge/AWS-HealthLake-FF9900)](https://aws.amazon.com/healthlake/)
[![GEMA 5.0](https://img.shields.io/badge/Clinical-GEMA_5.0-009688)](https://www.gemasma.com/)
[![Symbicort](https://img.shields.io/badge/Therapy-Symbicort®-2196F3)](https://www.astrazeneca.com/our-therapy-areas/respiratory.html)

## 📌 Clinical Foundations
### Patient Simulation (GEMA 5.0 Compliant)
| Parameter               | Clinical Basis                                                                 | Implementation                              |
|-------------------------|-------------------------------------------------------------------------------|---------------------------------------------|
| Severity Distribution   | Spanish Asthma Guidelines ([GEMA 5.0](https://www.gemasma.com/))              | Intermittent:25%, Mild:35%, Moderate:25%   |
| Adherence Model         | REALISE Study ([2017](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5533072/)) | Beta(α,β) per severity level                |
| Comorbidities           | MARCOS Cohort ([2021](https://www.archbronconeumol.org/es-estudio-marcos-articulo-S0300289621002368)) | 75% rhinitis in severe patients |

### Exacerbation Risk Calculation
```python
# Based on TENOR Study equation ([JACI 2004](https://www.jacionline.org/article/S0091-6749(04)01786-X/))
base_risk = severity_risk * (1.15 - symbicort_adherence)
risk = base_risk * (1.15 - adherence)  # TENOR study equation [⁴](#references)
