# Exponential Corrections to Ramanujan’s Second Formula for the Ellipse Perimeter: ultra-accurate closed-form formulas

Authors:  Salvador E. Ayala-Raggi and Manuel Rendón-Marín

This repository contains the code and extended data for the paper:

**An Exponential Correction to Ramanujan’s Second Formula for Ellipse Perimeter Computation**
Ayala-Raggi, S.E., Rendón-Marín, M., MDPI AppliedMath (2026)
**Link to the paper:** https://www.mdpi.com/2673-9909/6/4/56

As an addendum to the aforementioned paper, we present a new set of formulas for calculating the ellipse perimeter with a maximum relative error as low as 0.02 ppm. These additional findings are described in our most recent paper (preprint):

**Exponential Corrections to Ramanujan’s Second Formula for the Ellipse Perimeter: Ultra-Accurate Closed-Form Approximations**

Ayala-Raggi, S.E., Rendón-Marín, M. (2026)

**Full Document in ResearchGate (NEW):** https://www.researchgate.net/publication/403504375_Exponential_Corrections_to_Ramanujan's_Second_Formula_for_the_Ellipse_Perimeter_A_set_of_ultra-accurate_closed-form_formulas

**Permanent Document in Zenodo (Zenodo DOI):**  https://doi.org/10.5281/zenodo.19421410



## Abstract

We investigate exponential corrections to Ramanujan's second formula for the perimeter of an ellipse. By introducing flexible exponents and increasing the number of exponential terms, we obtain a sequence of increasingly accurate approximations. Numerical evidence suggests that the correction admits a structured expansion consisting of a stable low-rate core and a rapidly convergent corrective tail.

---

## 1. Introduction

Ramanujan's second approximation is:

$$
P_{R2} = \pi (a+b)\left(1+\frac{3h}{10+\sqrt{4-3h}}\right),
\quad
h=\left(\frac{a-b}{a+b}\right)^2
$$

Its maximum relative error is:

> **402.34 ppm**

We consider corrections of the form:

$$
P \approx \frac{P_{R2}}{1 - \varepsilon(h)}
$$

---

## 2. First Improvement: Simple R2/2EXP (no powers)

$$
P \approx \frac{P_{R2}}{1 - \left(A e^{-B(1-h)} + C e^{-D(1-h)}\right)}
$$

Constraint imposed:

$$
A + C = S = 4.0233749415669598e-04 \approx 1 - \frac{7 \pi}{22}
$$

### Constants

* A = 3.37528e-04
* C = 6.48093e-05
* B = 1.029662e+01
* D = 4.089043e+01

**Maximum relative error:**

> **0.57 ppm**

---

## 3. Flexible (with powers) R2/F2EXP Model 

$$
P \approx \frac{P_{R2}}{
1 - \left(
A e^{-B(1-h)^q} + C e^{-D(1-h)^r}
\right)
}
$$
Constraint imposed:
$$
A + C = S = 4.0233749415669598e-04 \approx 1 - \frac{7 \pi}{22}
$$

### Constants

* A = 1.6242914264570106e-04
* C = 2.3990835151099492e-04
* B = 2.1127747251579692e+01
* D = 1.0519954055960756e+01
* q = 9.3255508111834906e-01
* r = 1.1169090428284505e+00

**Maximum relative error::**

> **0.20 ppm**

---

## 4. Flexible (with powers) Three-Exponential Model (R2/F3EXP)

$$
P \approx \frac{P_{R2}}{
1 - \left(
Ae^{-B(1-h)^q}+Ce^{-D(1-h)^r}+ Ee^{-F(1-h)^s}
  \right)}
$$
Constraint imposed:
$$
A + C + E = S = 4.0233749415669598e-04 \approx 1 - \frac{7 \pi}{22}
$$

### Constants

* A = 3.2704699236212363e-04
* C = 3.5335850957003023e-05
* E = 3.9954650837569342e-05
* B = 1.2891219860454603e+01
* D = 4.2799551357558165e+01
* F = 1.1430650693589627e+01
* q = 9.9547975928209875e-01
* r = 9.5254540577122027e-01
* s = 1.7028129039833189e+00

**Maximum relative error:**

> **0.055 ppm**

---

## 5. Flexible (with powers) Four-Exponential Model (R2/F4EXP)

$$
P \approx \frac{P_{R2}}{
1 - \left(
A e^{-B(1-h)^q} + C e^{-D(1-h)^r} + E e^{-F(1-h)^s} + G e^{-H(1-h)^u}
  \right)
  }
$$
Constraint imposed:
$$
A + C + E + G = S = 4.0233749415669598e-04 \approx 1 - \frac{7 \pi}{22}
$$

### Constants

* A = 2.9366973637885561e-04
* C = 6.2926379208842441e-05
* E = 2.9058024718557447e-05
* G = 1.6683353850440461e-05
* B = 1.2794318096403178e+01
* D = 3.3874449738045193e+01
* F = 1.1324362335219835e+01
* H = 5.6819551657952999e+01
* q = 1.0518831146664076e+00
* r = 1.0185882839340432e+00
* s = 1.7946231875051635e+00
* u = 8.9334188171605333e-01

**Maximum relative error:**

> **0.03 ppm**

---

## 6. Flexible (with powers) Five-Exponential Model (R2/F5EXP)

$$
P \approx \frac{P_{R2}}{
1 - \left(
Ae^{-B(1-h)^q} + Ce^{-D(1-h)^r} + Ee^{-F(1-h)^s} + Ge^{-H(1-h)^u} + Ie^{-J(1-h)^v}
  \right)
}
$$
Constraint imposed:
$$
A + C + E + G + I = S = 4.0233749415669598e-04 \approx 1 - \frac{7 \pi}{22}
$$

### Constants

* A = 2.8343296895393181e-04
* C = 7.4224722610190177e-05
* E = 1.7465913309214121e-05
* G = 2.3676211292968272e-05
* I = 3.5376779903916304e-06
* B = 1.2505836332877127e+01
* D = 3.8348802290432275e+01
* F = 1.1947955069812146e+01
* H = 6.9087704651767510e+01
* J = 1.0915293353669674e+02
* q = 1.0876352435861614e+00
* r = 1.0965569005811395e+00
* s = 2.0263941926207321e+00
* u = 9.6710133279772192e-01
* v = 8.4364533457417601e-01

**Maximum relative error:**

> **0.0216 ppm**

---

## 7. Error Evolution

| Model              | Max Error (ppm) |
| ------------------ | --------------- |
| Ramanujan II       | 402.34          |
| R2/2EXP simple     | 0.57            |
| R2/F2EXP           | 0.20            |
| R2/F3EXP           | 0.055           |
| R3/F4EXP           | 0.03            |
| R4/F5EXP           | 0.0216          |

---

## 8. Key Structural Insight

A clear structure emerges:

### Core

* Two dominant exponentials
* decay rates ≈ 11–13
* exponents ≈ 1 and ≈ 2

### Tail

* additional exponentials
* increasing decay rates
* rapidly decreasing amplitudes

$$
\varepsilon(h) =
\varepsilon_{\text{core}}(h)
+
\varepsilon_{\text{tail}}(h)
$$

---

## 9. Conjectures

### Infinite Expansion

$$
\varepsilon(h) = \sum_{k=1}^{\infty} a_k e^{-b_k(1-h)^{p_k}}
$$

### Sum Constraint

$$
\sum_{k=1}^{\infty} a_k = S
$$

### Structure

* stable 2-term core
* rapidly convergent tail

### Error convergence

402.34 → 0.57 → 0.20 → 0.055 → 0.03 → 0.0216 ppm

---

## 10. Discussion

To the best of our knowledge, no previous compact closed-form approximation in this style has shown such a systematic reduction of the maximum relative error from the classical Ramanujan II level down to the **0.02 ppm range**, while preserving a single-line analytical structure.

---

## 11. Conclusion

The exponential correction appears to form a structured and rapidly convergent series, capable of driving the approximation error arbitrarily close to zero with only a few terms.

---

## Notes

* All formulas are **closed-form**
* Based on **minimax optimization**
* Valid over full eccentricity range
* Suitable for engineering applications
