# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 18:15:06 2025

@authors: Salvador E. Ayala-Raggi, Manuel Rendon-Marin
"""
# -----------------------------------------
# -  Approximations to Ellipse Permimeter -

import numpy as np
import matplotlib.pyplot as plt
from mpmath import ellipe
from math import pi, sqrt, log

# ---- Exact Perimeter ----
def P_exact(a, b):
    m = 1 - (b**2)/(a**2)
    return 4*a*ellipe(m)

# ---- Fagnano's formula ----
def P_Fag(a, b):
    return pi*(a+b)

# ---- Euler's Formula ----
def P_Eul(a, b):
    return 2*pi*sqrt((a**2+b**2)/2)

# ---- Ramanujan's Formula ----
def P_R1(a, b):
    return pi*(3*(a+b) - sqrt((3*a + b)*(a + 3*b)))

# ---- Koshy's I Formula
def P_koshy1(a, b):

    k = 0.03214 - 0.0734*(b/a)+0.0863*(b/a)**2 - 0.0681*(b/a)**3 + 0.02306*(b/a)**4
    p = log(2)/log(pi/2) + 1 - (b/a)**k
    Q = (a**p + b**p)**(1/p)
    return 4 * Q

# ---- Koshy's II Formula
def P_koshy2(a, b):
    GM = sqrt(a * b)
    AM = (a + b) / 2.0
    ratio = b / a
    k = (2.6071 
         + 1.2243 * ratio 
         - 1.2673 * (ratio**2) 
         + 0.45566 * (ratio**3))
    Q = (sqrt(a**2 + b**2) 
         + (pi/2 - sqrt(2)) 
         * ((GM / AM) ** k) 
         * sqrt(a * b))
    return 4 * Q

# ---- Moscato's partial 1 Formula
def P_Mu(a, b):
    # h de la fórmula clásica
    h = ((a - b)**2) / ((a + b)**2)
    value = pi * (a + b) * (1 + ((44/pi - 11) * h) / (10 + sqrt(4 - 3*h)))
    return value


# ---- Cantrell's formula ----
def P_cantrell(a, b):
    h = ((a-b)/(a+b))**2
    c = (4.0/pi) - (14.0/11.0)
    return pi*(a+b)*(1 + (3*h)/(10+sqrt(4-3*h)) + c*(h**12))

# ---- Ramanujan 2 formula ----
def P_R2(a, b):
    h = ((a-b)/(a+b))**2
    return pi*(a+b)*(1 + (3*h)/(10+sqrt(4-3*h)))


# ---- Proposed R2/1exp formula
def P_R2div1exp(a, b, A=3.62077e-4, B=10.826):   #E=7.103186105e-9
    h = ((a-b)/(a+b))**2
    return P_R2(a,b)/(1-(A*np.exp(-B*(1-h)))*(1/(1+np.exp(-60*(h-0.33)))))

# ---- Proposed R2/2exp formula
def P_R2div2exp(a, b, A=3.37528e-4, B=10.29662, C=6.48093e-5, D = 40.89043):   #E=7.103186105e-9
    h = ((a-b)/(a+b))**2
    return P_R2(a,b)/(1-(A*np.exp(-B*(1-h)) + C*np.exp(-D*(1-h)))*(1/(1+np.exp(-60*(h-0.33)))))

# ---- Proposed R2/3exp formula
def P_R2div3exp(a, b, A=3.27615e-4, B=10.1405, C=6.93567e-5, D = 33.2054, E = 5.36608e-6, F = 120.109):   #E=7.103186105e-9
    h = ((a-b)/(a+b))**2
    return P_R2(a,b)/(1-(A*np.exp(-B*(1-h)) + C*np.exp(-D*(1-h)) + E*np.exp(-F*(1-h)))*(1/(1+np.exp(-60*(h-0.33)))))

# ---- Moscato's partial 2 Formula
def P_MRu(a, b):
    # h de la fórmula clásica
    h = ((a - b)**2) / ((a + b)**2)
    h1 = ((P_R2(a,b) - P_Mu(a,b))**2) / ((P_R2(a,b) + P_Mu(a,b))**2)
    value = P_R2(a,b)+9938*P_Mu(a, b)*(h**7)*h1
    return value

# ---- Moscato's partial 3 Formula
def P_Mp(a, b):
    # Definición de h
    h = ((a - b)**2) / ((a + b)**2)

    # Numerador y denominador de la fracción
    num = 135168 - 85760*h - 5568*(h**2) + 3867*(h**3)
    den = 135168 - 119552*h + 22208*(h**2) - 345*(h**3)

    # Fórmula completa
    value = pi * (a + b) * (num / den)
    return value

# ---- Moscato's Mmc Final Formula
def P_Mmc(a,b):
    h = ((a - b)**2) / ((a + b)**2)
    h1 = ((P_R2(a,b) - P_Mu(a,b))**2) / ((P_R2(a,b) + P_Mu(a,b))**2)
    h2 = ((P_Mp(a,b) - P_Mu(a,b))**2) / ((P_Mp(a,b) + P_Mu(a,b))**2)
    mi = P_Mp(a,b) * (P_Mu(a, b) / P_Mp(a,b)) ** (((h1**2) / 615) **((h1 / h2)/a))   
    value = mi*(1 + (h**18) / (a**2 - a**3))
    return value



# ---- H DOMAIN ---- setting a = 1000 and varying b from 1 to 999.9
a = 1000
rs = np.linspace(0.9999, 0.001, 10000)
Hs = ((a - a*rs)/(a + a*rs))**2

# ----- Evaluating Formulas in whole domain
Ps = np.array([P_exact(a, a*r) for r in rs], dtype=float) #Exact Perimeter
PFag = np.array([P_Fag(a, a*r) for r in rs], dtype=float)
PEul = np.array([P_Eul(a, a*r) for r in rs], dtype=float)
PR1 = np.array([P_R1(a, a*r) for r in rs], dtype=float)
PR2 = np.array([P_R2(a, a*r) for r in rs], dtype=float)
PMu = np.array([P_Mu(a, a*r) for r in rs], dtype=float)
PMRu = np.array([P_MRu(a, a*r) for r in rs], dtype=float)
PMmc = np.array([P_Mmc(a, a*r) for r in rs], dtype=float)
PKoshy1 = np.array([P_koshy1(a, a*r) for r in rs], dtype=float)
PKoshy2 = np.array([P_koshy2(a, a*r) for r in rs], dtype=float)
PCant = np.array([P_cantrell(a, a*r) for r in rs], dtype=float)
PR2e1 = np.array([P_R2div1exp(a, a*r) for r in rs], dtype=float)
PR2e2 = np.array([P_R2div2exp(a, a*r) for r in rs], dtype=float)

# ---- Signed Relative Errors ----
rel_Fag_signed = ((PFag - Ps) / Ps) * 100.0
rel_Eul_signed = ((PEul - Ps) / Ps) * 100.0
rel_R1_signed = ((PR1 - Ps) / Ps) * 100.0
rel_R2_signed = ((PR2 - Ps) / Ps) * 100.0
rel_K1_signed = ((PKoshy1 - Ps) / Ps) * 100.0
rel_K2_signed = ((PKoshy2 - Ps) / Ps) * 100.0
rel_Cant_signed = ((PCant - Ps) / Ps) * 100.0
rel_R2e1_signed = ((PR2e1 - Ps) / Ps) * 100.0
rel_R2e2_signed = ((PR2e2 - Ps) / Ps) * 100.0
rel_Mu_signed = ((PMu - Ps) / Ps) * 100.0
rel_MRu_signed = ((PMRu - Ps) / Ps) * 100.0
rel_Mmc_signed = ((PMmc - Ps) / Ps) * 100.0


# ---- Graph 1 ----
plt.figure(figsize=(8,5))
plt.plot(Hs, rel_K1_signed, label="Koshy 1", color="orange")
plt.plot(Hs, rel_K2_signed, label="Koshy 2", color="green")
plt.plot(Hs, rel_Cant_signed, label="Cantrell", color="red")
plt.plot(Hs, rel_R2e1_signed, label="R2/1 exp", color="lightblue")
plt.plot(Hs, rel_R2e2_signed, label="R2/2 exp", color="blue")
plt.axhline(0, color="k", linestyle="--", lw=1)
plt.xlabel("h = ((a - b)/(a + b))^2   (calculated using a=1000 and b from 999 to 1)")
plt.ylabel("Relative error (%)")
plt.title("Signed relative error comparison of ellipse perimeter approximations")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()


# ------ Printing Maximum relative errors
print("Maximum relative errors (signed values ignored, %) for a=1000 and b \u2208 [1, 999]:")
print(f"  Fagnano: {np.max(np.abs(rel_Fag_signed)):.6e}%/{np.max(np.abs(rel_Fag_signed))*1e4:.2f}ppm")
print(f"  Euler:  {np.max(np.abs(rel_Eul_signed)):.6e}%/{np.max(np.abs(rel_Eul_signed))*1e4:.2f}ppm")
print(f"  Ramanujan 1:  {np.max(np.abs(rel_R1_signed)):.6e}%/{np.max(np.abs(rel_R1_signed))*1e4:.2f}ppm")
print(f"  Ramanujan 2: {np.max(np.abs(rel_R2_signed)):.6e}%/{np.max(np.abs(rel_R2_signed))*1e4:.2f}ppm")
print(f"  Koshy 1:  {np.max(np.abs(rel_K1_signed)):.6e}%/{np.max(np.abs(rel_K1_signed))*1e4:.2f}ppm")
print(f"  Koshy 2:  {np.max(np.abs(rel_K2_signed)):.6e}%/{np.max(np.abs(rel_K2_signed))*1e4:.2f}ppm")
print(f"  Cantrell: {np.max(np.abs(rel_Cant_signed)):.6e}%/{np.max(np.abs(rel_Cant_signed))*1e4:.2f}ppm")
print(f"  R2/1 exp: {np.max(np.abs(rel_R2e1_signed)):.6e}%/{np.max(np.abs(rel_R2e1_signed))*1e4:.2f}ppm")
print(f"  R2/2 exp: {np.max(np.abs(rel_R2e2_signed)):.6e}%/{np.max(np.abs(rel_R2e2_signed))*1e4:.2f}ppm")
print(f"  Moscato: {np.max(np.abs(rel_Mmc_signed)):.6e}%/{np.max(np.abs(rel_Mmc_signed))*1e4:.2f}ppm")


#----------  "a" DOMAIN -------------------------------------------------------------------------------
#---------- setting b = 1  and varying a from 1.0002 to 10000 
a = 1
rs = np.logspace(0.0001, 4, 10000)

# ----- Evaluating Formulas in whole domain
Ps = np.array([P_exact(a*r, 1) for r in rs], dtype=float) #Exact Perimeter
PFag = np.array([P_Fag(a*r, 1) for r in rs], dtype=float)
PEul = np.array([P_Eul(a*r, 1) for r in rs], dtype=float)
PR1 = np.array([P_R1(a*r, 1) for r in rs], dtype=float)
PR2 = np.array([P_R2(a*r, 1) for r in rs], dtype=float)
PMu = np.array([P_Mu(a*r, 1) for r in rs], dtype=float)
PMRu = np.array([P_MRu(a*r, 1) for r in rs], dtype=float)
PMmc = np.array([P_Mmc(a*r, 1) for r in rs], dtype=float)
PKoshy1 = np.array([P_koshy1(a*r, 1) for r in rs], dtype=float)
PKoshy2 = np.array([P_koshy2(a*r, 1) for r in rs], dtype=float)
PCant = np.array([P_cantrell(a*r, 1) for r in rs], dtype=float)
PR2e1 = np.array([P_R2div1exp(a*r, 1) for r in rs], dtype=float)
PR2e2 = np.array([P_R2div2exp(a*r, 1) for r in rs], dtype=float)

# ---- Signed Relative Errors % ----
rel_Fag_signed = ((PFag - Ps) / Ps) * 100.0
rel_Eul_signed = ((PEul - Ps) / Ps) * 100.0
rel_R1_signed = ((PR1 - Ps) / Ps) * 100.0
rel_R2_signed = ((PR2 - Ps) / Ps) * 100.0
rel_K1_signed = ((PKoshy1 - Ps) / Ps) * 100.0
rel_K2_signed = ((PKoshy2 - Ps) / Ps) * 100.0
rel_Cant_signed = ((PCant - Ps) / Ps) * 100.0
rel_R2e1_signed = ((PR2e1 - Ps) / Ps) * 100.0
rel_R2e2_signed = ((PR2e2 - Ps) / Ps) * 100.0
rel_Mu_signed = ((PMu - Ps) / Ps) * 100.0
rel_MRu_signed = ((PMRu - Ps) / Ps) * 100.0
rel_Mmc_signed = ((PMmc - Ps) / Ps) * 100.0


# ---- Graph 2 ----
plt.figure(figsize=(8,5))
plt.semilogx(rs, rel_K1_signed, label="Koshy 1", color="orange", linestyle='--')
plt.semilogx(rs, rel_K2_signed, label="Koshy 2", color="green", linestyle='--')
plt.semilogx(rs, rel_Cant_signed, label="Cantrell", color="red", linestyle='--')
plt.semilogx(rs, rel_Mmc_signed, label="Moscato", color="pink", linestyle='--')
plt.semilogx(rs, rel_R2e1_signed, label="R2/1exp (ours)", color="purple")
plt.semilogx(rs, rel_R2e2_signed, label="R2/2exp (ours)", color="blue")


plt.axhline(0, color="k", linestyle="--", lw=1)
plt.xlabel("a   (with b=1 and a \u2208 [1, 10000]")
plt.ylabel("Relative error (%)")
plt.title("Signed relative error comparison of ellipse perimeter approximations")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# ------ Printing Maximum relative errors
print("Maximum relative errors (signed values ignored, %) for b=1 and a \u2208 [1, 10000]:")
print(f"  Fagnano: {np.max(np.abs(rel_Fag_signed)):.6e}%/{np.max(np.abs(rel_Fag_signed))*1e4:.2f}ppm")
print(f"  Euler:  {np.max(np.abs(rel_Eul_signed)):.6e}%/{np.max(np.abs(rel_Eul_signed))*1e4:.2f}ppm")
print(f"  Ramanujan 1:  {np.max(np.abs(rel_R1_signed)):.6e}%/{np.max(np.abs(rel_R1_signed))*1e4:.2f}ppm")
print(f"  Ramanujan 2: {np.max(np.abs(rel_R2_signed)):.6e}%/{np.max(np.abs(rel_R2_signed))*1e4:.2f}ppm")
print(f"  Koshy 1:  {np.max(np.abs(rel_K1_signed)):.6e}%/{np.max(np.abs(rel_K1_signed))*1e4:.2f}ppm")
print(f"  Koshy 2:  {np.max(np.abs(rel_K2_signed)):.6e}%/{np.max(np.abs(rel_K2_signed))*1e4:.2f}ppm")
print(f"  Cantrell: {np.max(np.abs(rel_Cant_signed)):.6e}%/{np.max(np.abs(rel_Cant_signed))*1e4:.2f}ppm")
print(f"  R2/1 exp: {np.max(np.abs(rel_R2e1_signed)):.6e}%/{np.max(np.abs(rel_R2e1_signed))*1e4:.2f}ppm")
print(f"  R2/2 exp: {np.max(np.abs(rel_R2e2_signed)):.6e}%/{np.max(np.abs(rel_R2e2_signed))*1e4:.2f}ppm")
print(f"  Moscato: {np.max(np.abs(rel_Mmc_signed)):.6e}%/{np.max(np.abs(rel_Mmc_signed))*1e4:.2f}ppm")




# ---- Graph 3 ----

# ---- Absolute Relative Errors  ----
rel_R2 = abs((PR2 - Ps) / Ps)
rel_K1 = abs((PKoshy1 - Ps) / Ps)
rel_K2 = abs((PKoshy2 - Ps) / Ps)
rel_Cant = abs((PCant - Ps) / Ps)
rel_R2e1 = abs((PR2e1 - Ps) / Ps)
rel_R2e2 = abs((PR2e2 - Ps) / Ps)
rel_Mu = abs((PMu - Ps) / Ps)
rel_MRu = abs((PMRu - Ps) / Ps)
rel_Mmc = abs((PMmc - Ps) / Ps)

plt.figure(figsize=(8,5))
plt.loglog(rs, rel_R2, label="Ramanujan 2", color="orange", linestyle='--')
plt.loglog(rs, rel_K1, label="Koshy 1", color="green", linestyle='--')
plt.loglog(rs, rel_Cant, label="Cantrell", color="red", linestyle='--')
plt.loglog(rs, rel_Mmc, label="Moscato", color="pink", linestyle='--')
plt.loglog(rs, rel_R2e1, label="R2/1exp (ours)", color="purple")
plt.loglog(rs, rel_R2e2, label="R2/2exp (ours)", color="blue")

plt.ylim(10e-16, None)   # de 1e-13 hasta el valor máximo automático

plt.axhline(0, color="k", linestyle="--", lw=1)

plt.axvline(x = 3.93087, color='grey', linestyle='--', label='Comet Halley')
plt.axvline(x = 10.1139, color='grey', linestyle=':', label='Comet Hale-Bopp')
plt.axvline(x = 70.7124, color='grey', linestyle='-', label='Comet Ikeya-Seki')


plt.xlabel(" a    (while keeping b=1)")
plt.ylabel("Absolute relative error")
plt.title("Absolute relative error comparison of ellipse perimeter approximations")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()



