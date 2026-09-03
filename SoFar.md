# Project Journey: 3-Way Voice Deepfake Detection Fusion

This document serves as a historical record of the empirical trials, experiments, and conclusions reached while building our 3-expert Voice Deepfake Detection system. We relied strictly on empirical data (Error Equal Rate / Error Rates) on held-out slices to drive every architectural decision.

## 1. The Goal
We originally started with two separate backends (WavLM-based and LFCC-LCNN based). Our goal was to fuse them to eliminate their respective blind spots:
- **LFCC-LCNN (Hybrid)** is extremely robust against unseen generators and Hindi telephony data.
- **WavLM** is highly capable on high-quality English and catches synthetic voice changers (pitch shifting) that occasionally fool LFCC.
- **TakHemlata SSL** was later added as a specialized third expert to act as a highly confident arbiter for English ASVspoof edge-cases.

---

## 2. Trial 1: Baseline 3-Way Fusion Evaluation
To properly fuse these three models, we first evaluated their standalone capabilities against two fusion methods:
1. **LR Fusion**: A Logistic Regression model fit on 5,378 Dev set files.
2. **Fixed Heuristic**: A simple rule summing the logits `LFCC + SSL > 0` (ignoring WavLM).

### Baseline Results Table
| Slice Name | WavLM | LFCC | SSL | LR Fusion | Fixed (LFCC+SSL) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Eval: In-Domain** | 13.18% | 40.81% | 2.24% | 2.37% | 2.87% |
| **Eval: Held-Out Gen** | 31.10% | 3.01% | 14.08% | 7.70% | **2.35%** |
| **Hard Negative: Synth** | 26.88% | 24.38% | 0.00% | 0.00% | 0.00% |
| **Hard Negative: Real** | 41.86% | 30.23% | 16.28% | 0.00% | **2.33%** |

### The Conclusion from Trial 1:
- **WavLM is the weakest standalone model** on massive held-out slices, struggling particularly with unseen generators (31.10% error).
- **The Fixed Heuristic (LFCC+SSL) dominated LR Fusion.** By purely looking at the two strongest models, it achieved a staggering 2.35% error rate on Unseen Generators (compared to LR's 7.70%).

**Decision:** We initially favored the LFCC+SSL heuristic because the Dev-fit Logistic Regression failed to generalize to completely unseen deepfakes.

---

## 3. Trial 2: The Real-World Smoke Test Failure
We deployed the LFCC+SSL heuristic against a small folder of completely uncurated, real-world WhatsApp audio and YouTube rips (`testdata/`).

### The Problem:
- The `LFCC+SSL > 0` heuristic had a massive **False Positive** problem in noisy, WhatsApp-compressed reality. It was too paranoid, flagging almost all real humans as fake. 
- The LR Fusion had a massive **False Negative** problem, letting deepfakes slip through.

**Decision:** We needed to rigorously calibrate the threshold on a strictly split real-world dataset, and we needed to mathematically prove if WavLM was completely useless before throwing it away.

---

## 4. Trial 3: Complementarity & Overlap Analysis
If `LFCC+SSL` is the best primary rule, does WavLM have *any* value? We ran an error-overlap script to find out.

### Overlap Results Table
| Slice Name | Total LFCC+SSL Errors | Errors independently caught by WavLM | Percentage |
| :--- | :--- | :--- | :--- |
| **Eval: In-Domain** | 56 errors | 49 right | **87.5%** |
| **Eval: Held-Out Gen** | 287 errors | 27 right | **9.4%** |
| **Hard Negative: Real** | 1 error | 1 right | **100.0%** |

### The Conclusion from Trial 3:
WavLM **is not useless.** When LFCC and SSL both hallucinate and fail, WavLM acts as a critical safety net, catching up to 87.5% of their mistakes on standard in-domain data. It contains unique, complementary knowledge that must be preserved.

---

## 5. Trial 4: The Escalation OR-Rule vs Tuned Heuristic
To capture WavLM's value, we tested an "Escalation OR-Rule": `if (WavLM > Tw) OR (LFCC > Tl) OR (SSL > Ts) then Spoof`. We set the thresholds to target 98% precision on the Dev set.

Simultaneously, we split 20 real-world files 50/50 into a `Calibration` and `Final Test` set to find the optimal real-world threshold `T` for `LFCC+SSL > T`. The optimal threshold was found to be `-5.0`.

### Final Confrontation Results Table
| Slice Name | LR Fusion | LFCC+SSL (T=0) | Tuned LFCC+SSL (T=-5) | Escalation OR-Rule |
| :--- | :--- | :--- | :--- | :--- |
| **Eval: In-Domain** | 2.37% | 2.87% | 2.87% | 7.00% |
| **Eval: Held-Out Gen** | 7.70% | 2.35% | **2.35%** | 41.50% |
| **Hard Negative: Synth** | 0.00% | 0.00% | 0.00% | 64.38% |
| **Hard Negative: Real** | 23.26% | 2.33% | **0.00%** | 53.49% |
| **Realworld Calibration**| 54.76% | 23.81% | **23.81%** | 60.00% |
| **Realworld Final Test** | 7.14% | 7.14% | **7.14%** | 50.00% |

### The Conclusion from Trial 4:
- **The Escalation OR-Rule was a disaster.** Voice deepfake detection is far too noisy for hard OR-rules; if one model hallucinates due to background noise, the entire clip fails (resulting in a 64% error rate on synthetic data).
- **Tuned LFCC+SSL remains the king of unseen data.** It achieved 0.00% error on Hard Negative Reals and 2.35% on Unseen Generators, vastly outperforming LR Fusion.

---

## 6. Trial 5: Mathematical Ensembles vs UI Confidence
We knew `Tuned LFCC+SSL` was the best primary decision boundary, but we still needed to utilize WavLM's complementarity (from Trial 3). 
We tested two mathematical ensembles:
1. `(LR_Fusion_Probability + Sigmoid(LFCC + SSL)) / 2`
2. `LFCC+SSL + 0.5 * WavLM_Logit` (Tie-breaker)

### Ensemble Scratch Test Results
| Slice Name | Tuned LFCC+SSL | Sigmoid Avg | WavLM Tie-Break |
| :--- | :--- | :--- | :--- |
| **Eval: In-Domain** | 2.87% | **2.30%** | 2.84% |
| **Eval: Held-Out Gen** | **2.35%** | 3.35% | 2.80% |

### The Conclusion from Trial 5:
Because WavLM performs poorly on unseen generators (31.10% baseline error), allowing its math to bleed into the primary decision boundary drages the overall average down on unseen deepfakes. We **cannot** mathematically combine them without sacrificing real-world robustness.

---

## 7. The Final Production Architecture
Based on these exhaustive empirical trials, we deployed the following system:

**1. The Primary Decision Engine:** 
The backend relies strictly on the `Tuned LFCC+SSL` mathematical heuristic (shifted by +5.0 so `fused_logit > 0` is the threshold). This provides the most robust, lowest-error decision boundary for unseen deepfakes.

**2. The UI Confidence Metric:** 
We do not throw WavLM's complementary knowledge away, nor do we let it corrupt the primary math. Instead, we use it in the Frontend UI. If the `LFCC+SSL` decision disagrees with WavLM's independent probability, the UI flags the result as an **"Expert Disagreement / Low Confidence"** prediction. This allows human moderators to leverage WavLM's safety net manually when reviewing tricky files!
