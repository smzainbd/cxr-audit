> **Paper Reproducibility Note:** To reproduce the published *Intelligence-Based Medicine* (Elsevier, 2026) paper, check out git tag `v1.0-paper`.

# CXR Faithfulness Benchmark

**Do Deep Chest X-Ray Classifiers Explain Where They Look? A Multi-Architecture Faithfulness Benchmark with False-Positive Attribution Audit on VinDr-CXR**

## 📌 Overview

Automated chest radiograph interpretation relies heavily on deep neural networks. To build trust, clinical AI systems routinely visualize their predictions via saliency maps (Explainable AI / XAI) prior to clinical use. However, whether these maps *faithfully* reflect the model's actual reasoning—and whether this faithfulness varies across architectural families—remains poorly characterized.

This repository contains the code and experimental pipeline for a comprehensive benchmarking study that evaluates the attribution faithfulness of three modern architectural families:
1. **Classic CNN** (DenseNet-121)
2. **Modern CNN** (ConvNeXtV2-Tiny)
3. **Vision Transformer** (Swin-Base with LoRA)

Using the 15,000-study **VinDr-CXR** dataset, we utilize Integrated Gradients (IG) and null-controlled false-positive audits to analyze how models behave both when they are correct (True Positives) and when they fail (False Positives / False Negatives).

---

## 🔍 Key Findings

* **Classification vs. Faithfulness**: Higher classification accuracy does not guarantee reliable explanations. While Swin-B+LoRA ranked highest in both classification macro-AUC (0.808) and primary faithfulness metrics (precision-mass=0.153; recall top-50%=0.591), absolute localization metrics remained generally modest across all models.
* **The "Plausible-Wrong" Phenomenon (Type B False Positives)**: When models make a wrong prediction, their explanations often still highlight clinically relevant areas, making the AI *look* smart while being clinically wrong. Our study showed that False Positive heatmaps concentrated significantly more mass in bounding boxes than an analytic random null baseline (pooled $\Delta=9.8\%$; $p=7.36 \times 10^{-16}$).
* **Attentive False Negatives**: Models frequently missed critical findings even when their internal attention maps successfully localized the anomaly. For instance, among the Swin-B+LoRA false negatives, **51.4% were "attentive"** (including all 15 Cardiomegaly misses). 

> **Conclusion**: Saliency overlays (heatmaps) should not be used as standalone evidence for AI reliability in clinical settings. Higher faithfulness on true positives does not prevent plausible-wrong explanations on false positives nor attentive false negatives.

---

## 🛠️ Repository Structure

* `notebooks/` - Contains the full experimental pipeline (Jupyter Notebooks):
  * **Data Prep & Splits**: `NB00.ipynb`, `NB01.ipynb`
  * **Model Training**: `NB02_model_training.ipynb`
  * **Evaluation**: `NB03_CLASSIFICATION-Sum.ipynb`
  * **XAI Map Generation**: `NB04_ig_generation.ipynb` (Integrated Gradients), `NB05_lime_.ipynb` (LIME)
  * **Faithfulness Metrics**: `NB06.ipynb`, `NB06_1_attribution_metrics.ipynb`
  * **Failure Audits**: `NB07_fp_analysis_fixed.ipynb` (False Positives), `NB09_fn_clinical_audit.ipynb` (False Negatives)
  * **Clinical Stratification & Sanity Checks**: `NB08.ipynb`, `NB10_1weight_randomization_sanity.ipynb`, `NB10_appendix_ablation.ipynb`
* `scripts/` - Automation and utility scripts.
* `convnext_code.py` - Core network definitions and training utilities.
* `config/` - Configuration settings and environment requirements.
* `paper/` - Drafts and literature context for the research submission.

*(Note: Large datasets, model weights, and generated image maps are not included in this repository due to size constraints. The VinDr-CXR dataset is publicly available via PhysioNet/Kaggle).*

---

## 🚀 Environment Setup

This pipeline was executed using Python 3.10 and CUDA 11.8 on Google Colab.
Key dependencies (found in `config/requirements.txt`):
* `torch==2.1.0`
* `torchvision==0.16.0`
* `timm==0.9.12`
* `captum==0.7.0`
* `peft==0.6.2`

---

## 📝 Authors & Acknowledgements
This project was conducted using the de-identified, publicly available VinDr-CXR dataset. Please refer to the full manuscript for detailed ethical disclosures, funding information, and extended methodology.

---

## 📄 Citation
If you use this code or methodology, please cite:
> S.M. Zain, E. Mahamud, Md. Assaduzzaman, N. Fahad, T.H. Liew, "Do deep chest X-ray classifiers explain where they look? A multi-architecture faithfulness benchmark with false-positive attribution audit on VinDr-CXR," *Intelligence-Based Medicine*, 2026. [https://doi.org/10.1016/j.ibmed.2026.100431](https://doi.org/10.1016/j.ibmed.2026.100431)
