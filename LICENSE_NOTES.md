# License and Data Provenance Notes

## 1. PhysioNet Credentialed Health Data License 1.5.0
The VinDr-CXR dataset was accessed under the PhysioNet Credentialed Health Data License 1.5.0:
> "The LICENSEE will not share access to PhysioNet restricted data with anyone else...
> The LICENSEE will use the data for the sole purpose of lawful use in scientific research and no other."

## 2. Dataset Publication Usage Notes & Legal Status
The VinDr-CXR dataset release paper states:
> "To download the dataset, users are required to accept a Data Usage Agreement (DUA) called 
> PhysioNet Credentialed Health Data License 1.5.0... We also encourage such authors to release 
> their code and models, which will help the community to reproduce experiments and to boost 
> the research in the field of medical imaging."
> — Nguyen et al. (Scientific Data 2022 / arXiv:2012.15029v3, 20 Mar 2022)

**Legal Clarification:** Academic encouragement by dataset authors in a descriptive paper is not a 
legally binding license grant overriding PhysioNet License 1.5.0. To respect the primary DUA, 
all trained checkpoint weights (`*.pt`) remain private in secure Hugging Face storage and are never 
distributed publicly in this repository. Public evaluation outputs contain strictly aggregate statistical 
metrics with zero patient radiographs or raw saliency maps.

## 3. Kaggle VinBigData Chest X-ray Abnormalities Detection Competition Rules
Retrieved: 2026-11-03
> "Open Source
> DATA ACCESS AND USE: Non-commercial, Research/Academic Use Only"

## 4. Demo Radiographs (NIH ChestX-ray14)
All visual demo images (`assets/sample_cxr/`) are sourced exclusively from the NIH ChestX-ray14 dataset 
(Wang et al., CVPR 2017), provided under the Creative Commons CC0 1.0 Universal Public Domain Dedication. 
Zero VinDr-CXR radiographs are hosted in this repository or the public web demo.
