# Implementation and Evaluation of TeCoA for Adversarial Robustness in CLIP

An empirical study of adversarial robustness in CLIP's zero-shot image classification, evaluating a transfer-based PGD attack and comparing three parameter-efficient defense strategies: standard adversarial fine-tuning, text-guided contrastive adversarial training (TeCoA), and visual prompt tuning.

**Institution:** University of Tehran, School of Electrical and Computer Engineering
**Course:** Neural Networks and Deep Learning
**Author:** Babak Hosseini Mohtasham
**Full report:** [`CA5_Q2_report.pdf`](./CA5_Q2_report.pdf)
**Assignment specification:** [`NNDL_HW5.pdf`](./NNDL_HW5.pdf)
**Reference papers:** [`1298_understanding_zero_shot_advers.pdf`](./1298_understanding_zero_shot_advers.pdf), [`1912_lora_low_rank_adaptation_of_la.pdf`](./1912_lora_low_rank_adaptation_of_la.pdf)

---

## Overview

CLIP performs image classification "zero-shot" by matching image embeddings against text embeddings of class prompts (e.g. *"a photo of a cat"*), rather than through a trained classification head. This project examines whether that zero-shot mechanism is robust to adversarial perturbations, and whether lightweight, parameter-efficient fine-tuning can defend it without discarding CLIP's pretrained representations.

**Objectives:**

1. Measure CLIP's zero-shot classification accuracy on CIFAR-10 under clean conditions.
2. Evaluate how a PGD adversarial attack, crafted against a separate surrogate classifier, transfers to and degrades CLIP's zero-shot predictions.
3. Fine-tune CLIP with LoRA under a standard cross-entropy adversarial training objective and assess robustness gained.
4. Fine-tune CLIP with LoRA under TeCoA, a contrastive loss aligned with CLIP's native image–text training objective, and compare it to the cross-entropy approach.
5. Test whether adjusting the TeCoA temperature affects the robustness/accuracy trade-off.
6. Test Visual Prompt Tuning — learning a single additive perturbation bias to the input image while keeping CLIP fully frozen — as a minimal-parameter alternative defense.

## Methodology

| Stage | Description |
|---|---|
| **Base model** | `openai/clip-vit-base-patch32` used for zero-shot classification on CIFAR-10, via cosine similarity between image embeddings and text embeddings of the prompt `"a photo of a {class}"` for each of the 10 classes |
| **Attack** | PGD (ε = 8/255, α = 2/255, 7 steps, random start) generated against a separately pretrained `cifar10_resnet20` surrogate classifier, then transferred to CLIP — a black-box, transfer-attack setting rather than a white-box attack on CLIP directly |
| **Defense 1 — LoRA + cross-entropy** | CLIP's `q_proj`/`v_proj` attention projections fine-tuned with LoRA (r=8, α=32) under a standard cross-entropy adversarial training loss over image–text similarity logits |
| **Defense 2 — LoRA + TeCoA** | Same LoRA configuration, trained instead with a text-guided contrastive loss (temperature-scaled similarity, cross-entropy against the diagonal/matching pairs) that better matches CLIP's original contrastive pretraining objective |
| **Defense 3 — Temperature ablation** | TeCoA re-run with a higher temperature (0.1 vs. 0.01) to study its effect on the accuracy/robustness trade-off |
| **Defense 4 — Visual Prompt Tuning (VPT)** | CLIP kept entirely frozen; a single learned additive bias image (`3×224×224` parameters) is added to every input and trained with the TeCoA loss — the most parameter-efficient of the four defenses |

All models were evaluated with accuracy, precision, recall, and F1 (micro/macro/weighted and per-class), on both clean and adversarially perturbed CIFAR-10 test data.

## Repository Structure

| File | Description |
|---|---|
| [`NNDL_CA5_2.ipynb`](./NNDL_CA5_2.ipynb) | Complete implementation: dataset preparation, zero-shot CLIP evaluation, PGD attack setup, all four defense methods, and final test-set comparison. All cells were executed and outputs preserved. |
| [`CA5_Q2_report.pdf`](./CA5_Q2_report.pdf) | Written report: theoretical background, methodology, full result tables, and discussion |
| [`NNDL_HW5.pdf`](./NNDL_HW5.pdf) | Original assignment specification |
| [`1298_understanding_zero_shot_advers.pdf`](./1298_understanding_zero_shot_advers.pdf) | Reference paper on zero-shot adversarial robustness, underlying the TeCoA defense |
| [`1912_lora_low_rank_adaptation_of_la.pdf`](./1912_lora_low_rank_adaptation_of_la.pdf) | Reference paper on LoRA, the parameter-efficient fine-tuning method used throughout |

### Notebook outline

1. Prepare CIFAR-10 dataset
2. Load the CLIP model and a CIFAR-10 ResNet20 surrogate model for attack generation
3. Evaluate zero-shot CLIP on clean images
4. Evaluate zero-shot CLIP on transferred PGD adversarial examples
5. Adversarial training: standard cross-entropy loss (LoRA)
6. Adversarial training: text-guided contrastive loss — TeCoA (LoRA)
7. Temperature ablation for TeCoA
8. Visual Prompt Tuning
9. Final comparison of all four models on the held-out test set

## Key Results

**Zero-shot CLIP is reasonably strong out of the box, but not immune to transfer attacks.** Clean zero-shot accuracy on CIFAR-10 reached 87.7% (test set), and dropped to 72.3% under the transferred PGD attack — a real but comparatively modest degradation, likely because the attack is crafted against a different (surrogate) model rather than directly against CLIP.

**Final test-set comparison (held-out data, all defenses trained for 10 epochs):**

| Model | Clean Accuracy | Adversarial Accuracy |
|---|---|---|
| CLIP (zero-shot, no defense) | 87.7% | 72.3% |
| LoRA, cross-entropy loss | 91.7% | 82.1% |
| LoRA, TeCoA loss | **94.9%** | **90.5%** |
| Visual Prompt Tuning, TeCoA loss | 88.7% | 74.2% |

**TeCoA is the clear winner among the defenses tested.** Fine-tuning with the contrastive TeCoA loss not only recovered the most robustness (90.5% adversarial accuracy, vs. 82.1% for plain cross-entropy adversarial training) but also improved clean accuracy the most — likely because its objective stays consistent with CLIP's original image–text contrastive pretraining, rather than repurposing it as a conventional classifier.

**Visual Prompt Tuning trades effectiveness for efficiency.** With only a single learned bias image and the entire CLIP backbone frozen, VPT achieved far fewer trainable parameters than LoRA fine-tuning, but its robustness gains were correspondingly modest (74.2% vs. 90.5% for LoRA+TeCoA) — a clear illustration of the capacity/robustness trade-off when defenses are constrained to touch only the input rather than the model's internal representations.

**Robustness gains generally came with an accuracy improvement, not just a trade-off**, in contrast to typical adversarial training behavior on standard classifiers — plausibly because LoRA fine-tuning also lets the model adapt its embeddings to the CIFAR-10 distribution specifically, on top of any robustness benefit.

## Reproducing the Results

The notebook is self-contained and was developed on Google Colab.

1. Open [`NNDL_CA5_2.ipynb`](./NNDL_CA5_2.ipynb) in Google Colab or Jupyter.
2. Run all cells from top to bottom. Key dependencies (`transformers`, `torchattacks`, `peft`) are installed within the notebook.
3. A GPU runtime is strongly recommended — the notebook fine-tunes four CLIP variants (three with LoRA/VPT training, evaluated both on a validation split and again on the held-out test set).
4. All outputs (training curves, evaluation tables, comparison plots) are already preserved in the notebook and can be reviewed directly on GitHub without re-execution.

## Notes on Scope

- The adversarial attack is a **transfer attack**: PGD perturbations are generated against a separate CIFAR-10 ResNet20 model and then applied to CLIP, rather than computed directly on CLIP's own gradients. This reflects a more realistic black-box threat model but means reported robustness numbers are not directly comparable to white-box attacks on CLIP itself.
- Experiments use a fixed random subsample of 10,000 CIFAR-10 training images (rather than the full 50,000) for the exploratory/validation runs; the final comparison table (above) is computed on the full, separate CIFAR-10 test set.
- LoRA fine-tuning is restricted to the `q_proj` and `v_proj` attention projections of the CLIP vision encoder, keeping the number of trainable parameters small relative to full fine-tuning.
