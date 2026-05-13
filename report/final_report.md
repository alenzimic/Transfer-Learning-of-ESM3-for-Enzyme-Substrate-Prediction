# Transfer learning of ESM3 for enzyme substrate prediction

Nate Smith (nss97), Alen Zimic (amz63), Chase Holdener (ch2228), Darke Hull (drh257)

## Introduction

Enzymes catalyze the chemical reactions that produce the diversity of molecules found in living organisms. BAHD acyltransferases (BAHDs) are a large, promiscuous plant enzyme family whose members catalyze acyl-group transfer across a wide range of chemically distinct acceptor and donor substrates. Predicting which substrate classes a given BAHD acts on is a multi-label classification problem characterized by high class imbalance, limited labeled data, and long input sequences of more than 400 residues.

In prior work, FuncFetch was developed as an LLM-assisted pipeline that screens published manuscripts and extracts structured enzyme-substrate metadata. Using published outputs of this pipeline, curated datasets were compiled for BAHDs (n = 366) and UGT glycosyltransferases (UGTs, n = 482). Each enzyme's experimentally verified substrates were represented as SMILES strings and clustered into categorical superclass labels using NPClassifier. These labels serve as the multi-label prediction targets, with the BAHD dataset used for training and validation and the UGT dataset held out to evaluate cross-family generalizability.

ESM3 is a 98B-parameter masked generative transformer pretrained on 2.78 billion proteins. It jointly tokenizes protein sequence, 3D structure, and functional annotation as parallel discrete tracks, processing them through a shared bidirectional encoder. However, ESM3's function-token vocabulary is coarse enough that it assigns the same label to essentially all BAHDs, collapsing substrate-level diversity. Rather than replicating this coarse native-label result, this project asks whether ESM3's intermediate representations encode finer-grained substrate information that downstream classifiers can recover, and whether the resulting models can yield interpretable, biochemically meaningful signals about which protein regions drive substrate specificity.

## Methodology

Internal ESM3 representations were repurposed as transfer-learning features for a more granular enzyme substrate-prediction task. Two ESM3 intervention points were evaluated: function logits (FL), the per-residue function-track outputs used by ESM3 to predict functional annotations, and last hidden layer (LL) embeddings, the final hidden states from which all track-specific logits are computed. BAHD protein structures were loaded as PDB files and passed through ESM3 inference to generate per-residue FL and LL tensors. These tensors were mean-pooled across sequence length to obtain fixed-length protein representations and used as input features for multi-label classification of acceptor and donor substrates. The UGT dataset was reserved as a held-out cross-family test. Results are the average of 5-fold cross validation unless stated otherwise.

BAHD acceptor micro-AUPR (BAuPR) was selected as the primary optimization metric because micro-averaging ensures rare classes contribute proportionally to the score, and AUPR is more discriminating than AUROC under severe class imbalance. Labels with fewer than two examples were dropped.

A LazyPredict-inspired screening script evaluated initial linear, tree-based, and other models, applying 5-fold cross-validation and calculating BAuPR. The top performing model from screening, a logistic regression, a shallow MLP, and a transformer-based model were then compared with a 15-run random parameter search. The highest performing models for FL and LL were run 30 additional times in a refined range to select final configurations. The two models were compared with 5-fold and leave-one-out cross-validation in terms of micro and macro AUPR and AUROC to estimate generalizability. The higher performing model on average was selected as the final model, retrained on the entire BAHD dataset, and evaluated against the 20 shared acceptors in the UGT dataset.

To assess interpretability, two complementary analyses were performed on the final LL-based logistic-regression model and the underlying unpooled ESM3 representations. First, because mean pooling, standardization, and logistic regression are linear operations, each one-vs-rest prediction score could be exactly decomposed into per-residue and per-feature contributions, identifying which sequence positions most influence a given substrate prediction. Second, to test whether mean pooling obscured localized biological signals, the original per-residue ESM3 hidden-state tensors were probed directly. Residues were grouped into 50 relative-position bins, and each feature-by-bin activation was treated as a univariate classifier for each substrate label, with AUROC used to measure discriminative power and permutation tests used to assess significance. The strongest feature-position signals were visualized with cross-protein heatmaps comparing positive and negative examples, and protein-length distributions across labels were checked to evaluate whether apparent positional signals could be explained by length confounding rather than substrate-specific representation.

## Results and Analysis

The original publication of ESM3 predicted broad InterPro annotations for BAHD and UGT enzymes. This implementation instead predicts meaningful, more granular functions. The most prominent challenge was the limited size of the dataset and the rarity of several labels, which prevented more complex approaches such as fine-tuning the ESM3 encoder without overfitting. More data may have improved the performance of nonlinear models in initial screening. The success of donor prediction and the modest generalizability of acceptor prediction suggest that ESM3 hidden states contain untapped prediction potential.

Interpretability analysis showed that the predictive signal present in ESM3 was partially obscured by mean pooling over the residue dimension of the protein embeddings before modeling. Examining per-feature contributions after mean pooling produced diffuse explanations: no single feature dominated the prediction, which is consistent with the smoothing effect of mean pooling and L2 regularization. However, direct probing of ESM3 hidden states across features and residues revealed feature-specific regions of local residues with substrate-specific signals that classifier-level explanations did not expose.

When individual feature-by-residue-bin activations were evaluated as univariate classifiers, several single feature-by-bin combinations achieved high AUROC for specific acceptor classes, including phenolic acids and naphthalenes. Cross-protein heatmaps showed consistent activation differences between positive and negative enzymes. Visualizing these informative residue regions on folded proteins can identify candidate protein regions important for enzyme activity. This suggests that ESM3 encodes information relevant to substrate specificity in spatially localized regions. Protein-length comparisons did not show a consistent label-wide separation, reducing the likelihood that these localized signals were explained solely by length confounding, although alignment-aware structural validation remains a future step.

## Reflections

This project showed that ESM3 can be repurposed for a more granular biochemical task than the original objective of its function track. Although ESM3 assigned overly broad functional labels to enzymes, its hidden states contained substrate-specific information that simple classifiers could recover. The main limitation was the size and imbalance of the curated dataset: several labels had only a few positive examples, making 5-fold cross-validation estimates unstable and limiting rare-class performance. This also constrained model complexity, since MLPs, transformer heads, or full ESM3 fine-tuning were more likely to overfit than improve generalization. These limitations are common in biological datasets, and lightweight classifiers were well-suited to this task.

Mean pooling was sample-efficient and predictive, but it obscured localized residue-level signals, while direct probing of unpooled ESM3 tensors revealed stronger position-specific substrate information. Future work should expand the dataset across additional enzyme families, use structure- or alignment-aware residue binning, and evaluate position-aware architectures such as attention pooling, regional probes, or adapter-based fine-tuning. Overall, the results suggest that protein foundation models contain latent substrate-specific information, but recovering it requires task-specific feature extraction rather than relying only on their native output tracks.

This work establishes a baseline for substrate-class prediction from frozen ESM3 representations, but more intricate modifications to the model could sharpen the signal. Fine-tuning the ESM3 encoder on substrate-labeled data, for example through adapter layers or low-rank updates, may evoke clearer substrate-specificity features in the hidden states than the frozen representations used here. On the interpretability side, mapping the high-signal residue positions identified by the probing analysis onto known catalytic motifs and conserved domains would strengthen the evidence that ESM3 has learned biochemically relevant features rather than spurious correlations. Future work could also explore the effect of alternative residue segmentation strategies, such as single-residue resolution or k-mer-based partitions, on both prediction accuracy and interpretability.

## Figures

Fig. 1: General BAHD acyltransferase reaction scheme. A CoA-thioester donor transfers its acyl group to a hydroxyl- or amine-bearing acceptor, forming an ester or amide product. Variable chemical scaffolds define the substrate classes predicted in this study.

Fig. 2: Distribution of ESM3 function tokens predicted across all BAHD residues in the dataset. Nearly all BAHDs receive the same small set of tokens, illustrating the coarseness of ESM3's native functional vocabulary.

Fig. 3: Overview of the project pipeline. BAHD protein structures are passed through ESM3 inference, producing per-residue function logits and last hidden layer embeddings. These are mean-pooled into fixed-length vectors and used as input features for binary substrate classifiers. The interpretability pathway decomposes predictions back to per-residue contributions.

Fig. 4: Initial model screening via 15-run random hyperparameter search on BAHD acceptor prediction. Each point represents one 5-fold cross-validation run; the y-axis is BAuPR. FL and LL feature sets are shown separately.

Fig. 5: Performance of the final logistic regression model. Metrics include micro and macro AUPR and AUROC for BAHD 5-fold cross-validation and held-out UGT evaluation.

Fig. 6: Analysis of raw ESM3 protein embeddings. ESM3 last-layer embedding feature 543 of 1536 at residue bin 40 of 50 achieves high activation separation between naphthalene acceptors and non-acceptors. Visualization of residue bin 40 for Aeuc_SAT2, Lery_AAT2, and Epla_AAT2-2 shows a shared candidate protein region useful for naphthalene acceptor prediction.

## References

Hayes, T., Rao, R., Akin, H., Sofroniew, N. J., Oktay, D., Lin, Z., Verkuil, R., Tran, V. Q., Deaton, J., Wiggert, M., Badkundri, R., Shafkat, I., Gong, J., Derry, A., Molina, R. S., Thomas, N., Khan, Y. A., Mishra, C., Kim, C., et al. (2025). Simulating 500 million years of evolution with a language model. Science, 387(6736), 850-858. https://doi.org/10.1126/science.ads0018

Kim, H. W., Wang, M., Leber, C. A., Nothias, L.-F., Reher, R., Kang, K. B., van der Hooft, J. J. J., Dorrestein, P. C., Gerwick, W. H., & Cottrell, G. W. (2021). NPClassifier: A deep neural network-based structural classification tool for natural products. Journal of Natural Products, 84(11), 2795-2807. https://doi.org/10.1021/acs.jnatprod.1c00399

Kruse, L. H., Weigle, A. T., Irfan, M., Martinez-Gomez, J., Chobirko, J. D., Schaffer, J. E., Bennett, A. A., Specht, C. D., Jez, J. M., Shukla, D., & Moghe, G. D. (2022). Orthology-based analysis helps map evolutionary diversification and predict substrate class use of BAHD acyltransferases. The Plant Journal, 111(5), 1453-1468. https://doi.org/10.1111/tpj.15902

Pandala, S. R. (2019). LazyPredict (Version 0.3.0). https://github.com/shankarpandala/lazypredict

Smith, N., Yuan, X., Melissinos, C., & Moghe, G. D. (2025). FuncFetch: An LLM-assisted workflow enables mining thousands of enzyme-substrate interactions from published manuscripts. Bioinformatics, 41(1), btae756. https://doi.org/10.1093/bioinformatics/btae756

## AI Disclosure

Generative AI was used in this assignment as a tool for analysis, coding, syntax, grammar, and debugging.

## Definitions

Enzyme: A protein that catalyzes or accelerates a specific chemical reaction, acting on one or more substrates and converting them into products.

Substrate: The input molecule an enzyme acts on. In acyltransferase reactions there are two substrates: an acceptor, the molecule being modified, and a donor, the molecule supplying the transferred chemical group, typically a CoA thioester.

BAHD acyltransferases: A large plant enzyme family that transfers acyl groups from donor molecules to diverse acceptor scaffolds. The name comes from the initials of the first four characterized members.

UGT glycosyltransferases: A large enzyme family that attaches sugar groups to acceptor molecules, modifying their solubility, stability, and biological activity. UGTs are used here as a held-out cross-family test set.

CoA: Coenzyme A, a carrier molecule that activates acyl groups for transfer. In BAHD reactions, the donor substrate is typically a CoA thioester, such as acetyl-CoA or coumaroyl-CoA.

SMILES string: A text-based representation of a molecular structure, such as CC(=O)O for acetic acid. SMILES strings are used here to encode substrate identity before clustering into categorical labels.
