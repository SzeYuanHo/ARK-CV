## HIBOU: A FAMILY OF FOUNDATIONAL VISION TRANSFORMERS FOR PATHOLOGY

Authors: Dmitry Nechaev * 1 , Alexey Pchelnikov †1 , and Ekaterina Ivanova ‡1 1 HistAI

## ABSTRACT

Pathology, the microscopic examination of diseased tissue, is critical for diagnosing various medical conditions, particularly cancers. Traditional methods are labor-intensive and prone to human error. Digital pathology, which converts glass slides into high-resolution digital images for analysis by computer algorithms, revolutionizes the field by enhancing diagnostic accuracy, consistency, and efficiency through automated image analysis and large-scale data processing. Foundational transformer pretraining is crucial for developing robust, generalizable models as it enables learning from vast amounts of unannotated data.

This paper introduces the Hibou family of foundational vision transformers for pathology, leveraging the DINOv2 framework to pretrain two model variants, Hibou-B and Hibou-L, on a proprietary dataset of over 1 million whole slide images (WSIs) representing diverse tissue types and staining techniques. Our pretrained models demonstrate superior performance on both patch-level and slide-level benchmarks, surpassing existing state-of-the-art methods. Notably, Hibou-L achieves the highest average accuracy across multiple benchmark datasets. To support further research and application in the field, we have open-sourced the Hibou-B model, which can be accessed at https://github.com/HistAI/hibou.

## 1 Introduction

Pathology is the study of diseased tissue under a microscope, which plays a crucial role in medical diagnosis by allowing pathologists to examine tissue samples to detect abnormalities and disease conditions. It is the gold standard for diagnosing various conditions, particularly cancers, by identifying cellular abnormalities and changes in tissue. Traditional pathology methods involve staining tissue samples and examining them manually under a microscope. While these methods provide detailed insights, they are time-consuming, subject to human error, and heavily reliant on the expertise of the pathologist. Moreover, manual examination limits the scalability and throughput necessary for high-volume clinical settings.

In recent years, there has been a significant shift from traditional pathology to digital pathology, driven by advancements in imaging technology and computational methods. Digital pathology involves scanning conventional glass slides to produce high-resolution digital images, known as whole slide images (WSIs), which can be analyzed using computer algorithms. This transition enhances diagnostic accuracy and efficiency by enabling the use of advanced computational techniques such as machine learning and artificial intelligence (AI). These technologies facilitate automated image analysis, reducing the subjectivity associated with human interpretation and allowing for consistent and reproducible results [1].

## 2 Related work

One of the most promising advancements in computational methods for image analysis in digital pathology is the adoption of Vision Transformers (ViTs). ViTs have revolutionized the field of computer vision by achieving state-ofthe-art results in various tasks such as image classification, object detection, and segmentation. These models leverage the self-attention mechanism to model long-range dependencies, a fundamental strength over convolutional neural networks (CNNs) which excel at capturing local patterns but struggle with global contexts [2].

Foundational pretraining techniques for ViTs include supervised learning on large annotated datasets, self-supervised learning where the model is trained using an unlabeled dataset, and transfer learning which involves fine-tuning pre-trained models on new tasks [3]. Among these techniques, self-supervised learning stands out as a particularly useful approach since it enables models to learn robust features from unlabeled data, making it valuable in fields like histopathology, where annotated datasets are often limited and costly to produce. By leveraging self-supervised learning, ViTs can be pre-trained on vast amounts of unannotated data, enhancing their ability to generalize and perform well on downstream tasks with limited labeled examples.

Recent works in the field of ViT pretraining for histopathology have predominantly utilized frameworks such as iBot [4] and DINOv2 [5]. The iBot framework is used by the popular open-source model Phikon [6]. As a more recent and advanced framework, DINOv2 has seen adoption in several notable studies, including Virchow, RudolfV, and Prov-Gigapath, among others [7, 8, 9, 10, 11, 12].

In this work, we leverage the DINOv2 framework to pretrain a novel family of vision transformer models, collectively referred to as Hibou. Specifically, we develop two variants: Hibou-B, based on the ViT-B/14 architecture, and Hibou-L, based on the ViT-L/14 architecture. Both models were pretrained on our proprietary histopathology dataset, which comprises over 1 million WSIs representing a diverse array of tissue types and staining techniques (See Figure 1 for an overview of the dataset composition). To promote further research and development, we have made the Hibou-B model publicly available under an Apache 2.0 license. This release is intended to facilitate reproducibility and encourage the application of our pretrained models in various histopathological studies.

## 3 Methodology

## 3.1 Data

We trained our foundation models using proprietary data from what we believe to be the most diverse large dataset collected for AI algorithm development. This dataset comprises 936,441 H&amp;E and 202,464 non-H&amp;E stained slides sourced from 306,400 unique cases. Our training data includes human tissues from various localizations as well as veterinary biopsies. Additionally, we enriched our dataset with 2,676 cytology slides.

To prepare data for training we generate a filtered dataset by splitting WSIs into nonoverlapping patches and filtering out the background patches using Otsu thresholding. In training, we randomly sample tissue patches from the filtered dataset. We use subsets of different sizes depending on the model being trained. For Hibou-L model we use 1.2B clean patches, for Hibou-B we use 512M clean patches. Each unique patch is sampled only once per training.

## 3.1.1 Data Augmentations

DINOv2 uses data augmentations to generate different views of the same image. We use the following set of augmentations in training:

- Random angle rotation [9]
- Random horizontal and vertical flips
- RandStainNA [13]
- Color jittering

We use RandstainNA in addition to a standard color jittering augmentation as it was shown to improve the performance on WSI-specific downstream tasks [14]. We also don't use solarization in line with [8].

## 3.2 Training details

We use DINOv2 framework [5] with registers [15]. Hibou-B model is trained on 8 A100-80G GPUs with a total batch size of 1024 for 500k iterations. Hibou-L model is trained on 32 A100-40G GPUs with a total batch size of 1024 for 1.175M iterations. Model weights are initialized randomly.

Figure 1: A distribution of tissue types and stains in our dataset

<!-- image -->

Figure 2: Dataset used for training

<!-- image -->

## 4 Results

To evaluate our models we use public datasets and perform evaluation on both patch-level and slide-level tasks. Since our models were trained exclusively on a private dataset it makes the evaluation on public data a fair representation of the ability of our models to generalize to the unseen data.

## 4.1 Patch-level benchmarks

To evaluate a model performance on a patch-level classification task we use a linear probing protocol. We extract features from each image using the pretrained model and then train a linear layer to perform classification. We use SGD as an optimizer and a cosine annealing learning rate. No data augmentations are used in training. For datasets with predefined train-validation-test splits, the official splits are used. In cases where only train-test splits are provided, the training set is randomly partitioned into training and validation subsets. The model checkpoint that achieves the best performance on the validation set is selected, and this checkpoint is then used to evaluate the test set to obtain the final test metrics.

We use the following datasets:

- CRC-100K : This publicly available dataset includes 107,180 H&amp;E-stained images (224×224 pixels) at 20× magnification, obtained from colorectal cancer scans. The images are classified into nine tissue types, representing various components of colorectal tissue, including both healthy and cancerous structures. For our experiments, we utilized only the unnormalized version of the dataset (NCT-CRC-HE-100K-NONORM).
- MHIST : Dataset for colorectal polyp classification, consists of 3,152 H&amp;E-stained images (224×224 pixels). The dataset's primary task is to distinguish between hyperplastic polyps (HP) and sessile serrated adenomas (SSA).
- PCam : The PatchCamelyon public dataset comprises 327,680 H&amp;E-stained images (96×96 pixels). These images are derived from lymph node sections of breast cancer patients and are labeled with binary annotations indicating the presence or absence of metastatic tissue. For testing, we upsampled the images to 224×224 pixels.
- MSI-CRC : The dataset comprises 193,312 unique image patches (224×224 pixels, 0.5 µm/px) derived from histological images of colorectal cancer patients in the TCGA cohort. Images are color-normalized using the Macenko method. The dataset is categorized into "MSS" (microsatellite stable) and "MSIMUT" (microsatellite instable or highly mutated) groups.
- MSI-STAD : The dataset comprises 218,578 unique image patches (224×224 pixels, 0.5 µm/px) derived from histological images of gastric cancer patients in the TCGA cohort. Images are color-normalized using the Macenko method. The dataset is categorized into "MSS" (microsatellite stable) and "MSIMUT" (microsatellite instable or highly mutated) groups.
- TIL-DET : This dataset consists of 304,097 H&amp;E images (100×100 pixels, 0.5 µm/px) with or without tumor-infiltrating lymphocytes (TILs) covering 23 different cancer types from the TCGA cohort.

Table 1: Linear probing benchmarks reporting top-1 accuracy. * Metrics for Virchow and RudolfV are derived from the respective papers, as these models are not open-sourced [7, 8].

| Dataset   |   Phikon [6] |   Kaiko-B8 [10] | Virchow [7]   | RudolfV [8]   |   Prov-GigaPath [12] |   Hibou-B |   Hibou-L |
|-----------|--------------|-----------------|---------------|---------------|----------------------|-----------|-----------|
| CRC-100K  |        0.917 |           0.949 | 0.968 *       | 0.973 *       |                0.968 |     0.955 |     0.966 |
| PCAM      |        0.916 |           0.919 | 0.933 *       | 0.944 *       |                0.947 |     0.946 |     0.943 |
| MHIST     |        0.791 |           0.832 | 0.834 *       | 0.821 *       |                0.839 |     0.812 |     0.849 |
| MSI-CRC   |        0.75  |           0.786 | -             | 0.755 *       |                0.771 |     0.779 |     0.797 |
| MSI-STAD  |        0.76  |           0.814 | -             | 0.788 *       |                0.784 |     0.797 |     0.825 |
| TIL-DET   |        0.944 |           0.945 | -             | 0.943 *       |                0.939 |     0.942 |     0.943 |
| AVG (1-3) |        0.875 |           0.9   | 0.912         | 0.913         |                0.918 |     0.904 |     0.919 |
| AVG (1-6) |        0.846 |           0.874 | -             | 0.871         |                0.875 |     0.872 |     0.887 |

Hibou-L achieves the highest average accuracy across all six datasets, as indicated in Table 1, setting new state-of-the-art performance. The consistent performance across multiple datasets demonstrates the robustness of Hibou-L in handling various histopathological tasks. This robustness is critical for practical applications in clinical settings, where variability in tissue samples can be significant.

## 4.2 Slide-level benchmarks

We evaluate our model on a classification task using publicly available datasets hosted on The Cancer Genome Atlas (TCGA). We use a weakly supervised approach where each slide is divided into nonoverlapping foreground patches and each sequence of patches corresponding to a single slide is assigned a single label. For feature extraction, we utilize a pretrained model to generate features for each patch. Then we use a pooling model based on the attention mechanism to aggregate these feature sequences and perform classification. During the training process, only the parameters of the pooling model are updated, while the parameters of the feature extractor remain frozen. We use the AdamW [16] optimizer for training and do not apply any data augmentations.

The evaluation is conducted on the following datasets:

- BRCA : A TCGA-BRCA project, containing 963 WSIs that are labeled: infiltrating duct carcinoma (767 WSIs) or lobular carcinoma (196 WSIs).
- NSCLC : A combination of TCGA-LUAD and TCGA-LUSC projects, containing 973 WSIs that are labeled: squamous cell carcinoma (520 WSIs) or adenocarcinoma (453 WSIs).

- RCC : A combination of TCGA-KIRC, TCGA-KIRP, and TCGA-KICH projects, containing 927 WSIs that are labeled: renal cell carcinoma (113 WSIs), clear cell adenocarcinoma (523 WSIs), papillary adenocarcinoma (291 WSIs).

Each dataset is divided into training, validation, and test subsets following an 80:10:10 ratio. The model is trained using the training subset, and its performance is monitored on the validation subset. We select the checkpoint with the highest validation performance and use this model to evaluate the test subset.

Table 2: AUC, WSI subtyping benchmarks, test subset

| Dataset   |   Prov-GigaPath[12] |   Hibou-B |   Hibou-L |
|-----------|---------------------|-----------|-----------|
| BRCA      |               0.918 |     0.929 |     0.946 |
| NSCLC     |               0.967 |     0.952 |     0.969 |
| RCC       |               0.987 |     0.993 |     0.996 |

Hibou-L achieves the highest AUC across all three datasets, as shown in Table 2, while Hibou-B surpasses ProvGigaPath 4 in two out of three benchmarks despite having 13 times fewer parameters. This achievement underscores the advanced capabilities of the Hibou models in generating high-quality patch-level features that contribute to accurate slide-level predictions. The efficiency of Hibou-B, in particular, highlights its potential for practical applications where computational resources may be limited, yet high performance is still required. The consistent top performance of HibouL across diverse datasets further demonstrates its robustness and adaptability in handling various histopathological classification tasks.

## 5 Discussion and Future Work

In this study, we introduced the Hibou family of vision transformer models, leveraging the DINOv2 framework for self-supervised pretraining on histopathology data. Despite the promising results, the Hibou-L model has only been trained on approximately one-sixth of our full dataset. We anticipate that further training on more data will enhance the model's performance metrics, as additional data often leads to improved generalization and robustness in Vision Transformers.

Future work will focus on expanding our evaluation benchmarks to include additional subtyping tasks and new tasks like segmentation, which are critical for comprehensive histopathological analysis. Furthermore, we plan to investigate slide-level pretraining as this approach has the potential to improve the performance on WSI downstream tasks. Another promising direction for future research involves utilizing Hibou models as vision encoders in Large Vision-Language Models (LVLMs). These models integrate visual and textual data, enabling sophisticated interactions with histopathological slides. For instance, an LVLM could allow pathologists to query the model in natural language about specific features or abnormalities observed in a slide, receive detailed explanations, and even generate descriptive reports. This interactive capability could enhance diagnostic accuracy, streamline workflows, and facilitate a more intuitive and comprehensive analysis of histopathological data.

We have open-sourced the Hibou-B model to support further research and development in the community. The model is available for a wide range of applications, including commercial use, and can be accessed at https://github.com/HistAI/hibou. We encourage researchers and practitioners to build upon our work, contributing to the advancement of AI in histopathology.
