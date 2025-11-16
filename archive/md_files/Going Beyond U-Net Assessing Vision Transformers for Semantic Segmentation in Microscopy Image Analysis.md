## Going Beyond U-Net: Assessing Vision Transformers for Semantic Segmentation in Microscopy Image Analysis

Authors: Illia Tsiporenko 1 , Pavel Chizhov 1 , 2 , and Dmytro Fishman 1 , 3

## Abstract

Segmentation is a crucial step in microscopy image analysis. Numerous approaches have been developed over the past years, ranging from classical segmentation algorithms to advanced deep learning models. While U-Net remains one of the most popular and wellestablished models for biomedical segmentation tasks, recently developed transformer-based models promise to enhance the segmentation process of microscopy images. In this work, we assess the efficacy of transformers, including UNETR, the Segment Anything Model, and Swin-UPerNet, and compare them with the well-established U-Net model across various image modalities such as electron microscopy, brightfield, histopathology, and phase-contrast. Our evaluation identifies several limitations in the original Swin Transformer model, which we address through architectural modifications to optimise its performance. The results demonstrate that these modifications improve segmentation performance compared to the classical U-Net model and the unmodified Swin-UPerNet. This comparative analysis highlights the promise of transformer models for advancing biomedical image segmentation. It demonstrates that their efficiency and applicability can be improved with careful modifications, facilitating their future use in microscopy image analysis tools.

Keywords: Biomedical image segmentation · Image analysis · Transformers · Deep Learning

## 1 Introduction

Identifying objects in microscopy images is a crucial first step in successful image analysis [11]. Precise segmentation of various cellular structures, including nuclei, enables the extraction and analysis of vital morphological features. However, achieving accurate and efficient segmentation remains challenging due to the complex and heterogeneous nature of microscopy data.

Deep learning algorithms are powerful tools for segmentation tasks, given their ability to generalise and understand underlying image structures. For a long time, the traditional Convolutional Neural Network (CNN) U-Net [24] has been one of the most popular and well-established models in this field, demonstrating notable results in various microscopy image segmentation tasks. However, many new deep learning models have been developed over the past few years, with transformers among the most promising. Transformers use the attention mechanism [28] at their core, allowing them to capture complex image structures, provide an unlimited receptive field, and incorporate more local context than traditional CNNs. These features are particularly advantageous for enhancing the segmentation process of microscopy images, where capturing local context is essential for improving the finer details in segmentation masks.

This paper provides an assessment of segmentation models, which utilise some of the most popular vision transformers as image encoders - Vision Transformer [8] (ViT), present in the UNETR [13] model, and Swin Transformer [22], present in the Swin-UPerNet [22] model. Additionally, we assess the novel foundational Segment Anything Model (SAM), which uses user-defined prompts to enhance the segmentation process. We compare these models to the robust and lightweight U-Net model, which serves as our baseline.

Swin Transformer in combination with UPerNet-based decoder demonstrated promising performance in semantic image segmenation [22]. However, the model's reliance on processing image patches of size 4 inevitably leads to the loss of fine-grained details. This loss of low-level details, coupled with the use of bilinear interpolation in the decoder, may compromise the overall performance of the model by reducing the precision of the segmentation boundaries and affecting the accuracy of object delineation. In our work, we aim to address this issue by designing encoder and decoder enhancements to introduce local context and improve the flexibility of mask generation, thereby improving detail capture and segmentation accuracy.

By studying the capabilities of these transformer-based models, we aim to highlight their potential advantages and drawbacks compared to the traditional U-Net model. In our comparative analysis, we seek to demonstrate the promise of transformers in advancing microscopy image segmentation.

## 2 Related Work

While U-Net [24] remains one of the most popular models for segmentation tasks in the biomedical domain, recent years have seen the development of many new transformer-based models [4-6,10,12, 13,17,22,25,31]. These models can be roughly divided into two broad categories: transformer-CNN and hybrid models. Transformer-CNN models use transformers as the primary image encoder, while CNN layers in the decoder generate the segmentation masks. Examples include UNETR [13], Swin UNETR [12], Swin-UPerNet [22], and SETR [31]. On the other hand, hybrid models utilize both CNN and transformer layers in the encoder but retain CNN layers in the decoder. An example of this model type are TransUNet [5], SU-Net [10], and CS-UNet [1].

Even though hybrid models are more flexible in design and allow for more architectural experiments, one major advantage of Transformer-CNN models is the use of intact transformer encoders pre-trained on large datasets like ImageNet [7]. Such models generally show superior performance over hybrid ones, as the improvement coming from transformers is mostly related to large and diverse pre-training [22]. This difference renders hybrid models less preferable, thus we decided to omit them in our experiments.

There is also a separate category of models that have been recently gaining popularity foundational models. These models are typically trained on massive datasets and offer zero-shot generalisation. Such a model was recently introduced for image segmentation - Segment Anything Model [17] (SAM). SAM uses user-defined prompts, such as bounding boxes or points, to guide and improve the segmentation process.

As the Swin Transformer has demonstrated superior performance in many imaging tasks, numerous new re developed using Swin as a basis [4, 5, 12]. Swin-UPerNet was the first model that used Swin as the encoder in combination with the UPerNet decoder. Following its success, many other segmentation models that employ Swin as the backbone were developed. Some propose different types of decoders compared to the original Swin-UPerNet, such as Swin UNETR [12] and SSformer [25], while others follow the idea of hybrid models such as CS-UNet [1], where both the encoder and decoder are revised. However, to the best of our knowledge, there is a notable gap in research on the original Swin-UPerNet, with opportunities for enhancing this model. Thus, in this work, we explore Swin-UPerNet, identify its potential limitations, and address them through custom modifications. These modifications, while greatly improving its performance, preserve the original architecture of the model, enabling the reuse of the pre-trained weights, which improves the convergence of the loss during training and enhances the overall performance of the model.

Table 1: Detailed overview of datasets used in the study. Here, we detail the number of images present in each dataset, the resolution and the number of channels of each image in the dataset, the segmentation target and the modality of the images.

| Dataset             | Images   | Images   | Resolution   | Channels   | Target           | Modality            |
|---------------------|----------|----------|--------------|------------|------------------|---------------------|
| Dataset             | (Train   | / Test)  |              |            |                  |                     |
| Seven Cell Lines    | 2016     | / 504    | 1080 × 1080  | 1          | Nuclei           | Brightfield         |
| Electron Microscopy | 366      | / 99     | Varies       | 3          | Varies           | Electron Microscopy |
| LIVECell            | 3253     | / 1986   | 768 × 512    | 1          | Individual Cells | Phase Contrast      |
| MoNuSeg             | 250      | / 140    | 512 × 512    | 3          | Nuclei           | Histopathology      |

## 3 Methods

Our work aims to comprehensively compare the well-established U-Net model and notable transformerbased models, including UNETR [13], Swin-UPerNet [22], and the Segment Anything Model [17], specifically within the microscopy domain. Additionally, we design custom modifications for SwinUPerNet to enhance its performance in microscopy image segmentation tasks. In this section, we will describe the datasets, the configuration of the models, and the approaches for training and evaluation.

## 3.1 Datasets

To assess the performance and capabilities of those models, we chose four different datasets, each representing a distinct image modality, offering unique segmentation challenges. Table 1 provides a detailed overview of the chosen datasets and Figure 1 provides an example image from each dataset. The Electron Microscopy dataset [30] contains images of various resolutions, focusing on electron microscopy image modality. The Seven Cell Lines dataset [11] contains brightfield images with a resolution of 1024 × 1024 , targeting nuclei of cells. The LIVECell dataset [9] consists of phase-contrast images of 768 × 512 , mainly focusing on individual cells. MoNuSeg dataset [18] [19] includes whole-slide histopathology images, which we tiled into smaller images of size 512 × 512 , with the main target being nuclei of tissue cells. This thorough collection of datasets allows us to fairly evaluate the capabilities of each of the segmentation models in various segmentation scenarios, ensuring an in-depth assessment.

Fig. 1: We present example crops of the images from (a) Electron Microscopy dataset [30], (b) MoNuSeg dataset [18,19], (c) Seven Cell Lines dataset [11], and (d) LIVECell dataset [9]

<!-- image -->

## 3.2 Segmentation Models

The U-Net [24] model serves as our practical baseline for this study as it is known for its robust and notable performance in microscopy image segmentation tasks. Its architecture was specifically designed for biomedical imaging tasks, featuring a symmetric encoder-decoder structure with skip connections.

For transformer-based models, we chose the models that employ different types of vision transformer encoders and different approaches for segmentation to assess their advantages and limitations. Among many transformer-based image encoders, two are well-established in the field - the ViT [8] and the Swin Transformer [22]. The first transformer model we chose for assessment was UNETR [13], which was initially designed for 3D biomedical image segmentation. We adapted UNETR for 2D image segmentation to use in our experiments. UNETR utilises ViT at its core to encode the images. The decoder part is similar to the U-Net model, consisting of a series of convolutional layers and transposed convolutions to upscale the feature maps produced by the encoder. On the other hand, Swin-UPerNet [22] utilises a different type of encoder - Swin Transformer, with its unique windowed attention mechanism and patch merging operations making it possible to extract the features from the input image on different scales. UPerNet serves as the decoder of the network, consisting of a Feature Pyramid Network [20] (FPN), a Pyramid Pooling Module [14] (PPM), and the final upscaling layer - bilinear interpolation. Lastly, the Segment Anything Model [17] presents a unique approach to the segmentation tasks by utilising user prompts, such as points or bounding boxes, enhancing the performance of the model in complex microscopy image segmentation tasks.

U-Net We utilised the Segmentation Models Pytorch [15] (SMP) framework to construct the UNet model. ResNet34, pre-trained on the ImageNet dataset [7], was used as the backbone for the network. We kept the network parameters and configuration as predefined in the framework. The depth of the encoder was set to 5 stages, where each stage generates feature maps two times smaller in spatial dimension than the previous one.

Table 2: Detailed overview of parameters of the small (Swin-S) and base (Swin-B) versions of SwinUPerNet.

| Parameter            | Swin-S       | Swin-B       |
|----------------------|--------------|--------------|
| Patch size           | 4 × 4        | 4 × 4        |
| Embedding dimension  | 96           | 128          |
| Window size          | 7            | 12           |
| Depth of transformer | 2, 2, 18, 2  | 2, 2, 18, 2  |
| Heads in each stage  | 3, 6, 12, 24 | 4, 8, 16, 32 |
| Hidden size in MLP   | 768          | 1024         |

UNETR We adapted the original version of UNETR [13], designed for biomedical tasks, to handle 2D microscopy images. We followed the same original architectural ideas of the model with slight adjustments - all of Conv3D layers in the decoder part of the network were replaced with Conv2D. We utilised the base version of ViT, pre-trained on ImageNet dataset [7], with the patch size of 16 × 16

Segment Anything Model We utilised SAM [17] out of the box, pre-trained on the SA-1B dataset [17]. We assessed all of the ways to segment images with SAM: automatic segmentation, providing user-defined point or bounding box prompts. The bounding boxes represent the highest degree of user interaction with the model and, thus - the highest degree of effort compared to the point prompting. The model expects bounding boxes as input in the [B × 4 ] format, where B represents the number of output masks. Similarly, the input format for point prompts is [B × N × 2 ], where B is the number of output masks, and N represents the number of points per object. On the other hand, automatic segmentation requires no interaction with the model from the user side, segmenting all potential objects and structures in the image. To assess the performance of the model, we used the OpenCV [16] framework to generate relevant points and bounding boxes from the ground truth masks, which were provided in the datasets. These prompts served as the input to the model alongside the corresponding image to obtain final results.

Swin-UPerNet We used the Swin-UPerNet [22] model, pre-trained on the ImageNet dataset [7]. We utilised the small (Swin-S) and base (Swin-B) versions of the Swin Transformer for the encoder. The decoder remained the same, consisting of FPN, PPM, and final linear interpolation. The default configuration of Swin-UPerNet uses a patch size of 4 × 4 with a window of size 7 (Table 2 provides a detailed overview of configuration).

## 3.3 Swin-UPerNet Modifications

While exploring the Swin-UPerNet, we identified several issues in the network. As the original model uses a patch size of 4 × 4 , the input size reduces by 4 times after the patch partitioning operation. This causes the misalignment in the decoder part of the network. In order to align the dimension of the final segmentation mask with the input image, bilinear interpolation is used in the original implementation of the model. While this approach provides a clear and lightweight solution, it has some drawbacks. Bilinear interpolation does not have learnable parameters and can introduce artefacts in the segmentation mask, potentially decreasing the performance of the model. To address this issue, we propose an architectural improvement by replacing the bilinear interpolation with a series of convolutional and transposed convolutional layers, introducing more learnable parameters and enhancing the quality of the segmentation mask.

Fig. 2: Representation of Swin-UPerNet architecture, which consists of Swin Transformer (blue blocks) and the UPerNet decoder (green blocks). Orange dotted rectangles provide an overview of our proposed modifications to the architecture of the model. Conv denotes a convolutional block, which is made of a convolutional layer, batch normalisation, and ReLU activation. Deconv denotes transposed convolutional operation. The circle with a line denotes an addition operation, followed by a convolutional operation with kernel size 3 × 3 .

<!-- image -->

As the segmentation tasks can be challenging when dealing with microscopy images, and the size of the objects may greatly vary, ranging from tiny nuclei to whole cells, it is necessary to induce more local information. We proposed different ways to achieve it and enhance the performance of Swin-UPerNet, specifically in microscopy image segmentation tasks:

1. Decreasing the patch size to induce more local context.
2. Adding a skip connection with a convolutional block from the input image to the decoder part of the network.
3. Adding additional Swin Transformer stages into the backbone.

To address these issues, we designed several architectural and configurational improvements. Figure 2 illustrates the main ideas of our designed modifications, and Table 3 provides an overview of modification present in different types of proposed architectures.

These modifications aim to induce more local context in the model and potentially increase its performance. We present a detailed overview and explanation of each designed modfication below.

Swin-S-PS2 We changed the patch size from 4 × 4 to 2 × 2 , increasing the ability of the model to capture finer details. We kept all other parameters the same.

Swin-S-Conv We replaced the bilinear interpolation, which has no learnable parameters and sometimes produces artefacts in the final segmentation mask, with a series of convolutional blocks alongside transposed convolutions. Each convolutional block consists of a Conv2D layer with a kernel size of 3 × 3 and padding of 1, Batch normalisation, and ReLU activation function. We further enhanced the model by adding a skip connection with a convolutional block and merging its feature maps with those generated by the decoder. This set of modifications aims to enhance the quality of the final segmentation mask and increase the performance of the model.

Swin-S-TB We decreased the patch size of 2 × 2 and kept the convolutional and transposed convolutions in the decoder part. Additionally, we integrated one more stage in the backbone of the network, consisting of two consecutive Swin Transformer blocks. All of those changes aim to increase the ability of the model to process complex features and induce more of the local context.

Swin-S-TB-Skip We kept the previous ideas of Swin-S-TB. Additionally, we added a skip connection in order to introduce more low-level information and see how much it contributes to the quality of the final segmentation mask.

Swin-S-Pyramid We decided to decrease the patch size even more - to 1 × 1 . We extended the backbone of the network, adding two additional stages. We changed the embedding dimension to 24 in order to align with the desired input of the rest of the backbone, keeping the pre-trained weights. The architecture of the decoder was adjusted so that the output of two additional stages in the backbone aligns with the FPN, yielding the final segmentation mask with the same dimensions as the input image. With these changes, we do not need to have any additional convolutional layers or interpolation in the decoder.

## 8 Tsiporenko et al.

Table 3: Detailed overview of Swin-UPerNet modifications. Each row represents the Swin-UPerNet modification. The checkmarks show modifications present in the architecture. Deconv2D denotes a series of convolutional blocks and transposed convolutional layers. Skip denotes the presence of a skip connection from the input image to the decoder part of the network. Extra Stage denotes the additional Swin stages in the encoder of the network. Pyramid denotes the modification with an extended encoder and decoder.

| Models         | Modifications   | Modifications   | Modifications   | Modifications   |
|----------------|-----------------|-----------------|-----------------|-----------------|
|                | Patch Size      | Deconv2D        | Skip Connection | Extra Stage     |
| Swin-UPerNet   | 4 × 4           | -               | -               | -               |
| Swin-S-PS2     | 2 × 2           | -               | -               | -               |
| Swin-S-Conv    | 4 × 4           | ✓               | ✓               | -               |
| Swin-S-Pyramid | 1 × 1           | -               | ✓               | ✓               |
| Swin-S-TB      | 2 × 2           | ✓               | -               | ✓               |
| Swin-S-TB-Skip | 2 × 2           | ✓               | ✓               | ✓               |

## 3.4 Training Pipeline

We designed our custom training pipeline to effectively train and switch between different deep learning models and their modifications. We utilised Pytorch [23], Weights and Biases [2], and Hydra [29] frameworks for flexible training, configuration management, tracking and logging the experiments.

Data Preprocessing Input images are normalised and transformed using the Albumentations [3] framework. We apply horizontal and vertical flips and random rotation to the input. This choice enhances the ability of the model to understand the structure of the images across different orientations and scales of the input images, which often can be the case in microscopy images.

Additionally, we use the random cropping of size 224 × 224 during the training process. We found this crop size beneficial as it allows us to avoid unnecessary padding during training SwinUPerNet and other transformer-based models. If the height and the width of the input image were not multiple of the product of window size and scaling factors across the layers of the network, the additional padding is applied to process the image. This padding can lead to artefacts in the final segmentation mask and potentially affect the performance of the model.

Training Each model was trained for 150 epochs, which we found optimal for convergence. The batch size of 16 images was used - the maximum that could fit into our GPU memory. We sampled 500 images from the dataset during training each epoch to provide diverse examples and enhance the robustness of the model. We chose the combination of Dice [26] and Focal [21] losses for training with weight coefficients set to 0.9 and 0.1, serving as the standard ratio. We compute the loss as follows:

<!-- formula-not-decoded -->

Here Y is the ground truth mask, ˆ Y is the predicted mask, and α and β are the weight coefficients.

Table 4: Performance results of UNETR, Swin-UPerNet with Swin-S and Swin-B as the backbones, and Segment Anything Model operating in three different modes (the number of point and bounding box prompts are equal to the number of ground truth instances in each test image) compared to U-Net across datasets. Each row represents the model, while each column represents the obtained F1 and IoU values on each dataset. The best scores are highlighted in bold , and the second best scores are underlined.

| Models               | LIVECell   | LIVECell   | Seven Cell Lines   | Seven Cell Lines   | MoNuSeg   | MoNuSeg   | Electron Microscopy   | Electron Microscopy   |
|----------------------|------------|------------|--------------------|--------------------|-----------|-----------|-----------------------|-----------------------|
|                      | F1         | IoU        | F1                 | IoU                | F1        | IoU       | F1                    | IoU                   |
| U-Net                | 0.92       | 0.86       | 0.81               | 0.70               | 0.80      | 0.68      | 0.92                  | 0.88                  |
| UNETR                | 0.93       | 0.87       | 0.80               | 0.68               | 0.82      | 0.70      | 0.83                  | 0.75                  |
| Swin-S               | 0.92       | 0.85       | 0.75               | 0.61               | 0.82      | 0.70      | 0.93                  | 0.88                  |
| Swin-B               | 0.92       | 0.86       | 0.77               | 0.64               | 0.83      | 0.71      | 0.93                  | 0.88                  |
| SAM (Bounding Box)   | 0.86       | 0.76       | 0.78               | 0.64               | 0.88      | 0.79      | 0.87                  | 0.80                  |
| SAM (Point Prompts)  | 0.57       | 0.46       | 0.27               | 0.16               | 0.71      | 0.57      | 0.61                  | 0.52                  |
| SAM (Automatic Mode) | 0.46       | 0.35       | 0.17               | 0.10               | 0.66      | 0.50      | 0.77                  | 0.67                  |

Evaluation For the evaluation, we used the F1 score and IoU score as our primary metrics to thoroughly assess the performance of all models. The evaluation itself was done on separate test sets of full-size images. For the UNETR and U-Net models, we applied a custom tiling algorithm - the input image was divided into tiles, and the model predicted the segmentation mask for each of the tiles. Those segmentation masks of tiles were merged back to obtain the final full-size segmentation mask.

## 3.5 Computational Resourses

We trained all of the models on the High-Performance Computing Cluster of the University of Tartu [27], which has Nvidia Tesla V100 GPUs with 32 gigabytes of VRAM and Nvidia Tesla A100 GPUs with 40 and 80 gigabytes of VRAM running CUDA 12.3 with Driver version 545.23.08.

## 4 Results

Here, we present the results of our experiments. Firstly, we compare the U-Net model against the chosen transformer-based models - UNETR, Swin-S and Swin-B, and SAM. Next, we will present and detail the results of our designed modifications for Swin-S, specifically designed to enhance its performance in microscopy image segmentation tasks. These modifications seek to induce much more local context and finer details, which is necessary when dealing with microscopy images.

## 4.1 Comparison of Transformer-based Models

We fine-tuned and evaluated U-Net, UNETR, and Swin-UPerNet on each dataset separately, following our pipeline outlined in Section 3.4. In contrast, we assessed SAM's out-of-the-box performance without fine-tuning to evaluate its immediate usability. We provided the bounding boxes and point prompts equal to the number of instances in each test image for a fair comparison. The results, detailed in Table 4, show that U-Net consistently performs well across all datasets, achieving the highest IoU of 0.88 on the Electron Microscopy dataset, setting a strong baseline for other models. UNETR generally matches the performance of U-Net but lags on the Electron Microscopy dataset with the 0.75 IoU score. Both small and basic versions of Swin-UPerNet are behind U-Net and UNETR across almost all of the datasets, except for the Electron Microscopy, showing the same results as the U-Net. These observations highlight that the traditional CNN approach remains good and robust. Segment Anything Model, when using bounding box prompts, shows reasonable performance but does not achieve the same levels as fine-tuned models. Although bounding box prompts provide decent test scores, performance decreases when switching to point prompts or using automatic mode, particularly on the Seven Cell Lines dataset.

Swin-TB-Skip

Fig. 3: Predicted segmentation masks of Swin-S-TB-Skip, UNETR, U-Net, and Segment Anything Model (utilising bounding box and point prompts and enabling automatic segmentation). The white contour represents the outline of the ground truth mask. The colour overlay represents the predicted segmentation mask of the model: green colour for Swin-S-TB-Skip, red colour for UNETR, blue colour for U-Net, and purple colour for SAM. We made the image from MoNuSeg dataset grayscale for the purpose of better visibility of predicted segmentation masks.

<!-- image -->

Table 5: Performance results of Swin-UPerNet (Swin-S) modifications compared to U-Net and original Swin-UPerNet (Swin-S and Swin-B) across datasets. Each row represents the model or the modification, while each column represents the obtained F1 and IoU values on each dataset. The best scores are highlighted in bold , and the second best scores are underlined.

| Models         | LIVECell   | LIVECell   | Seven Cell Lines   | Seven Cell Lines   | MoNuSeg   | MoNuSeg   | Electron Microscopy   | Electron Microscopy   |
|----------------|------------|------------|--------------------|--------------------|-----------|-----------|-----------------------|-----------------------|
|                | F1         | IoU        | F1                 | IoU                | F1        | IoU       | F1                    | IoU                   |
| U-Net          | 0.92       | 0.86       | 0.81               | 0.70               | 0.80      | 0.68      | 0.92                  | 0.88                  |
| Swin-S         | 0.92       | 0.85       | 0.75               | 0.61               | 0.82      | 0.70      | 0.93                  | 0.88                  |
| Swin-B         | 0.92       | 0.86       | 0.77               | 0.64               | 0.83      | 0.71      | 0.93                  | 0.88                  |
| Swin-S-PS2     | 0.93       | 0.87       | 0.81               | 0.69               | 0.82      | 0.70      | 0.94                  | 0.89                  |
| Swin-S-Conv    | 0.92       | 0.86       | 0.77               | 0.64               | 0.81      | 0.69      | 0.95                  | 0.90                  |
| Swin-S-TB      | 0.93       | 0.87       | 0.83               | 0.72               | 0.82      | 0.70      | 0.91                  | 0.86                  |
| Swin-S-TB-Skip | 0.93       | 0.88       | 0.84               | 0.74               | 0.82      | 0.71      | 0.95                  | 0.91                  |
| Swin-S-Pyramid | 0.91       | 0.85       | 0.80               | 0.67               | 0.80      | 0.67      | 0.90                  | 0.84                  |

## 4.2 Comparison of Swin-UPerNet Modifications

We fine-tuned all of the designed modifications on each dataset separately, utilising the proposed train pipeline, described in Section 3.4 and draw a comparison between the original U-Net and Swin-UPerNet proposed architectures, aiming to increase its performance in microscopy image segmentation. All of the presented modifications were based on the Swin-S architecture. From Table 5, we can see that our Swin-S-TB-Skip modification excels across almost all datasets, surpassing U-Net and the original Swin-UPerNet models, both small (Swin-S) and basic (Swin-B) versions, achieving higher IoU score. Apart from this, we can see a notable increase in the performance of Swin-S-TB-Skip compared to Swin-UPerNet on the Seven Cell Lines dataset, which contains brightfield images with the nuclei as a target. We consider this modification to be our best among the others.

## 4.3 FLOPs and Parameters of the Models

Here, we provide an overview of the FLOPs and parameters for UNETR, U-Net, SAM, SwinUPerNet, and its best modification - Swin-S-TB-Skip. The FLOPs were calculated using a 3channel image of 224 × 224 as the input to each model. Table 6 shows that U-Net has the least amount of FLOPs among all other models.

## 12 Tsiporenko et al.

Table 6: Model Parameters and FLOPs. We calculated the number of FLOPs by passing the 3-channel image of size 224 × 224 to the model. Swin-S-PS2 denotes our modification of Swin-UPerNet with decreased patch size to 2 × 2 . Swin-S-TB-Skip denotes our best modification, with the extension of the encoder, decrease in patch size, replacement of interpolation and addition of skip connection. We cannot provide FLOPs for SAM as it depends on the amount of prompts provided to the model, which can greatly vary.

| Model                 |   Params (M) | FLOPs   | (G)   |
|-----------------------|--------------|---------|-------|
| U-Net                 |         24.4 |         | 12    |
| UNETR                 |        111.7 |         | 234   |
| Swin-B                |        121.1 |         | 128   |
| Swin-S                |         81.1 |         | 98    |
| Swin-S-PS2 (ours)     |         81.1 |         | 390   |
| Swin-S-TB-Skip (ours) |         82.1 |         | 452   |
| SAM                   |         93.7 |         | -     |

Another notable observation is that Swin-S-TB has almost 4.5 times more FLOPs than the original Swin-UPerNet. While this may sound alarming, the reason behind it is simple. As the patch size decreases to 2 × 2 , the number of patches in the image increases, leading to a fourfold increase in the size of the attention matrix. Although this modification requires many more FLOPs to run, it still fits within the memory constraints of the same GPU. We could not provide FLOPs for the SAM model, as it depends on the number of prompts passed to the model by the user, which can greatly vary.

## 5 Discussion

Our experimental results offer an overview of the capabilities of modern transformer-based models -UNETR, Swin-UPerNet, and SAM in recognizing and segmenting various objects and structures within microscopy images across different modalities. We compared these models to the established U-Net model and evaluated their performance. Figure 3 provides examples of the predictions of our best modification - Swin-TB-Skip compared to the UNETR, U-Net, and Segment Anything Model across all of the datasets. The results in Table 4 show that U-Net remains a strong contender for semantic segmentation in microscopy images. While UNETR and Swin-UPerNet generally match UNet's performance, their significant computational demands make them less practical for real-world applications.

The original Swin-UPerNet slightly falls behind the U-Net across most of the datasets, but several innovative modifications we introduced greatly enhanced its performance. Modifications such as extending the encoder, replacing interpolation layers in the decoder, adding an extra skip connection, and reducing patch size aim to improve local context modelling. This is crucial for better segmentation quality in microscopy images, where object and structure variability is high. Our top-performing modification, Swin-S-TB-Skip, showed notable improvements across all datasets. It is also notable, that our best modification surpassed the basic version of the original SwinUperNet, which has more parameters, highlighting the value and the relevance of our architectural improvements. We emphasize the substantial performance boost on the Seven Cell Lines dataset compared to the original Swin-UPerNet. The brightfield modality and the challenge of segmenting cell nuclei make this dataset especially difficult.

Segment Anything Model, the first foundational segmentation model, has delivered mixed results across various datasets. The model heavily depends on user-defined prompts to achieve improvements in performance, particularly noticeable when bounding boxes are employed in contrast to its baseline automatic segmentation capabilities. This reliance on user input for optimal performance significantly diminishes its utility compared to other models. Without user interaction, SAM's segmentations are often suboptimal, limiting its direct comparability and competitiveness with automated models that do not require such inputs. Moreover, SAM lacks class awareness, indiscriminately segmenting all detectable objects and structures. This restricts its applicability for specialized tasks, such as cell segmentation, where targeted recognition of specific classes is crucial. Nonetheless, with further developments, such as user interface, SAM could evolve into a valuable tool for interactive annotation.

These findings demonstrate that there is still potential for advancement in transformer-based models. Their unique attention mechanisms hold promise for achieving cutting-edge performance in segmentation tasks. By continuously refining and improving these architectures, we can unlock their full potential and establish new benchmarks in the field.

## 6 Conclusion

In this study, we make two major contributions to the field of microscopy image segmentation. Our first contribution is a detailed comparison between the well-established and popular U-Net model and several innovative transformer-based deep learning models. These include Swin-UPerNet, which features a unique windowed attention mechanism, the Segment Anything Model with its interactive prompt segmentation approach, and UNETR, which blends a traditional U-Net-like decoder with a modern vision transformer encoder. Our evaluations reveal that while these modern transformerbased models perform comparably to U-Net, there is still room for improvement.

Our second major contribution focuses on enhancing the performance of the Swin-UPerNet model. We conducted a series of experiments aimed at increasing its robustness and performance across various microscopy images. The modifications we implemented greatly improved the performance of the model. Our revised version, Swin-S-TB-Skip, on average, outperformed the original Swin-UPerNet and U-Net across all tested microscopy datasets, achieving a higher IoU score.

However, these performance gains come with increased computational demand (see Table 6). Future research should, therefore, concentrate on optimising these architectural enhancements for practical applications and integrating them into diverse microscopy image analysis workflows and tools.
