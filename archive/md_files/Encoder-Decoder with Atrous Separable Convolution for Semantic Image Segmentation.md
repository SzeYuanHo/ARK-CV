## Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

Authors: Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam

## Abstract

Spatial pyramid pooling module or encode-decoder structure are used in deep neural networks for semantic segmentation task. The former networks are able to encode multi-scale contextual information by probing the incoming features with filters or pooling operations at multiple rates and multiple effective fields-of-view, while the latter networks can capture sharper object boundaries by gradually recovering the spatial information. In this work, we propose to combine the advantages from both methods. Specifically, our proposed model, DeepLabv3+, extends DeepLabv3 by adding a simple yet effective decoder module to refine the segmentation results especially along object boundaries. We further explore the Xception model and apply the depthwise separable convolution to both Atrous Spatial Pyramid Pooling and decoder modules, resulting in a faster and stronger encoder-decoder network. We demonstrate the effectiveness of the proposed model on PASCAL VOC 2012 and Cityscapes datasets, achieving the test set performance of 89.0% and 82.1% without any post-processing. Our paper is accompanied with a publicly available reference implementation of the proposed models in Tensorflow at https: //github.com/tensorflow/models/tree/master/research/deeplab .

Keywords: Semantic image segmentation, spatial pyramid pooling, encoderdecoder, and depthwise separable convolution.

## 1 Introduction

Semantic segmentation with the goal to assign semantic labels to every pixel in an image [1,2,3,4,5] is one of the fundamental topics in computer vision. Deep convolutional neural networks [6,7,8,9,10] based on the Fully Convolutional Neural Network [8,11] show striking improvement over systems relying on hand-crafted features [12,13,14,15,16,17] on benchmark tasks. In this work, we consider two types of neural networks that use spatial pyramid pooling module [18,19,20] or encoder-decoder structure [21,22] for semantic segmentation, where the former one captures rich contextual information by pooling features at different resolution while the latter one is able to obtain sharp object boundaries.

In order to capture the contextual information at multiple scales, DeepLabv3 [23] applies several parallel atrous convolution with different rates (called Atrous Spatial Pyramid Pooling, or ASPP), while PSPNet [24] performs pooling operations at different grid scales. Even though rich semantic information is encoded in the last feature map, detailed information related to object boundaries is missing due to the pooling or convolutions with striding operations within the network backbone. This could be alleviated by applying the atrous convolution to extract denser feature maps. However, given the design of state-of-art neural networks [7,9,10,25,26] and limited GPU memory, it is computationally prohibitive to extract output feature maps that are 8, or even 4 times smaller than the input resolution. Taking ResNet-101 [25] for example, when applying atrous convolution to extract output features that are 16 times smaller than input resolution, features within the last 3 residual blocks (9 layers) have to be dilated. Even worse, 26 residual blocks ( 78 layers!) will be affected if output features that are 8 times smaller than input are desired. Thus, it is computationally intensive if denser output features are extracted for this type of models. On the other hand, encoder-decoder models [21,22] lend themselves to faster computation (since no features are dilated) in the encoder path and gradually recover sharp object boundaries in the decoder path. Attempting to combine the advantages from both methods, we propose to enrich the encoder module in the encoder-decoder networks by incorporating the multi-scale contextual information.

Fig. 1. We improve DeepLabv3, which employs the spatial pyramid pooling module (a), with the encoder-decoder structure (b). The proposed model, DeepLabv3+, contains rich semantic information from the encoder module, while the detailed object boundaries are recovered by the simple yet effective decoder module. The encoder module allows us to extract features at an arbitrary resolution by applying atrous convolution.

<!-- image -->

In particular, our proposed model, called DeepLabv3+, extends DeepLabv3 [23] by adding a simple yet effective decoder module to recover the object boundaries, as illustrated in Fig. 1. The rich semantic information is encoded in the output of DeepLabv3, with atrous convolution allowing one to control the density of the encoder features, depending on the budget of computation resources. Furthermore, the decoder module allows detailed object boundary recovery.

Motivated by the recent success of depthwise separable convolution [27,28,26,29,30], we also explore this operation and show improvement in terms of both speed and accuracy by adapting the Xception model [26], similar to [31], for the task of semantic segmentation, and applying the atrous separable convolution to both the ASPP and decoder modules. Finally, we demonstrate the effectiveness of the proposed model on PASCAL VOC 2012 and Cityscapes datasts and attain the test set performance of 89.0% and 82.1% without any post-processing, setting a new state-of-the-art.

In summary, our contributions are:

- -We propose a novel encoder-decoder structure which employs DeepLabv3 as a powerful encoder module and a simple yet effective decoder module.
- -In our structure, one can arbitrarily control the resolution of extracted encoder features by atrous convolution to trade-off precision and runtime, which is not possible with existing encoder-decoder models.
- -We adapt the Xception model for the segmentation task and apply depthwise separable convolution to both ASPP module and decoder module, resulting in a faster and stronger encoder-decoder network.
- -Our proposed model attains a new state-of-art performance on PASCAL VOC 2012 and Cityscapes datasets. We also provide detailed analysis of design choices and model variants.
- -We make our Tensorflow-based implementation of the proposed model publicly available at https://github.com/tensorflow/models/tree/master/ research/deeplab .

## 2 Related Work

Models based on Fully Convolutional Networks (FCNs) [8,11] have demonstrated significant improvement on several segmentation benchmarks [1,2,3,4,5]. There are several model variants proposed to exploit the contextual information for segmentation [12,13,14,15,16,17,32,33], including those that employ multi-scale inputs ( i.e ., image pyramid) [34,35,36,37,38,39] or those that adopt probabilistic graphical models (such as DenseCRF [40] with efficient inference algorithm [41]) [42,43,44,37,45,46,47,48,49,50,51,39]. In this work, we mainly discuss about the models that use spatial pyramid pooling and encoder-decoder structure.

Spatial pyramid pooling: Models, such as PSPNet [24] or DeepLab [39,23], perform spatial pyramid pooling [18,19] at several grid scales (including imagelevel pooling [52]) or apply several parallel atrous convolution with different rates (called Atrous Spatial Pyramid Pooling, or ASPP). These models have shown promising results on several segmentation benchmarks by exploiting the multi-scale information.

Encoder-decoder: The encoder-decoder networks have been successfully applied to many computer vision tasks, including human pose estimation [53], object detection [54,55,56], and semantic segmentation [11,57,21,22,58,59,60,61,62,63,64]. Typically, the encoder-decoder networks contain (1) an encoder module that gradually reduces the feature maps and captures higher semantic information, and (2) a decoder module that gradually recovers the spatial information. Building on top of this idea, we propose to use DeepLabv3 [23] as the encoder module and add a simple yet effective decoder module to obtain sharper segmentations.

Fig. 2. Our proposed DeepLabv3+ extends DeepLabv3 by employing a encoderdecoder structure. The encoder module encodes multi-scale contextual information by applying atrous convolution at multiple scales, while the simple yet effective decoder module refines the segmentation results along object boundaries.

<!-- image -->

Depthwise separable convolution: Depthwise separable convolution [27,28] or group convolution [7,65], a powerful operation to reduce the computation cost and number of parameters while maintaining similar (or slightly better) performance. This operation has been adopted in many recent neural network designs [66,67,26,29,30,31,68]. In particular, we explore the Xception model [26], similar to [31] for their COCO 2017 detection challenge submission, and show improvement in terms of both accuracy and speed for the task of semantic segmentation.

## 3 Methods

In this section, we briefly introduce atrous convolution [69,70,8,71,42] and depthwise separable convolution [27,28,67,26,29]. We then review DeepLabv3 [23] which is used as our encoder module before discussing the proposed decoder module appended to the encoder output. We also present a modified Xception model [26,31] which further improves the performance with faster computation.

## 3.1 Encoder-Decoder with Atrous Convolution

Atrous convolution: Atrous convolution, a powerful tool that allows us to explicitly control the resolution of features computed by deep convolutional neural networks and adjust filter's field-of-view in order to capture multi-scale information, generalizes standard convolution operation. In the case of two-dimensional signals, for each location i on the output feature map y and a convolution filter w , atrous convolution is applied over the input feature map x as follows:

Fig. 3. 3 × 3 Depthwise separable convolution decomposes a standard convolution into (a) a depthwise convolution (applying a single filter for each input channel) and (b) a pointwise convolution (combining the outputs from depthwise convolution across channels). In this work, we explore atrous separable convolution where atrous convolution is adopted in the depthwise convolution, as shown in (c) with rate = 2.

<!-- formula-not-decoded -->

where the atrous rate r determines the stride with which we sample the input signal. We refer interested readers to [39] for more details. Note that standard convolution is a special case in which rate r = 1. The filter's field-of-view is adaptively modified by changing the rate value.

Depthwise separable convolution: Depthwise separable convolution, factorizing a standard convolution into a depthwise convolution followed by a pointwise convolution ( i.e ., 1 × 1 convolution), drastically reduces computation complexity. Specifically, the depthwise convolution performs a spatial convolution independently for each input channel, while the pointwise convolution is employed to combine the output from the depthwise convolution. In the TensorFlow [72] implementation of depthwise separable convolution, atrous convolution has been supported in the depthwise convolution ( i.e ., the spatial convolution), as illustrated in Fig. 3. In this work, we refer the resulting convolution as atrous separable convolution , and found that atrous separable convolution significantly reduces the computation complexity of proposed model while maintaining similar (or better) performance.

DeepLabv3 as encoder: DeepLabv3 [23] employs atrous convolution [69,70,8,71] to extract the features computed by deep convolutional neural networks at an arbitrary resolution. Here, we denote output stride as the ratio of input image spatial resolution to the final output resolution (before global pooling or fullyconnected layer). For the task of image classification, the spatial resolution of the final feature maps is usually 32 times smaller than the input image resolution and thus output stride = 32. For the task of semantic segmentation, one can adopt output stride = 16 (or 8) for denser feature extraction by removing the striding in the last one (or two) block(s) and applying the atrous convolution correspondingly ( e.g ., we apply rate = 2 and rate = 4 to the last two blocks respectively for output stride = 8). Additionally, DeepLabv3 augments the Atrous Spatial Pyramid Pooling module, which probes convolutional features at multiple scales by applying atrous convolution with different rates, with the image-level features [52]. We use the last feature map before logits in the original DeepLabv3 as the encoder output in our proposed encoder-decoder structure. Note the encoder output feature map contains 256 channels and rich semantic information. Besides, one could extract features at an arbitrary resolution by applying the atrous convolution, depending on the computation budget.

Proposed decoder: The encoder features from DeepLabv3 are usually computed with output stride = 16. In the work of [23], the features are bilinearly upsampled by a factor of 16, which could be considered a naive decoder module. However, this naive decoder module may not successfully recover object segmentation details. We thus propose a simple yet effective decoder module, as illustrated in Fig. 2. The encoder features are first bilinearly upsampled by a factor of 4 and then concatenated with the corresponding low-level features [73] from the network backbone that have the same spatial resolution ( e.g ., Conv2 before striding in ResNet-101 [25]). We apply another 1 × 1 convolution on the low-level features to reduce the number of channels, since the corresponding lowlevel features usually contain a large number of channels ( e.g ., 256 or 512) which may outweigh the importance of the rich encoder features (only 256 channels in our model) and make the training harder. After the concatenation, we apply a few 3 × 3 convolutions to refine the features followed by another simple bilinear upsampling by a factor of 4. We show in Sec. 4 that using output stride = 16 for the encoder module strikes the best trade-off between speed and accuracy. The performance is marginally improved when using output stride = 8 for the encoder module at the cost of extra computation complexity.

## 3.2 Modified Aligned Xception

The Xception model [26] has shown promising image classification results on ImageNet [74] with fast computation. More recently, the MSRA team [31] modifies the Xception model (called Aligned Xception) and further pushes the performance in the task of object detection. Motivated by these findings, we work in the same direction to adapt the Xception model for the task of semantic image segmentation. In particular, we make a few more changes on top of MSRA's modifications, namely (1) deeper Xception same as in [31] except that we do not modify the entry flow network structure for fast computation and memory efficiency, (2) all max pooling operations are replaced by depthwise separable convolution with striding, which enables us to apply atrous separable convolution to extract feature maps at an arbitrary resolution (another option is to extend the atrous algorithm to max pooling operations), and (3) extra batch normalization [75] and ReLU activation are added after each 3 × 3 depthwise convolution, similar to MobileNet design [29]. See Fig. 4 for details.

## 4 Experimental Evaluation

We employ ImageNet-1k [74] pretrained ResNet-101 [25] or modified aligned Xception [26,31] to extract dense feature maps by atrous convolution. Our implementation is built on TensorFlow [72] and is made publicly available.

Fig. 4. We modify the Xception as follows: (1) more layers (same as MSRA's modification except the changes in Entry flow), (2) all the max pooling operations are replaced by depthwise separable convolutions with striding, and (3) extra batch normalization and ReLU are added after each 3 × 3 depthwise convolution, similar to MobileNet.

<!-- image -->

The proposed models are evaluated on the PASCAL VOC 2012 semantic segmentation benchmark [1] which contains 20 foreground object classes and one background class. The original dataset contains 1 , 464 ( train ), 1 , 449 ( val ), and 1 , 456 ( test ) pixel-level annotated images. We augment the dataset by the extra annotations provided by [76], resulting in 10 , 582 ( trainaug ) training images. The performance is measured in terms of pixel intersection-over-union averaged across the 21 classes (mIOU).

We follow the same training protocol as in [23] and refer the interested readers to [23] for details. In short, we employ the same learning rate schedule ( i.e ., 'poly' policy [52] and same initial learning rate 0 . 007), crop size 513 × 513, fine-tuning batch normalization parameters [75] when output stride = 16, and random scale data augmentation during training. Note that we also include batch normalization parameters in the proposed decoder module. Our proposed model is trained end-to-end without piecewise pretraining of each component.

## 4.1 Decoder Design Choices

We define 'DeepLabv3 feature map' as the last feature map computed by DeepLabv3 ( i.e ., the features containing ASPP features and image-level features), and [ k × k, f ] as a convolution operation with kernel k × k and f filters.

When employing output stride = 16, ResNet-101 based DeepLabv3 [23] bilinearly upsamples the logits by 16 during both training and evaluation. This simple bilinear upsampling could be considered as a naive decoder design, attaining the performance of 77 . 21% [23] on PASCAL VOC 2012 val set and is 1 . 2% better than not using this naive decoder during training ( i.e ., downsampling groundtruth during training). To improve over this naive baseline, our proposed model 'DeepLabv3+' adds the decoder module on top of the encoder output, as shown in Fig. 2. In the decoder module, we consider three places for different design choices, namely (1) the 1 × 1 convolution used to reduce the channels of the low-level feature map from the encoder module, (2) the 3 × 3 convolution used to obtain sharper segmentation results, and (3) what encoder low-level features should be used.

To evaluate the effect of the 1 × 1 convolution in the decoder module, we employ [3 × 3 , 256] and the Conv2 features from ResNet-101 network backbone, i.e ., the last feature map in res2x residual block (to be concrete, we use the feature map before striding). As shown in Tab. 1, reducing the channels of the low-level feature map from the encoder module to either 48 or 32 leads to better performance. We thus adopt [1 × 1 , 48] for channel reduction.

We then design the 3 × 3 convolution structure for the decoder module and report the findings in Tab. 2. We find that after concatenating the Conv2 feature map (before striding) with DeepLabv3 feature map, it is more effective to employ two 3 × 3 convolution with 256 filters than using simply one or three convolutions. Changing the number of filters from 256 to 128 or the kernel size from 3 × 3 to 1 × 1 degrades performance. We also experiment with the case where both Conv2 and Conv3 feature maps are exploited in the decoder module. In this case, the decoder feature map are gradually upsampled by 2, concatenated with Conv3 first and then Conv2, and each will be refined by the [3 × 3 , 256] operation. The whole decoding procedure is then similar to the U-Net/SegNet design [21,22]. However, we have not observed significant improvement. Thus, in the end, we adopt the very simple yet effective decoder module: the concatenation of the DeepLabv3 feature map and the channel-reduced Conv2 feature map are refined by two [3 × 3 , 256] operations. Note that our proposed DeepLabv3+ model has output stride = 4. We do not pursue further denser output feature map ( i.e ., output stride &lt; 4) given the limited GPU resources.

## 4.2 ResNet-101 as Network Backbone

To compare the model variants in terms of both accuracy and speed, we report mIOU and Multiply-Adds in Tab. 3 when using ResNet-101 [25] as network backbone in the proposed DeepLabv3+ model. Thanks to atrous convolution, we

| Channels   | 8      | 16     | 32     | 48     | 64     |
|------------|--------|--------|--------|--------|--------|
| mIOU       | 77.61% | 77.92% | 78.16% | 78.21% | 77.94% |

Table 1. PASCAL VOC 2012 val set. Effect of decoder 1 × 1 convolution used to reduce the channels of low-level feature map from the encoder module. We fix the other components in the decoder structure as using [3 × 3 , 256] and Conv2.

Table 2. Effect of decoder structure when fixing [1 × 1 , 48] to reduce the encoder feature channels. We found that it is most effective to use the Conv2 (before striding) feature map and two extra [3 × 3 , 256] operations. Performance on VOC 2012 val set.

| Features Conv2 Conv3      | 3 × 3 Conv Structure   | mIOU   |
|---------------------------|------------------------|--------|
| glyph[check]              | [3 × 3 , 256]          | 78.21% |
| glyph[check]              | [3 × 3 , 256] × 2      | 78.85% |
| glyph[check]              | [3 × 3 , 256] × 3      | 78.02% |
| glyph[check]              | [3 × 3 , 128]          | 77.25% |
| glyph[check]              | [1 × 1 , 256]          | 78.07% |
| glyph[check] glyph[check] | [3 × 3 , 256]          | 78.61% |

Table 3. Inference strategy on the PASCAL VOC 2012 val set using ResNet-101 . train OS : The output stride used during training. eval OS : The output stride used during evaluation. Decoder : Employing the proposed decoder structure. MS : Multiscale inputs during evaluation. Flip : Adding left-right flipped inputs.

|   Encoder train OS eval OS |   Encoder train OS eval OS | Decoder      | MS           | Flip mIOU           | Multiply-Adds   |
|----------------------------|----------------------------|--------------|--------------|---------------------|-----------------|
|                         16 |                         16 |              |              | 77.21%              | 81.02B          |
|                         16 |                          8 |              |              | 78.51%              | 276.18B         |
|                         16 |                          8 |              | glyph[check] | 79.45%              | 2435.37B        |
|                         16 |                          8 |              | glyph[check] | glyph[check] 79.77% | 4870.59B        |
|                         16 |                         16 | glyph[check] |              | 78.85%              | 101.28B         |
|                         16 |                         16 | glyph[check] | glyph[check] | 80.09%              | 898.69B         |
|                         16 |                         16 | glyph[check] | glyph[check] | glyph[check] 80.22% | 1797.23B        |
|                         16 |                          8 | glyph[check] |              | 79.35%              | 297.92B         |
|                         16 |                          8 | glyph[check] | glyph[check] | 80.43%              | 2623.61B        |
|                         16 |                          8 | glyph[check] | glyph[check] | glyph[check] 80.57% | 5247.07B        |
|                         32 |                         32 |              |              | 75.43%              | 52.43B          |
|                         32 |                         32 | glyph[check] |              | 77.37%              | 74.20B          |
|                         32 |                         16 | glyph[check] |              | 77.80%              | 101.28B         |
|                         32 |                          8 | glyph[check] |              | 77.92%              | 297.92B         |

are able to obtain features at different resolutions during training and evaluation using a single model.

Table 4. Single-model error rates on ImageNet-1K validation set.

| Model                 | Top-1 Error Top-5   | Error   |
|-----------------------|---------------------|---------|
| Reproduced ResNet-101 | 22.40%              | 6.02%   |
| Modified Xception     | 20.19%              | 5.17%   |

Baseline: The first row block in Tab. 3 contains the results from [23] showing that extracting denser feature maps during evaluation ( i.e ., eval output stride = 8) and adopting multi-scale inputs increases performance. Besides, adding leftright flipped inputs doubles the computation complexity with only marginal performance improvement.

Adding decoder: The second row block in Tab. 3 contains the results when adopting the proposed decoder structure. The performance is improved from 77 . 21% to 78 . 85% or 78 . 51% to 79 . 35% when using eval output stride = 16 or 8, respectively, at the cost of about 20B extra computation overhead. The performance is further improved when using multi-scale and left-right flipped inputs.

Coarser feature maps: We also experiment with the case when using train output stride = 32 ( i.e ., no atrous convolution at all during training) for fast computation. As shown in the third row block in Tab. 3, adding the decoder brings about 2% improvement while only 74.20B Multiply-Adds are required. However, the performance is always about 1% to 1.5% below the case in which we employ train output stride = 16 and different eval output stride values. We thus prefer using output stride = 16 or 8 during training or evaluation depending on the complexity budget.

## 4.3 Xception as Network Backbone

We further employ the more powerful Xception [26] as network backbone. Following [31], we make a few more changes, as described in Sec. 3.2.

ImageNet pretraining: The proposed Xception network is pretrained on ImageNet-1k dataset [74] with similar training protocol in [26]. Specifically, we adopt Nesterov momentum optimizer with momentum = 0.9, initial learning rate = 0.05, rate decay = 0.94 every 2 epochs, and weight decay 4 e -5. We use asynchronous training with 50 GPUs and each GPU has batch size 32 with image size 299 × 299. We did not tune the hyper-parameters very hard as the goal is to pretrain the model on ImageNet for semantic segmentation. We report the single-model error rates on the validation set in Tab. 4 along with the baseline reproduced ResNet-101 [25] under the same training protocol. We have observed 0.75% and 0.29% performance degradation for Top1 and Top5 accuracy when not adding the extra batch normalization and ReLU after each 3 × 3 depthwise convolution in the modified Xception.

The results of using the proposed Xception as network backbone for semantic segmentation are reported in Tab. 5.

Baseline: We first report the results without using the proposed decoder in the first row block in Tab. 5, which shows that employing Xception as network backbone improves the performance by about 2% when train output stride = eval output stride = 16 over the case where ResNet-101 is used. Further improvement can also be obtained by using eval output stride = 8, multi-scale inputs during inference and adding left-right flipped inputs. Note that we do not employ the multi-grid method [77,78,23], which we found does not improve the performance.

Adding decoder: As shown in the second row block in Tab. 5, adding decoder brings about 0.8% improvement when using eval output stride = 16 for all the different inference strategies. The improvement becomes less when using eval output stride = 8.

Using depthwise separable convolution: Motivated by the efficient computation of depthwise separable convolution, we further adopt it in the ASPP and the decoder modules. As shown in the third row block in Tab. 5, the computation complexity in terms of Multiply-Adds is significantly reduced by 33% to 41%, while similar mIOU performance is obtained.

Pretraining on COCO: For comparison with other state-of-art models, we further pretrain our proposed DeepLabv3+ model on MS-COCO dataset [79], which yields about extra 2% improvement for all different inference strategies.

Pretraining on JFT: Similar to [23], we also employ the proposed Xception model that has been pretrained on both ImageNet-1k [74] and JFT-300M dataset [80,26,81], which brings extra 0.8% to 1% improvement.

Test set results: Since the computation complexity is not considered in the benchmark evaluation, we thus opt for the best performance model and train it with output stride = 8 and frozen batch normalization parameters. In the end, our 'DeepLabv3+' achieves the performance of 87.8% and 89.0% without and with JFT dataset pretraining.

Qualitative results: We provide visual results of our best model in Fig. 6. As shown in the figure, our model is able to segment objects very well without any post-processing.

Failure mode: As shown in the last row of Fig. 6, our model has difficulty in segmenting (a) sofa vs . chair, (b) heavily occluded objects, and (c) objects with rare view.

## 4.4 Improvement along Object Boundaries

In this subsection, we evaluate the segmentation accuracy with the trimap experiment [14,40,39] to quantify the accuracy of the proposed decoder module near object boundaries. Specifically, we apply the morphological dilation on 'void' label annotations on val set, which typically occurs around object boundaries. We then compute the mean IOU for those pixels that are within the dilated band (called trimap) of 'void' labels. As shown in Fig. 5 (a), employing the proposed decoder for both ResNet-101 [25] and Xception [26] network backbones improves the performance compared to the naive bilinear upsampling. The improvement is more significant when the dilated band is narrow. We have observed 4.8% and 5.4% mIOU improvement for ResNet-101 and Xception respectively at the smallest trimap width as shown in the figure. We also visualize the effect of employing the proposed decoder in Fig. 5 (b).

Table 5. Inference strategy on the PASCAL VOC 2012 val set when using modified Xception . train OS : The output stride used during training. eval OS : The output stride used during evaluation. Decoder : Employing the proposed decoder structure. MS : Multi-scale inputs during evaluation. Flip : Adding left-right flipped inputs. SC : Adopting depthwise separable convolution for both ASPP and decoder modules. COCO : Models pretrained on MS-COCO. JFT : Models pretrained on JFT.

|   Encoder train OS eval OS |   Encoder train OS eval OS | Decoder      | MS           | Flip         | SC           | COCO         | mIOU   | Multiply-Adds   |
|----------------------------|----------------------------|--------------|--------------|--------------|--------------|--------------|--------|-----------------|
|                         16 |                         16 |              |              |              |              |              | 79.17% | 68.00B          |
|                         16 |                         16 |              | glyph[check] |              |              |              | 80.57% | 601.74B         |
|                         16 |                         16 |              | glyph[check] | glyph[check] |              |              | 80.79% | 1203.34B        |
|                         16 |                          8 |              |              |              |              |              | 79.64% | 240.85B         |
|                         16 |                          8 |              | glyph[check] |              |              |              | 81.15% | 2149.91B        |
|                         16 |                          8 |              | glyph[check] | glyph[check] |              |              | 81.34% | 4299.68B        |
|                         16 |                         16 | glyph[check] |              |              |              |              | 79.93% | 89.76B          |
|                         16 |                         16 | glyph[check] | glyph[check] |              |              |              | 81.38% | 790.12B         |
|                         16 |                         16 | glyph[check] | glyph[check] | glyph[check] |              |              | 81.44% | 1580.10B        |
|                         16 |                          8 | glyph[check] |              |              |              |              | 80.22% | 262.59B         |
|                         16 |                          8 | glyph[check] | glyph[check] |              |              |              | 81.60% | 2338.15B        |
|                         16 |                          8 | glyph[check] | glyph[check] | glyph[check] |              |              | 81.63% | 4676.16B        |
|                         16 |                         16 | glyph[check] |              |              | glyph[check] |              | 79.79% | 54.17B          |
|                         16 |                         16 | glyph[check] | glyph[check] | glyph[check] | glyph[check] |              | 81.21% | 928.81B         |
|                         16 |                          8 | glyph[check] |              |              | glyph[check] |              | 80.02% | 177.10B         |
|                         16 |                          8 | glyph[check] | glyph[check] | glyph[check] | glyph[check] |              | 81.39% | 3055.35B        |
|                         16 |                         16 | glyph[check] |              |              | glyph[check] | glyph[check] | 82.20% | 54.17B          |
|                         16 |                         16 | glyph[check] | glyph[check] | glyph[check] | glyph[check] | glyph[check] | 83.34% | 928.81B         |
|                         16 |                          8 | glyph[check] |              |              | glyph[check] | glyph[check] | 82.45% | 177.10B         |
|                         16 |                          8 | glyph[check] | glyph[check] | glyph[check] | glyph[check] | glyph[check] | 83.58% | 3055.35B        |
|                         16 |                         16 | glyph[check] |              |              | glyph[check] | glyph[check] | 83.03% | 54.17B          |
|                         16 |                         16 | glyph[check] | glyph[check] | glyph[check] | glyph[check] | glyph[check] | 84.22% | 928.81B         |
|                         16 |                          8 | glyph[check] |              |              | glyph[check] | glyph[check] | 83.39% | 177.10B         |
|                         16 |                          8 | glyph[check] | glyph[check] | glyph[check] | glyph[check] | glyph[check] | 84.56% | 3055.35B        |

## 4.5 Experimental Results on Cityscapes

In this section, we experiment DeepLabv3+ on the Cityscapes dataset [3], a large-scale dataset containing high quality pixel-level annotations of 5000 images (2975, 500, and 1525 for the training, validation, and test sets respectively) and about 20000 coarsely annotated images.

As shown in Tab. 7 (a), employing the proposed Xception model as network backbone (denoted as X-65) on top of DeepLabv3 [23], which includes the ASPP module and image-level features [52], attains the performance of 77.33% on the validation set. Adding the proposed decoder module significantly improves the performance to 78.79% (1.46% improvement). We notice that removing the augmented image-level feature improves the performance to 79.14%, showing that in DeepLab model, the image-level features are more effective on the PASCAL VOC 2012 dataset. We also discover that on the Cityscapes dataset, it is effective to increase more layers in the entry flow in the Xception [26], the same as what [31] did for the object detection task. The resulting model building on top of the deeper network backbone (denoted as X-71 in the table), attains the best performance of 79.55% on the validation set.

Table 6. PASCAL VOC 2012 test set results with top-performing models.

| Method                       |   mIOU |
|------------------------------|--------|
| Deep Layer Cascade (LC) [82] |   82.7 |
| TuSimple [77]                |   83.1 |
| Large Kernel Matters [60]    |   83.6 |
| Multipath-RefineNet [58]     |   84.2 |
| ResNet-38 MS COCO [83]       |   84.9 |
| PSPNet [24]                  |   85.4 |
| IDW-CNN [84]                 |   86.3 |
| CASIA IVA SDN [63]           |   86.6 |
| DIS [85]                     |   86.8 |
| DeepLabv3 [23]               |   85.7 |
| DeepLabv3-JFT [23]           |   86.9 |
| DeepLabv3+ (Xception)        |   87.8 |
| DeepLabv3+ (Xception-JFT)    |   89   |

Fig. 5. (a) mIOU as a function of trimap band width around the object boundaries when employing train output stride = eval output stride = 16. BU : Bilinear upsampling. (b) Qualitative effect of employing the proposed decoder module compared with the naive bilinear upsampling (denoted as BU ). In the examples, we adopt Xception as feature extractor and train output stride = eval output stride = 16.

After finding the best model variant on val set, we then further fine-tune the model on the coarse annotations in order to compete with other state-of-art models. As shown in Tab. 7 (b), our proposed DeepLabv3+ attains a performance of 82.1% on the test set, setting a new state-of-art performance on Cityscapes.

<!-- image -->

Fig. 6. Visualization results on val set. The last row shows a failure mode.

| Backbone   | Decoder      | ASPP         | Image-Level   |   mIOU |
|------------|--------------|--------------|---------------|--------|
| X-65       | glyph[check] | glyph[check] | glyph[check]  |  77.33 |
| X-65       |              | glyph[check] | glyph[check]  |  78.79 |
| X-65       | glyph[check] | glyph[check] |               |  79.14 |
| X-71       | glyph[check] | glyph[check] |               |  79.55 |

(a) val set results

| Method         | Coarse       |   mIOU |
|----------------|--------------|--------|
| ResNet-38 [83] | glyph[check] |   80.6 |
| PSPNet [24]    | glyph[check] |   81.2 |
| Mapillary [86] | glyph[check] |   82   |
| DeepLabv3      | glyph[check] |   81.3 |
| DeepLabv3+     | glyph[check] |   82.1 |

(b) test set results

Table 7. (a) DeepLabv3+ on the Cityscapes val set when trained with train fine set. (b) DeepLabv3+ on Cityscapes test set. Coarse : Use train extra set (coarse annotations) as well. Only a few top models are listed in this table.

## 5 Conclusion

Our proposed model 'DeepLabv3+' employs the encoder-decoder structure where DeepLabv3 is used to encode the rich contextual information and a simple yet effective decoder module is adopted to recover the object boundaries. One could also apply the atrous convolution to extract the encoder features at an arbitrary resolution, depending on the available computation resources. We also explore the Xception model and atrous separable convolution to make the proposed model faster and stronger. Finally, our experimental results show that the proposed model sets a new state-of-the-art performance on PASCAL VOC 2012 and Cityscapes datasets.
