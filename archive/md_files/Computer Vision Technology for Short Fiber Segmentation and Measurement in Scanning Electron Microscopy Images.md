## Computer Vision Technology for Short Fiber Segmentation and Measurement in Scanning Electron Microscopy Images

Citation: Kurkin, E.; Minaev, E.; Sedelnikov, A.; Pioquinto, J.G.Q.; Chertykovtseva, V.; Gavrilov, A. Computer Vision Technology for Short Fiber Segmentation and Measurement in Scanning Electron Microscopy Images. Technologies 2024 , 12 , 249. https://doi.org/10.3390/ technologies12120249


## Abstract

Computer vision technology for the automatic recognition and geometric characterization of carbon and glass fibers in scanning electron microscopy images is proposed. The proposed pipeline, combining the SAM model and DeepLabV3+, provides the generalizability and accuracy of the foundational SAM model and the ability to quickly train on a small amount of data via the DeepLabV3+ model. The pipeline was trained several times more rapidly with lower requirements for computing resources than fine-tuning the SAM model, with comparable inference time. On the basis of the pipeline, an end-to-end technology for processing images of electron microscopic fibers was developed, the input of which is images with metadata and the output of which is statistics on the distribution of the geometric characteristics of the fibers. This innovation is of great practical importance for modeling the physical characteristics of materials. This paper proposes a few-shot training procedure for the DeepLabV3+/SAM pipeline, combining the training of the DeepLabV3+ model weights and the SAM model parameters. It allows effective training of the pipeline using only 37 real labeled images. The pipeline was then adapted to a new type of fiber and background using 15 additional real labeled images. This article also proposes a method for generating synthetic data for training neural network models, which improves the quality of segmentation by the IoU and PixAcc metrics from 0.943 and 0.949 to 0.953 and 0.959, i.e., by 1% on average. The developed pipeline significantly reduces the time required to evaluate fiber length in scanning electron microscope images.

## Keywords: 

short carbon fibers; glass fibers; virtual training; image segmentation; instance segmentation; semantic segmentation; DeepLabv3+; SAM; Hough technique

## 1. Introduction

Modern short-reinforced composite materials have the following advantages: high strength characteristics, low specific weight, and ease of manufacture and processing [1,2]. These materials are used in many industrial sectors [3]. The peculiarity of such materials is the presence of fibers which significantly affect the mechanical characteristics of future products [4]. Among the most commonly used methods to produce short-reinforced materials are extrusion and injection molding, which use chopped or milled fibers with a low aspect ratio [5].

The macroscopic mechanical properties of fiber-reinforced composite materials depend on the microscopic characteristics of the fibers [6,7], such as their aspect ratio [8,9], orientation, and distribution in the product [10]. The aspect ratio of fibers is included in material models used for the design of short-fiber-reinforced structures [11]. During extrusion and injection, fibers experience shear stresses, resulting in fiber damage and fiber length reduction [12]. The critical fiber length, defined as the minimum length at which a fiber can sustain a load, is used to assess fiber failure. Several models are available for estimating the critical fiber length [13-15]. In [16], fiber modeling was performed, and the effective response of the composite and average stresses on the fibers were obtained. It was shown in [17] that the high aspect ratio of reinforced carbon fibers significantly affects the tensile, flexural, and dynamic mechanical properties, as well as the thermal deflection temperature and the impact strength of reinforced composites. These models are based on an integral approximation of the mechanical characteristics of a composite in a cell of representative volume and include a solution to the orientation tensor closure problem [18]. The authors of [19] presented results on the estimation of an effective viscoelastic stiffness module with controlled accuracy for short fiber composites with spheroidal and spherocylindrical inclusions. The elastic field of an ellipsoidal inclusion was described by Eshelby [20]. The mechanical characteristics of a unidirectional composite were obtained using the Mori-Tanaka model [18], and anisotropy is accounted for by Advani-Tucker averaging [21]. fiber can sustain a load, is used to assess fiber failure. Several models are available for estimating the critical fiber length [13-15]. In [16], fiber modeling was performed, and the e ff ective response of the composite and average stresses on the fibers were obtained. It was shown in [17] that the high aspect ratio of reinforced carbon fibers signi fi cantly a ff ects the tensile, flexural, and dynamic mechanical properties, as well as the thermal deflection temperature and the impact strength of reinforced composites. These models are based on an integral approximation of the mechanical characteristics of a composite in a cell of representative volume and include a solution to the orientation tensor closure problem [18]. The authors of [19] presented results on the estimation of an e ff ective viscoelastic sti ff ness module with controlled accuracy for short fiber composites with spheroidal and spherocylindrical inclusions. The elastic fi eld of an ellipsoidal inclusion was described by Eshelby [20]. The mechanical characteristics of a unidirectional composite were obtained using the Mori-Tanaka model [18], and anisotropy is accounted for by Advani-Tucker averaging [21].

The experimental evaluation of the microstructure of composites requires the recognition of fiber patterns in optical and electron microscope images and in spatial microtomographic images [22-24]. Obtaining two-dimensional images using microscopy requires simpler equipment than using computer microtomography; however, it is a more difficult task to recognize fiber patterns because of their possible layering on top of each other [24-28]. The experimental evaluation of the microstructure of composites requires the recognition of fiber pa tt erns in optical and electron microscope images and in spatial microtomographic images [22-24]. Obtaining two-dimensional images using microscopy requires simpler equipment than using computer microtomography; however, it is a more di ffi cult task to recognize fiber pa tt erns because of their possible layering on top of each other [24-28].

To generate a material model for reinforced polymers, it is necessary to extract a sample of the geometric characteristics of the fibers; this process is performed by analyzing images captured by electron microscopes. The quality of data depends on the visual ability of the researcher, and the speed of data acquisition is based on the researcher's measurement skills. We propose replacing this task, performed by people, with a neural network using deep learning techniques applied to computer vision. To generate a material model for reinforced polymers, it is necessary to extract a sample of the geometric characteristics of the fibers; this process is performed by analyzing images captured by electron microscopes. The quality of data depends on the visual ability of the researcher, and the speed of data acquisition is based on the researcher's measurement skills. We propose replacing this task, performed by people, with a neural network using deep learning techniques applied to computer vision.

In previous research [29,30], the use of image segmentation (specifically, instance segmentation) was proposed using the Mask R-CNN neural network [31,32], which was trained with artificial images that simulated images obtained by scanning electron microscopy. An API of the NX CAD program (in C language) was developed for the creation of cylinders (simulating short fibers) in random arrangements on a solid color background (see Figure 1). In previous research [29,30], the use of image segmentation (speci fi cally, instance segmentation) was proposed using the Mask R-CNN neural network [31,32], which was trained with arti fi cial images that simulated images obtained by scanning electron microscopy. An API of the NX CAD program (in C language) was developed for the creation of cylinders (simulating short fibers) in random arrangements on a solid color background (see Figure 1).

Figure 1. Examples of the arti fi cial images used to train the Mask R-CNN neural network [29]. Figure 1. Examples of the artificial images used to train the Mask R-CNN neural network [29].

The network demonstrated good performance in fiber detection in areas where there was no large accumulation of fibers, and the performance decreased as the accumulation The network demonstrated good performance in fiber detection in areas where there was no large accumulation of fibers, and the performance decreased as the accumulation of instances increased. The network demonstrated good performance in short-fiber detection in images with low numbers of fibers (from 5 to 20 fibers per image). At the moment of testing the network trained on real images of short carbon fiber samples, Mask R-CNN could detect the fibers in images in which there was not a large number of fibers and the fibers did not overlap (see Figure 2a). On the other hand, Mask R-CNN did not demonstrate good performance in images with high fiber concentrations (see Figure 2b). of testing the network trained on real images of short carbon fiber samples, Mask R-CNN could detect the fibers in images in which there was not a large number of fibers and the fibers did not overlap (see Figure 2a). On the other hand, Mask R-CNN did not demonstrate good performance in images with high fiber concentrations (see Figure 2b).

Figure 2. Examples of the performance of Mask R-CNN when detecting short fibers in: ( a ) an image with few fibers, and ( b ) in an image with more fibers and overlapping sets. Figure 2. Examples of the performance of Mask R-CNN when detecting short fibers in: ( a ) an image with few fibers, and ( b ) in an image with more fibers and overlapping sets.

The use of a two-stage image segmentation technique was proposed to solve the problem of fiber detection, especially in assemblies with a large number of overlapping parts. The fi rst stage required the application of a semantic segmentation technique to identify the regions of di ff erent labels in the image. In the second stage, the instance segmentation technique was used to identify objects by labeling each label individually [33]. For the fi rst stage, the DeepLabv3+ [34] or PVT [35] neural networks are recommended, while for the second stage, the use of SAM [36] or SEEM [37] neural networks is recommended [33]. The combination of these two techniques, particularly using the DeepLabv3+ and SAM architectures, achieved signi fi cant improvements in mask prediction [33,38,39]. Traditionally, highly specialized segmentation tasks are often solved by fi ne-tuning basic models such as SAM [40]. However, this comes with signi fi cant compuThe use of a two-stage image segmentation technique was proposed to solve the problem of fiber detection, especially in assemblies with a large number of overlapping parts. The first stage required the application of a semantic segmentation technique to identify the regions of different labels in the image. In the second stage, the instance segmentation technique was used to identify objects by labeling each label individually [33]. For the first stage, the DeepLabv3+ [34] or PVT [35] neural networks are recommended, while for the second stage, the use of SAM [36] or SEEM [37] neural networks is recommended [33]. The combination of these two techniques, particularly using the DeepLabv3+ and SAM architectures, achieved significant improvements in mask prediction [33,38,39]. Traditionally, highly specialized segmentation tasks are often solved by fine-tuning basic models such as SAM [40]. However, this comes with significant computational and time costs. 

In the present work, we propose the detection of short carbon and glass fibers using an image segmentation technique in two stages, notably with the use of the DeepLabv3+ and SAM architectures. As in our previous research, we continue to use arti fi cial images for the training of neural networks, in addition to real images captured with a scanning In the present work, we propose the detection of short carbon and glass fibers using an image segmentation technique in two stages, notably with the use of the DeepLabv3+ and SAM architectures. As in our previous research, we continue to use artificial images for the training of neural networks, in addition to real images captured with a scanning electron microscope. 

- The primary contributions of this paper are as follows: 
1. The two-stage pipeline combining SAM and DeepLabV3+ provides the generalizability and accuracy of the foundational SAM model and the ability to quickly train on a small amount of data from the DeepLabV3+ model (Section 2.5). The pipeline was trained several times more rapidly (~30 min of pipeline vs. ~180 min of SAM) with lower requirements (one RTX 4090 of pipeline vs. 4 Tesla K80 GPUs of SAM) for computing resources than fi ne-tuning the SAM model but with a comparable infer1. The two-stage pipeline combining SAM and DeepLabV3+ provides the generalizability and accuracy of the foundational SAM model and the ability to quickly train on a small amount of data from the DeepLabV3+ model (Section 2.5). The pipeline was trained several times more rapidly (~30 min of pipeline vs. ~180 min of SAM) with lower requirements (one RTX 4090 of pipeline vs. 4 Tesla K80 GPUs of SAM) for computing resources than fine-tuning the SAM model but with a comparable inference time (951 milliseconds of pipeline vs. 900 milliseconds of SAM) (Section 3.2).
2. End-to-end technology for processing images of electron microscope fibers from images with metadata to obtain statistics regarding the distribution of the geometric characteristics of fibers (Section 3.2). The result of this work is statistical data on the 2. End-to-end technology for processing images of electron microscope fibers from images with metadata to obtain statistics regarding the distribution of the geometric characteristics of fibers (Section 3.2). The result of this work is statistical data on the distribution of the geometric characteristics of such fibers; this is of great practical importance for modeling the physical characteristics of materials.
3. A few-shot training procedure for the DeepLabV3+/SAM pipeline combining training of the DeepLabV3+ model weights and SAM model parameters, making pipeline training possible using only 37 real labeled images. The pipeline was then adapted to a new type of fiber and background using 15 additional real labeled images (Section 3.2).
4. A method to generate synthetic data for additional training of neural networks for fiber segmentation (Section 2.4) allowed us to further improve the segmentation quality by 1%.

## 2. Materials and Methods

## 2.1. Methodology

The technology proposed by the authors for the automatic segmentation of carbon and glass short fibers in scanning electron microscopy images consists of two main stages:

1. Semantic segmentation of the original scanning electron microscope image to separate the fibers and the background.
2. Segmentation of individual fiber instances in the image using a foundational segmentation model, and filtering of the instance masks using the results of the first stage segmentation.

In the first stage, background images and fibers are separated using the DeepLabV3+ neural network, and in the second stage, instance segmentation is applied using the Segment Anything Model (SAM) network. In addition to selecting the SAM architecture for the second stage, the Hough transform method was used for comparison. The choice of the Hough transformation was based on the fact that the carbon and glass fibers in the image are straight segments; the search for such objects is a characteristic task of this method [41]. To train neural networks, we propose the use of artificial images and real images captured using a scanning electron microscope.

## 2.2. Creation of Real Images of Short Carbon Fibers

To create real images, the fibers were separated from a binder during the degradation process of a carbon-filled structural composite material based on polyamide-6, which was heated in a muffle furnace at 900 ◦ C for 20 min in a nitrogen atmosphere (Figures 3 and 4). The temperature was increased and decreased at a rate of 5 ◦ C/min, according to the technical process tested in [42]. The images of short glass fibers were obtained by degrading the 30% mass glass-fiber polyamide-6 composite material in a muffle furnace in a nitrogen atmosphere. The degradation process occurred for 1 h at a temperature of 550 ◦ C. This was done to evaluate the performance of the image recognition technique for different types of fibers. The temperature was ramped up and down at a rate of 5 ◦ C/min. The heating of the samples by the muffle furnace ensured the destruction of the binder, and the work in the nitrogen atmosphere protected the fibers from oxidation with oxygen from the air. As a result, short fiber samples were obtained while preserving their geometric characteristics for further study. The muffle furnace temperature was selected for each fiber type to ensure that the binder would degrade while the fibers remained intact.

To capture images that help identify short fibers, the following requirements are suggested:

1. The arrangement of the fibers in a thin layer, preferably a single layer, so that the underlying substrate is visible. The multilayer arrangement of fibers does not allow reliable separation of one fiber from another, especially when considering multiple superpositions of fibers with each other.
2. The fibers were examined on a well-polished background. The presence of background roughness caused, for example, by rough machining or sanding can sometimes cause false positives in subsequent image segmentation (see Figure 5).
3. The magnification of the electron microscope should ensure the capture of four to six medium lengths of fibers. On the one hand, this ensures the reliable measurement of their length and, on the other hand, does not cause large deviations due to the effect of incorrect measurement of the length of the fibers extending beyond the boundaries of the frame (see Figure 6).

Figure 3. Heating of composite material in a muffle furnace. Figure 3. Heating of composite material in a muffle furnace. Figure 3. Heating of composite material in a muffle furnace.

<!-- image -->

Figure 4. A sample subjected to heating by muffle furnace. Figure 4. A sample subjected to heating by muffle furnace. Figure 4. Asample subjected to heating by muffle furnace.

Figure 5. Obtaining images of short fibers with the help of the Tescan Vega scanning electron microscope. Figure 5. Obtaining images of short fibers with the help of the Tescan Vega scanning electron microscope.

Figure 6. Examples of images taken with the Tescan Vega scanning electron microscope. Figure 6. Examples of images taken with the Tescan Vega scanning electron microscope.

<!-- image -->

## 2.3. Labeling of Images 2.3. Labeling of Images

Image labeling was designed with the instance segmentation technique. Semantic and instance segmentation require the correct detection of all objects in an image and precise segmenting of each instance. Therefore, our approach combines object detection, object localization, and object classi fi cation. In other words, this type of segmentation makes a clear distinction between each object classi fi ed as similar instances. Image labeling was designed with the instance segmentation technique. Semantic and instance segmentation require the correct detection of all objects in an image and precise segmenting of each instance. Therefore, our approach combines object detection, object localization, and object classification. In other words, this type of segmentation makes a clear distinction between each object classified as similar instances.

It is necessary that the images used for training are linked to a file that indicates the corresponding annotations of each instance for a neural network to achieve object recognition with the instance segmentation technique. For this purpose, we used the LabelMe program [43]. LabelMe is an open-source image annotation tool. LabelMe allows you to create annotations using polygons for computer vision datasets for object detection, classification, and segmentation.

The fiber labeling of two real images is shown in Figure 7.

## 2.4. Creation of Virtual Images of Short Carbon Fibers and Labeling of Virtual Images

Siemens NX CAD software was used for the virtual creation of assembly images of the short carbon fibers. The software contains tools that simulate the texture and effects of light and shadow, as visualized in real images captured by electron microscopy.

NX Open is a collection of APIs for developing applications for NX. The APIs were implemented in a C programing environment to create .dll files which are executed in NX to create images in order to obtain our virtual dataset.

The virtual dataset comprises a series of images, simulating randomly disperse fibers on a metallic plate, and a .json file corresponding to each image in which the coordinates of the fibers within the picture are stored. Algorithm 1 shows the procedure used to create artificial images in NX. si fi cation, and segmentation. The fiber labeling of two real images is shown in Figure 7.

Figure 7. The fiber labeling of two real images.

<!-- image -->

# Algorithm 1. Algorithm for the creation of artificial images of short carbon fibers. 
Input : Ncyl, Nimg, Pxsize, Llow, Lup, Lleft , Lright , l min, lmax, Dmin, Dmax, Fjson Output : Simg, Sjson 1 Define the XY plane as the visualization plane; 2 Set the image size Pxsize × Pxsize pixels; 3 Delimit the cylinder insertion area (Acyl) of the images with the corners (Lleft , Lup) and (Lright , Llow); 4 for i = 1 to Nimg do 5 for j = 1 to Ncyl do 6 Choose the center of the cylinder randomly delimiting the insertion area; 7 Determine the cylinder length randomly in range [l min , l max]; 8 Determine the cylinder diameter randomly in range [Dmin, Dmax]; Algorithm 1. Algorithm for the creation of artificial images of short carbon fibers. Input : Ncyl , Nimg, Pxsize, L low , Lup, L left , Lright , lmin, lmax, Dmin, Dmax, Fjson Output : Simg, Sjson 1 Define the XY plane as the visualization plane; 2 Set the image size Pxsize × Pxsize pixels; 3 Delimit the cylinder insertion area (Acyl ) of the images with the corners (L left , Lup) and (Lright , L low ); 4 for i = 1 to Nimg do 5 for j = 1 to Ncyl do 6 Choose the center of the cylinder randomly delimiting the insertion area; 7 Determine the cylinder length randomly in range [lmin, lmax]; 8 Determine the cylinder diameter randomly in range [Dmin, Dmax]; 9 Randomly determine the angle of rotation with respect to the axis normal to the XY plane; 10 Randomly determine the angle of rotation with respect to an axis belonging to the XY plane and perpendicular to the longitudinal axis of the cylinder; 11 Build, position, and rotate cylinder Cj; 12 Projecting the cylinder onto the XY plane; 13 if projection ∈ Acyl then 14 Save coordinates of the projection in Fjson,i; 15 else 16 Remove cylinder Cj; 17 Determine colors of the cylinders, texture of the background and lighting of the image Ii; 18 I i → Simg; 19 Fjson,i → Sjson ; 20 return Output

Virtual images were created starting with a few cylinders in each image and gradually adding more cylinders until a similar image to the real images had been obtained. The origin and direction of the cylinders were uniformly distributed. The length of the fibers was modeled using a normal distribution with a mean of 150 mm and a standard deviation of 50 mm. The diameter was a constant value of 6 mm.

The illumination of the fibers is in gray tones. Therefore, the background color is a textured gray image, while the cylinders are black in the artificial images (see Figure 8). The resulting .png image is accompanied by a .json file that contains the information required to generate the masks for each short fiber. This file can be read with a mask generator, such as the LABELME application. The process of generating 100 random images with their annotations took 2 min on a computer with i5-3470 CPU with 16 Gb RAM and GeForce GTX 750. The illumination of the fibers is in gray tones. Therefore, the background color is a textured gray image, while the cylinders are black in the arti fi cial images (see Figure 8). The resulting .png image is accompanied by a .json fi le that contains the information required to generate the masks for each short fiber. This fi le can be read with a mask generator, such as the LABELME application. The process of generating 100 random images with their annotations took 2 min on a computer with i5-3470 CPU with 16 Gb RAM and GeForce GTX 750.

Figure 8. Example of an image created by the NX API. Figure 8. Example of an image created by the NX API.

<!-- image -->

## 2.5. Neural Network Architectures Tested 2.5. Neural Network Architectures Tested

In this paper, we propose a two-stage pipeline that combines SAM and DeepLabV3+ to provide the generalizability and accuracy of the foundational SAM model and the ability to train quickly on a small amount of DeepLabV3+ data. Figure 9 shows the general data processing flowchart. In this paper, we propose a two-stage pipeline that combines SAM and DeepLabV3+ to provide the generalizability and accuracy of the foundational SAM model and the ability to train quickly on a small amount of DeepLabV3+ data. Figure 9 shows the general data processing flowchart.

Figure 9. A two-stage pipeline is proposed combining SAM and DeepLabV3+. Figure 9. Atwo-stage pipeline is proposed combining SAM and DeepLabV3+.

The first main component of the pipeline is DeepLabV3+ [34]. This is a powerful semantic segmentation model, known for its accurate pixel-by-pixel image segmentation ability. It combines a reliable object extraction tool, such as ResNet 50 or ResNet101, with an e ffi cient decoder. This architecture does an excellent job of capturing both local and global contextual information, which makes it suitable for tasks where precise object boundaries and fine details are important. An important component is the Atrous Spatial Pyramid Pooling (ASPP) module, which uses several advanced convolutions to collect The first main component of the pipeline is DeepLabV3+ [34]. This is a powerful semantic segmentation model, known for its accurate pixel-by-pixel image segmentation ability. It combines a reliable object extraction tool, such as ResNet 50 or ResNet101, with an efficient decoder. This architecture does an excellent job of capturing both local and global contextual information, which makes it suitable for tasks where precise object boundaries and fine details are important. An important component is the Atrous Spatial Pyramid Pooling (ASPP) module, which uses several advanced convolutions to collect data at multiple scales. The decoder further enhances the output by combining high-level semantic functions with precise spatial data. High-precision segmentation in various applications is realized by combining understanding of context and location. data at multiple scales. The decoder further enhances the output by combining high-level semantic functions with precise spatial data. High-precision segmentation in various applications is realized by combining understanding of context and location.

The main components of DeepLabV3+ are an encoder, an ASPP module, a decoder, and a compression and excitation mechanism (SE) (see Figure 10). The main components of DeepLabV3+ are an encoder, an ASPP module, a decoder, and a compression and excitation mechanism (SE) (see Figure 10).

Figure 10. DeepLabV3+ architecture. Figure 10. DeepLabV3+ architecture.

<!-- image -->

The choice of this architecture was due to the sufficiently high segmentation accuracy for small training samples (no more than a thousand images). The choice of this architecture was due to the sufficiently high segmentation accuracy for small training samples (no more than a thousand images).

The second main component is Segment Anything Model (SAM) [36], which is an image segmentation model developed by Meta AI. The proposed model can identify the precise location of either specific objects in an image or every object in an image. SAM was released in April 2023. SAM is a cutting-edge image segmentation model that provides The second main component is Segment Anything Model (SAM) [36], which is an image segmentation model developed by Meta AI. The proposed model can identify the precise location of either specific objects in an image or every object in an image. SAMwas released in April 2023. SAM is a cutting-edge image segmentation model that provides prompt and accurate segmentation and unparalleled versatility for image analysis tasks. SAM's advanced design allows it to adapt to new image distributions and tasks without prior knowledge, a feature known as zero-shot transfer. SAM is trained on the expansive SA-1B dataset, which contains more than 1 billion masks, spread over 11 million carefully curated images, and demonstrates impressive zero-shot performance, surpassing previously fully supervised results in many cases. A new model, SAM2 [44], was also tested. However, it did not demonstrate significant improvement in our task. The result was significantly influenced by the settings of the neural network models. We did not use fine-tuning SAM because the memory and computing resource requirements would have been significantly higher than those of the proposed pipeline. The SAM architecture is shown in Figure 11. 10 of 22 prompt and accurate segmentation and unparalleled versatility for image analysis tasks. SAM's advanced design allows it to adapt to new image distributions and tasks without prior knowledge, a feature known as zero-shot transfer. SAM is trained on the expansive SA-1B dataset, which contains more than 1 billion masks, spread over 11 million carefully curated images, and demonstrates impressive zero-shot performance, surpassing previously fully supervised results in many cases. A new model, SAM2 [44], was also tested. However, it did not demonstrate signi fi cant improvement in our task. The result was signi fi cantly influenced by the se tt ings of the neural network models. We did not use fi netuning SAM because the memory and computing resource requirements would have been signi fi cantly higher than those of the proposed pipeline. The SAM architecture is shown in Figure 11.

Figure 11. SAM architecture. Figure 11. SAM architecture.

<!-- image -->

The complexing of masks from SAM and DeepLabV3+ yielded the fi nal segmentation result, which was used to calculate statistical characteristics of fibers. The complexing of masks from SAM and DeepLabV3+ yielded the final segmentation result, which was used to calculate statistical characteristics of fibers.

The separate models of SAM, DeepLabV3+, Mask R-CNN were used to compare the segmentation accuracy. The Hough method was used as a classical method for compariThe separate models of SAM, DeepLabV3+, Mask R-CNN were used to compare the segmentation accuracy. The Hough method was used as a classical method for comparison.

Hough transform [41] is a feature extraction technique used in image analysis, computer vision, and digital image processing. The purpose of this technique is to find imperfect instances of objects within a certain class of shapes by a voting procedure. This voting procedure is carried out in a parameter space from which object candidates are obtained as local maxima in a so-called accumulator space that is explicitly constructed by the algorithm to compute the Hough transform. The classical Hough transform was concerned with identifying lines in the image. However, later, the Hough transform was extended to identifying the positions of arbitrary shapes, most commonly circles or ellipses. puter vision, and digital image processing. The purpose of this technique is to fi nd imperfect instances of objects within a certain class of shapes by a voting procedure. This voting procedure is carried out in a parameter space from which object candidates are obtained as local maxima in a so-called accumulator space that is explicitly constructed by the algorithm to compute the Hough transform. The classical Hough transform was concerned with identifying lines in the image. However, later, the Hough transform was extended to identifying the positions of arbitrary shapes, most commonly circles or ellipses.

An additional task in the proposed pipeline is to recognize metadata in electron microscope images. An example of such metadata is shown in Figure 12. The TesseractOCR model was used to implement this component. Recognition of image metadata allows us to work with different scales and universally process data at different magnifications using the development pipeline. An additional task in the proposed pipeline is to recognize metadata in electron microscope images. An example of such metadata is shown in Figure 12. The Tesseract-OCR model was used to implement this component. Recognition of image metadata allows us to work with di ff erent scales and universally process data at di ff erent magni fi cations using the development pipeline.

<!-- image -->

Figure 12. Example of metadata. Figure 12. Example of metadata.

## 2.6. Performance Metrics 2.6. Performance Metrics

We employed two commonly used metrics: Intersection over Union (IoU) [45] Pixel the segmentation performance. We employed two commonly used metrics: Intersection over Union (IoU) [45] and Pixel accuracy (PixAcc) [46] to assess the segmentation performance. The IoU metric is defined as follows:

<!-- formula-not-decoded -->

where P is the predicted segmentation mask, T is the ground-truth mask, and the cardinality is the number of pixels. An IoU of 1 corresponds to a perfect prediction, that is, a pixelperfect overlap between the predicted segmentation mask and the ground truth.

The PixAcc metric represents the proportion of correctly labeled pixels (TP) among all possible pixels (n). This is calculated as follows:

<!-- formula-not-decoded -->

Aproper distribution of each class is essential for obtaining accurate pixel accuracy. When one class occupies a disproportionately large number of pixels, the pixel accuracy results may be unreliable.

## 3. Results

## 3.1. Database

As part of this study, 37 real images of carbon fibers were captured on a scanning electron microscope and were then subjected to rotation and symmetry operations (90 ◦ , 180 ◦ , 270 ◦ ), achieving 7 extra images for each image taken. Therefore, a set of 296 real images was obtained (see Figure 13).

In addition, 500 artificial images were created. We tried to reproduce the background of the real images with a density of 15 to 20 fibers in each image (see Figure 14).

For the second series of experiments, an additional dataset containing glass fibers was also collected (Figure 15). An additional 15 images were captured using a scanning electron microscope. A set of 120 real images was also obtained with augmentation.

Figure 13. A sample of the database of real images. In addition, 500 arti fi cial of the real images with a density

Figure 14. A sample of the database of artificial images.

Figure 15. Asample of the extra database of glass fibers real images.

<!-- image -->

## 3.2. Training and Evaluation of Neural Networks

3.2. Training and Evaluation of Neural Networks The fi rst series of experiments was conducted to investigate the e ff ectiveness of using synthesized images for DeepLabV3+ training in the semantic segmentation stage. To realize this, the dataset was redistributed in three versions: the real images dataset (RID), the arti fi cial images dataset (AID), and the hybrid images dataset (HID, with 796 images). Each dataset was divided into three parts: training, validation, and testing sets at ratios of The first series of experiments was conducted to investigate the effectiveness of using synthesized images for DeepLabV3+ training in the semantic segmentation stage. To realize this, the dataset was redistributed in three versions: the real images dataset (RID), the artificial images dataset (AID), and the hybrid images dataset (HID, with 796 images). Each dataset was divided into three parts: training, validation, and testing sets at ratios of 75%, 15%, and 10%, respectively. Table 1 lists the metrics obtained using the different databases.

# Table 1. Evaluation of the different data sets with DeepLabV3+.

| Method           |   IoU |   PixAcc |
|------------------|-------|----------|
| DeepLabv3+ (RID) | 0.943 |    0.949 |
| DeepLabv3+ (AID) | 0.851 |    0.854 |
| DeepLabv3+ (HID) | 0.953 |    0.959 |

Figure 16 shows the performance of DeepLabv3+ after training with the different databases. A real image unrelated to the images that make up the database was used for this test.

Figure 16. Examples of creating fibers and background label masks using DeepLabv3+ trained with different datasets. Figure 16. Examples of creating fibers and background label masks using DeepLabv3+ trained with different datasets.

<!-- image -->

The additional use of synthesized images improved the metrics by 1% on average. It should also be noted that we obtained an interesting result when training the model exclusively on synthesized data, i.e., the segmentation ability of real fibers reached 85.1% and 85.4% according to the IoU and PixAcc metrics, respectively. This result indicates the potential for the additional use of synthetic data when extremely small or missing sets of real images are used, especially with increased modeling accuracy. The additional use of synthesized images improved the metrics by 1% on average. It should also be noted that we obtained an interesting result when training the model exclusively on synthesized data, i.e., the segmentation ability of real fibers reached 85.1% and 85.4% according to the IoU and PixAcc metrics, respectively. This result indicates the potential for the additional use of synthetic data when extremely small or missing sets of real images are used, especially with increased modeling accuracy.

The second series of experiments investigated the impact of dataset expansion on segmentation quality and the possibility of adapting the model to new fiber classes. The HID was extended using a set of 120 real images of glass fibers (HID + GF), and DeepLabv3+ was trained on 916 images. The second series of experiments investigated the impact of dataset expansion on segmentation quality and the possibility of adapting the model to new fiber classes. The HIDwasextended using a set of 120 real images of glass fibers (HID + GF), and DeepLabv3+ was trained on 916 images.

The expansion of the dataset improved the quality of carbon fiber segmentation by approximately 0.5%. The quality of the glass fiber segmentation was signi fi cantly improved. Table 2 lists the metrics obtained from the dataset expansion on di ff erent test data. Figure 17 shows the segmentation visualization results. It is worth noting that for an unknown fiber type in the model, segmentation gives a large error because of the response to complex backgrounds. The expansion of the dataset improved the quality of carbon fiber segmentation by approximately 0.5%. The quality of the glass fiber segmentation was significantly improved. Table 2 lists the metrics obtained from the dataset expansion on different test data. Figure 17 shows the segmentation visualization results. It is worth noting that for an unknown fiber type in the model, segmentation gives a large error because of the response to complex backgrounds.

# Table 2. An evaluation of the expansion of datasets.

| Method                | Test Data     |   IoU | PixAcc   |
|-----------------------|---------------|-------|----------|
| DeepLabv3+ (HID)      | Carbon fibers | 0.953 | 0.959    |
| DeepLabv3+ (HID + GF) | Carbon fibers | 0.958 | 0.963    |
| DeepLabv3+ (HID)      | Glass fibers  | 0.734 | 0.757    |
| DeepLabv3+ (HID + GF) | Glass fibers  | 0.96  | 0.963    |
| DeepLabv3+ (HID)      | All           | 0.883 | 0.892 14 |
| DeepLabv3+ (HID + GF) | All           | 0.959 | 0.963    |

Figure 17. Examples of creating the masks of the fiber and background labels using DeepLabv3+ trained with the expansion of the dataset for di ff erent fiber types. Figure 17. Examples of creating the masks of the fiber and background labels using DeepLabv3+ trained with the expansion of the dataset for different fiber types.

<!-- image -->

Thus, additional data for training the model slightly improved the quality of the old data domains but signi fi cantly improved the results for the new data domains. Thus, additional data for training the model slightly improved the quality of the old data domains but significantly improved the results for the new data domains.

A third series of experiments was conducted with the HID to determine the e ff ectiveness of the entire automatic segmentation method. Athird series of experiments was conducted with the HID to determine the effectiveness of the entire automatic segmentation method.

For the SAM model, the post-processing parameters of the training sample were selected using the grid search with IoU minimization: For the SAM model, the post-processing parameters of the training sample were selected using the grid search with IoU minimization:

-  points\_per\_side = 30, · points\_per\_side = 30,



-  pred\_iou\_thresh = 0.86, · pred\_iou\_thresh = 0.86,
- stability\_score\_thresh = 0.89,
- crop\_n\_layers = 1,
- crop\_n\_points\_downscale\_factor = 5,
- •
- min\_mask\_region\_area = 50.

The model hyperparameters were selected as follows.

1. The DeepLabv3+ model was trained.
2. DeepLabv3+ and SAM image segmentation was performed in the validation sample.
3. The segmentation results were compared with the IoU metric.
4. The SAM parameters were changed by a table search.
5. Steps 2-4 were repeated.
6. The final parameters of SAM that minimize IoU were selected. The parameter search scheme is illustrated in Figure 18. 6. The fi nal parameters of SAM that minimize IoU were selected. The parameter search scheme is illustrated in Figure 18.

Figure 18. SAM model hyperparameters search. Figure 18. SAM model hyperparameters search.

<!-- image -->

This joint training scheme of SAM and DeepLabv3+ allowed us to signi fi cantly reduce the number of false segmentations of the SAM model, as shown in Figure 19. The following architectures were compared: DeepLabv3+/SAM, DeepLabv3+/Hough, SAM (see Figure 20).

The results of the metrics for each architecture are presented in Table 3. Table 3 also presents the results of the metrics of the Mask R-CNN architecture, which was used in [29].

Table 3. An evaluation of the different datasets with DeepLabV3+.

| Architecture     |   IoU |   Pix Acc |
|------------------|-------|-----------|
| DeepLabv3+/Hough | 0.915 |     0.911 |
| SAM              | 0.873 |     0.877 |
| DeepLabv3+/SAM   | 0.953 |     0.959 |
| Mask R-CNN       | 0.723 |     0.724 |
| SAM2             | 0.877 |     0.879 |

The obtained results demonstrate that the pure SAM model yielded quite a lot of false positives, which was a direct consequence of the universality of the model. Presegmentation with DeepLabV3+ eliminated most of the false positives of the SAM model and significantly improved the metrics due to this. DeepLabV3+/Hough demonstrated average results due to the insufficiently accurate transition from the found lines to the segments of individual fiber instances. The average total processing time was 0.95 s for a single fiber image. This joint training scheme of SAM and DeepLabv3+ allowed us to signi fi cantly reduce the number of false segmentations of the SAM model, as shown in Figure 19.

Figure 19. SAM model results ( a ) default parameters, ( b ) optimized parameters. Figure 19. SAM model results ( a ) default parameters, ( b ) optimized parameters.

<!-- image -->

The following

architectures were

compared:

DeepLabv3+/SAM,

Figure 20. A qualitative comparison ( a ) Input image ( b ) Manual segmentation ( c ) DeepLabv3+ ( d ) DeepLabv3+/Hough, ( e ) SAM, ( f ) DeepLabv3+/SAM. Figure 20. A qualitative comparison ( a ) Input image ( b ) Manual segmentation ( c ) DeepLabv3+ ( d ) DeepLabv3+/Hough, ( e ) SAM, ( f ) DeepLabv3+/SAM.

Table 4 compares the training time and inference values of the different models. We used the results reported in [40] for the SAM model. The average time was obtained because of the experiments for the other models. For DeepLabv3+/SAM, the training time of DeepLabv3+ and the search parameters time for the SAM model were summed. 

# Table 4. Performance comparison for different models.

Inference

| Architecture Architecture                      | Training Time (min) Training Time (min)   | Inference Time (ms) (ms)   | GPU/CPU GPU/CPU                                           |
|------------------------------------------------|-------------------------------------------|----------------------------|-----------------------------------------------------------|
| DeepLabv3+/Hough DeepLabv3+/Hough              | ~20 ~20                                   | 127 127                    | RTX 4090/Ryzen 9 4 NVIDIA RTX 4090/Ryzen 9 4 NVIDIA       |
| SAM (fine-tuning [38]) SAM( fi ne-tuning [38]) | ~180 ~180                                 | ~900 ~900                  | Tesla K80 GPUs, 24 CPU cores Tesla K80 GPUs, 24 CPU cores |
| DeepLabv3+/SAM DeepLabv3+/SAM                  | ~30 ~30                                   | 951 951                    | RTX 4090/Ryzen 9 RTX 4090/Ryzen 9                         |
| DeepLabv3+ DeepLabv3+                          | ~20 ~20                                   | 51 51                      | RTX 4090/Ryzen 9 RTX 4090/Ryzen 9                         |

The results demonstrate that the DeepLabv3+/SAM architecture was trained several times more rapidly with lower requirements for computing resources than fine-tuning the SAM model. The results demonstrate that the DeepLabv3+/SAM architecture was trained several times more rapidly with lower requirements for computing resources than fi ne-tuning the SAM model.

Experiments were also conducted with measurements of the geometric parameters of the fibers: quantity, length, and thickness, and the ratio of length to thickness. Figure 21 shows histograms of the distributions of length, thickness, and length-to-thickness ratios for fibers with automatic and manual segmentation. Ten real fiber images were used without augmentation to calculate the statistics. The average total processing time was 0.95 s for a single fiber image. The training time of the model was less than 30 min on a Nvidia 4090 RTX. Experiments were also conducted with measurements of the geometric parameters of the fibers: quantity, length, and thickness, and the ratio of length to thickness. Figure 21 shows histograms of the distributions of length, thickness, and length-to-thickness ratios for fibers with automatic and manual segmentation. Ten real fiber images were used without augmentation to calculate the statistics. The average total processing time was 0.95 s for a single fiber image. The training time of the model was less than 30 min on a Nvidia 4090 RTX.

Figure 21. Histograms of the distribution of lengths, thicknesses, and length-to-thickness ratios blue-automatic segmentation, green-manual segmentation. Figure 21. Histograms of the distribution of lengths, thicknesses, and length-to-thickness ratios blue-automatic segmentation, green-manual segmentation.

<!-- image -->

To test the generalizability of the proposed pipeline and the possible bias due to the use of synthetic data, images of glass fibers were collected separately. Glass fibers di ff er in texture and background from carbon fiber images. The qualitative segmentation results obtained by the proposed pipeline and fiber length determination are shown in Figure 22. For comparison, glass fibers were measured manually using electron microscope software (see Figure 23). Figure 24 illustrates the capabilities of a pipeline under the challenging conditions of intersecting fibers, high fiber density, and large amounts of foreign ma tt er. No signi fi cant correlation was found between the fiber density in the image (in the range from 64 to 274 fibers per image) and the quality of segmentation using the IoU metric (in the range from 0.955 to 0.990). To test the generalizability of the proposed pipeline and the possible bias due to the use of synthetic data, images of glass fibers were collected separately. Glass fibers differ in texture and background from carbon fiber images. The qualitative segmentation results obtained by the proposed pipeline and fiber length determination are shown in Figure 22. For comparison, glass fibers were measured manually using electron microscope software (see Figure 23). Figure 24 illustrates the capabilities of a pipeline under the challenging conditions of intersecting fibers, high fiber density, and large amounts of foreign matter. No significant correlation was found between the fiber density in the image (in the range from 64 to 274 fibers per image) and the quality of segmentation using the IoU metric (in the range from 0.955 to 0.990).

Figure 22. Glass fiber automatic segmentation with fiber length detection in micrometers. Figure 22. Glass fiber automatic segmentation with fiber length detection in micrometers. Figure 22. Glass fiber automatic segmentation with fiber length detection in micrometers.

<!-- image -->

Figure 23. Manual glass fiber measurement. Figure 23. Manual glass fiber measurement.

<!-- image -->

Figure 24. Challenging conditions for fiber segmentation: ( a ) intersecting fibers, ( b ) high fiber density, ( c ) large amounts of foreign ma tt er. Figure 24. Challenging conditions for fiber segmentation: ( a ) intersecting fibers, ( b ) high fiber density, ( c ) large amounts of foreign matter.

<!-- image -->

## 4. Conclusions 4. Conclusions

A two-stage pipeline for the automatic recognition and determination of the geometric characteristics of carbon and glass fibers is proposed. In the fi rst stage, the fundamental neural network model of segmentation SAM (Segment Anything Model) was used, and in the second stage, the segmentation results were improved using the DeepLabv3+ neural network model. The use of synthetic data for training improved the segmentation quality for the IoU and Pix Acc metrics from 0.943 and 0.949 to 0.953 and 0.959, i.e., an average of 1%. The average total processing time was 0.95 s for a single fiber image. An automatic calculation of geometric features and statistics that can be used to evaluate material properties was realized. Atwo-stage pipeline for the automatic recognition and determination of the geometric characteristics of carbon and glass fibers is proposed. In the first stage, the fundamental neural network model of segmentation SAM (Segment Anything Model) was used, and in the second stage, the segmentation results were improved using the DeepLabv3+ neural network model. The use of synthetic data for training improved the segmentation quality for the IoU and Pix Acc metrics from 0.943 and 0.949 to 0.953 and 0.959, i.e., an average of 1%. The average total processing time was 0.95 s for a single fiber image. An automatic calculation of geometric features and statistics that can be used to evaluate material properties was realized.

The primary contributions of this paper are summarized as follows: The primary contributions of this paper are summarized as follows:

1. The two-stage pipeline combining SAM and DeepLabV3+ provides the generalizability and accuracy of the foundational SAM model and the ability to quickly train on a small amount of data of the DeepLabV3+ model. The pipeline was trained several times more rapidly with lower requirements for computing resources than fi netuning the SAM model, with a comparable inference time. 1. The two-stage pipeline combining SAM and DeepLabV3+ provides the generalizability and accuracy of the foundational SAM model and the ability to quickly train on a small amount of data of the DeepLabV3+ model. The pipeline was trained several times more rapidly with lower requirements for computing resources than fine-tuning the SAM model, with a comparable inference time.
2. End-to-end technology for processing images of electron microscope fibers, from images with metadata to statistics of the distribution of geometric characteristics of fi -bers. The result of this work is statistical data on the distribution of geometric characteristics of fibers, which are of great practical importance for modeling the physical characteristics of materials. 2. End-to-end technology for processing images of electron microscope fibers, from images with metadata to statistics of the distribution of geometric characteristics of fibers. The result of this work is statistical data on the distribution of geometric characteristics of fibers, which are of great practical importance for modeling the physical characteristics of materials.
3. A few-shot training procedure for the DeepLabV3+/SAM pipeline, combining training of the DeepLabV3+ model weights and SAM model parameters, allowed pipeline training using only 37 real labeled images. The pipeline was then adapted to a new type of fiber and background using 15 additional real labeled images. 3. Afew-shot training procedure for the DeepLabV3+/SAM pipeline, combining training of the DeepLabV3+ model weights and SAM model parameters, allowed pipeline training using only 37 real labeled images. The pipeline was then adapted to a new type of fiber and background using 15 additional real labeled images.
4. A method to generate synthetic data for additional training of neural networks for fiber segmentation allowed us to further improve the segmentation quality by 1%. 4. A method to generate synthetic data for additional training of neural networks for fiber segmentation allowed us to further improve the segmentation quality by 1%.

The developed pipeline allowed us to signi fi cantly reduce the time required for fiber length evaluations in scanning electron microscope images, thereby increasing the accuracy of statistical data collection due to the multiple-fold increase in the amount of processed experimental material. This technique may also be useful in other industrial appliThe developed pipeline allowed us to significantly reduce the time required for fiber length evaluations in scanning electron microscope images, thereby increasing the accuracy of statistical data collection due to the multiple-fold increase in the amount of processed experimental material. This technique may also be useful in other industrial applications.