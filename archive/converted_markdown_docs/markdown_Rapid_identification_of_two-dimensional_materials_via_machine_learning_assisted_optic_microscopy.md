<!-- image -->

Contents lists available at ScienceDirect

## Journal of Materiomics

j o urnal homepage: www.journals.elsevier.com/journal-of-materiomics/

## Rapid identi /uniFB01 cation of two-dimensional materials via machine learning assisted optic microscopy

Yuhao Li b, c, 1 , Yangyang Kong a, b, 1 , Jinlin Peng b, d, 1 , Chuanbin Yu b, c , Zhi Li b, c , Penghui Li e , Yunya Liu d , Cun-Fa Gao c, ** , Rong Wu a, *

a School of Physics Science and Technology, Xinjiang University, Urumqi, 830046, Xinjiang, China

b Shenzhen Key Laboratory of Nanobiomechanics, Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, Shenzhen, 518055, Guangdong, China

c State Key Laboratory of Mechanics and Control of Mechanical Structures, Nanjing University of Aeronautics &amp; Astronautics, Nanjing, 210016, Jiangsu, China

d Key Laboratory of Low Dimensional Materials and Application Technology of Ministry of Education, Xiangtan University, Xiangtan, 411105, Hunan, China

e Institute of Biomedicine and Biotechnology, Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, Shenzhen, 518055, Guangdong, China

## a r t i c l e i n f o

Article history: Received 8 January 2019 Received in revised form 17 March 2019 Accepted 19 March 2019

Available online 28 March 2019

Keywords: Two-dimensional materials Optical contrast Total color difference Red-green-blue K -means clustering K -nearest neighbors ( k -NN)

## 1. Introduction

Since the /uniFB01 rst successful exfoliation of graphene in 2004 [1], atomically thin two-dimensional (2D) materials such as graphene [2], hexagonal boron nitride (h-BN) [3], transition metal dichalcogenides (TMDs) [4] and black phosphorus (BP) [5] have generated great excitement because of their unique and exotic functionalities [6]. These 2D materials often exhibit high crystalline quality, ballistic transport [7], excellent mechanical properties [8 e 10], and quantized conductance of size-con /uniFB01 ned Dirac fermions [11], and their unusual band structures and size effect allow researchers to probe a wide range of phenomena, including quantum valley and

* Corresponding author.

** Corresponding author.

E-mail addresses: cfgao@nuaa.edu.cn (C.-F. Gao), wurongxju@sina.com (R. Wu). Peer review under responsibility of The Chinese Ceramic Society.

1 These authors contributed equally to this work.

## a b s t r a c t

A combination of Fresnel law and machine learning method is proposed to identify the layer counts of 2D materials. Three indexes, which are optical contrast, red-green-blue, total color difference, are presented to illustrate and simulate the visibility of 2D materials on Si/SiO2 substrate, and the machine learning algorithms, which are k -mean clustering and k -nearest neighbors, are employed to obtain thickness database of 2D material and test the optical images of 2D materials via red-green-blue index. The results show that this method can provide fast, accurate and large-area property of 2D material. With the combination of arti /uniFB01 cial intelligence and nanoscience, this machine learning assisted method eases the workload and promotes fundamental research of 2D materials.

© 2019 The Chinese Ceramic Society. Production and hosting by Elsevier B.V. This is an open access article under the CC BY-NC-ND license (http://creativecommons.org/licenses/by-nc-nd/4.0/).

spin Hall effect [12 e 14], valley magnetoelectricity [15], giant tunneling magnetoresistance [16], light-emitting diodes [17], superconductivity [18], Li-ion battery [19 e 21] and biosensing [22]. Recently, 2D piezoelectricity, ferroelectricity and magnetism have also attracted great attention ferromagnetic [23 e 25]. These fascinating functional behavior of 2D materials are usually correlated with their layer counts, and a small change of which may result in striking difference in their properties [24,26,27]. As such, rapid and accurate identi /uniFB01 cation of layer count for 2D materials on a large scale is of signi /uniFB01 cant interests.

Atomic force microscopy (AFM) can be used to accurately determine the thickness of 2D materials, yet it is time-consuming with limited scan range, not suitable for rapid determination on large area. Other methods have also been adopted to identify the thickness of 2D materials, including Raman spectroscopy [28,29], scanning electron microscopy (SEM) and high resolution transmission electron microscopy (HRTEM). However, the weak thickness dependence of the Raman modes makes it dif /uniFB01 cult for an

<!-- image -->

<!-- image -->

unambiguous and accurate determination, especially for distinguishing few-layers of 2D materials [30], and it is not easily accessible either. Optical microscopy, on the other hand, is a simple, ef /uniFB01 cient and nondestructive method that can be widely applied to 2D materials over a large area [31 e 34], and several methods have been proposed to improve the resulted optical contrast, such as using narrow band illumination and selecting appropriate substrates [31,35,36]. With the emergence of arti /uniFB01 cial intelligence [37 e 40] application of machine learning algorithm in image analysis for identifying 2D materials also becomes viable [41], which could reduce human labors while improve ef /uniFB01 ciency and accuracy.

Here, we seek to develop a machine learning assisted method for rapid identi /uniFB01 cation of 2D materials over large scale. Built on previous works, we adopt three effective indexes, including optical contrast (OC), total color difference (TCD) and red-green-blue (RGB), to help us /uniFB01 nd the optimum substrates and identify the layer counts of 2D materials deposited on a substrate via optic microscopy. First, we utilize OC and TCD to identify the appropriate substrates that maximize the difference not only between 2D material and the substrate, but also among 2D material of different layers. Then, using graphene and MoS2 on Si/SiO2 substrate under the illumination of halogen lamp in an optical microscope as examples, we demonstrate learning process of the layer counts of 2D materials over large area based on k -means clustering [42] via RGB. Finally, based on consequent results, a supervised machinelearning methods, which is k -nearest neighbors ( k -NN) [43], was used to accomplish the identi /uniFB01 cation of 2D materials. This method is applicable to a wide range of 2D materials and substrates.

## 2. Materials and methods

Sample preparation . Few-layer of graphene and MoS2 /uniFB02 akes were mechanically exfoliated from commercially available highly oriented pyrolytic graphite (HOPG) and MoS2 crystals onto thermal oxide Si/SiO2 substrate using Scotch tape as the transfer medium. Before exfoliating, the substrates were ultrasonically cleaned in acetone, ethanol and DI water, respectively, and then subjected to oxygen plasma for 10 min to remove ambient adsorbates from surface. The mechanical exfoliation process consists of following steps. The tape is pressed into contact with HOPG and MoS2 crystals by thumb /uniFB01 rst, and the thick graphene and MoS2 /uniFB02 akes are stuck on the tape. The process is then repeated four or /uniFB01 ve times to thin the /uniFB02 akes, after which the substrate is pressed onto the tape with thin graphene and MoS2 /uniFB02 akes for transferring. Finally, the tape is removed from the substrates at approximately constant speed when the sample temperature is dropped to ambient temperature, which completes the exfoliation process, as illustrated in Fig. S1 (a)(f).

Sample characterizations . The optical images of 2D materials were captured by a metallurgical microscopy equipped with CCD and the topography of 2D materials was examined with an AFM (Asylum Research Cypher ES) using AC240-TS-R3 probe. Raman scattering measurement was performed on a Horiba Jobin-Yvon based on the different frequency of characteristic vibrations. A XYZ motorized sample stage controlled by LabSpec software was used to move sample accurately and the output power was controlled by neutral density /uniFB01 lters to protect the graphene and MoS2 samples.

K-means Clustering . The k -means clustering, one of the most practical methods of machine learning, was employed in our analysis. This method can partition samples or variables into clusters on the basis of similarity or its converse [42]. From those clusters, the characters of samples or variables can be unveiled. In this paper, k -means clustering without supervision was used to analyze optical images of 2D materials. A known set f x i ; i ¼ 1 ; 2 ; / ; n g can be clustered into k sets based on their similarity and the k kernels of clustering f m j ; j ¼ 1 ; 2 ; / ; k g are needed for every cycle. For the /uniFB01 rst cycle, the k kernels of clustering m j are generated at random. And then, the data x i are distributed into k newsets f c j ; j ¼ 1 ; 2 ; / ; k g based on the minimum principle of /C13 /C13 /C13 x i /C0 m j /C13 /C13 /C13 2 . In the next cycle, the new kernels of clustering m j are regenerated by m j ¼ mean ðf x i 2 c j gÞ . Based on new kernels of clustering m j and the minimum principle of /C13 /C13 /C13 x i /C0 m j /C13 /C13 /C13 2 , the data x i can be redistributed into k new sets c j and the kernels of clustering m j are also deduced. The k -means clustering completed until the kernels of clustering m j keep constant and sum of variance J ¼ P k j ¼ 1 P xi 2 cj /C12 /C12 /C12 /C12 /C12 /C12 x i /C0 m j /C12 /C12 /C12 j 2 is minimized. In our implementation, Python is adopted for the k -means clustering. The clustering number is determined by AFM topography results of related sample.

K-nearest neighbors. K -nearest neighbors method is one of the simplest machine learning algorithms, which calculates the distance between samples and training results and choses k ( /C21 1) neighbors to do clustering. In general, k -NN is a supervised methods which needs plenty of training data to calibrate. Hence, we use k -mean clustering to produce database follow by a testing process executed by k -NN which can classify and identify a new mix-layer optical image of 2D materials. In this paper, we use the toolbox in Matlab to implement the k -NN method.

## 3. Results and discussion

Optical Contrast . The image contrast between 2D materials and substrate under visible light origins from integrated contrasts of each wavelength component. To study the relationship between the observed colors and layer counts of 2D materials, a re /uniFB02 ection model based on Fresnel law has been adopted [31,32,34,35,44,45]. The model system consists of a dielectric /uniFB01 lm on silicon wafer and a 2D material on the dielectric /uniFB01 lm, as shown in Fig. 1(a). For the bare dielectric /uniFB01 lm on the wafer, there are two interfaces, while for the triple-material system involving 2D material, there is an additional interface. The total re /uniFB02 ected light is a beam resulted from all the optical paths, determined by the wavelength of the incident light ( l ), their incident angle ( q ), refractive indices of all materials ( n ) and their thickness ( h ). For the triple-material system, therefore, the wavelength dependent total re /uniFB02 ectivity of beam can be written as [46].

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

LabRam HR-VIS high-resolution confocal Raman microscope equipped with a 532-nm laser as the excitation source at room temperature to identify layer thickness of graphene and MoS2

where

Fig. 1. (a) Optical re /uniFB02 ection and transmission schematic for a layered thin/uniFB01 lm system; (b) CIE color-matching functions (CIE 1931 2 /C14 observer); (c) normalized total light source power of halogen lamp.

<!-- image -->

<!-- formula-not-decoded -->

are relative indexes of refraction and

<!-- formula-not-decoded -->

are the phase shifts because of paths change of light. Here, n 0, n 1, n 2 and n 3 are the refractive indexes of air ( n 0 ¼ 1), 2D materials, dielectric /uniFB01 lm and silicon, respectively, which are all dependent on the wavelength, while h 1 and h 2 are the thickness of 2D material and dielectric /uniFB01 lm, with q 1 and q 2 being incident angles which are /uniFB01 xed at 0 /C14 in our analysis. For the bare dielectric /uniFB01 lm in absence of 2D material, the re /uniFB02 ection is calculated by using n 1 ¼ n 0 ¼ 1.

In order to distinguish 2D materials from their substrate, the OC is de /uniFB01 ned as the relative intensity of re /uniFB02 ected light in the presence ( n 1 s 1) and absence ( n 1 ¼ n 0 ¼ 1) of the 2D material [34,47].

<!-- formula-not-decoded -->

To determine the most optimal Si/SiO2 substrate for distinguishing ultrathin 2D materials, the OC color maps of monolayer graphene and MoS2 have been calculated as a function of the illumination wavelength and the thickness of SiO2, as shown in Fig. 2(a) and (b), respectively. It is clearly seen that only a few selective dielectric /uniFB01 lm thicknesses are suitable for detecting monolayer graphene and MoS2. Due to the maximum sensitivity of human eye [48], we de /uniFB01 ne optimal substrates for the optical identi /uniFB01 cation as those with a SiO2 thickness that shows large OC for l ¼ 550 nm. Under this condition, Si wafers with a SiO2 layer of 90 nm and 270 nm are optimal for the optical identi /uniFB01 cation of monolayer graphene and MoS2.

To further distinguish graphene and MoS2 with different layer counts, the OC between 2D material (graphene or MoS2) and substrate is calculated as a function of wavelength and layer number for a different SiO2 thickness. Fig. 3(a) and (b) show the calculated OC betweengrapheneand100 nmand270nmSiO2,respectively,while Fig. 3(c) and (d) show those between MoS2 and 100 and 300 nm SiO2, respectively. These results suggest that OC increases with its layer number for both graphene and MoS2, consistent with previous reports [32,44]. For graphene deposited on 100 nm and 270 nm SiO2, Fig. 3(a) and (b) show that both the absolute value of OC for 1L-8L graphene and the difference among 1L e 8L graphene peak at the wavelengthof650 nmand550 nm,respectively.ForMoS2 deposited on100 nmand300nmSi/SiO2 substrate, on the other hand, Fig. 3(c) and (d) show that MoS2 observed under 610 nm monochromatic light present the best visibility. More OC results about graphene and MoS2 deposited on Si/SiO2 can be found in Fig. S2.

Total color difference . The visibility of human eye is actually determined from the integration of contrast for each wavelength component instead of under monochromatic light. In order to distinguish between sample and substrate under such context, the CIE 1976 L * a * b * color space [31] is adopted to illustrate the color difference. The total color difference (TCD) is given as following [31,46].

<!-- formula-not-decoded -->

where D L * is called lightness difference that represents the contrast without the effect of color factors and is similar to that presented under monochromatic light, and D a * and D b * are color difference, with

<!-- formula-not-decoded -->

Fig. 2. Numerical simulations of graphene and MoS2 deposited on Si/SiO2 . Calculated OC color maps for monolayer graphene (a) and MoS2 (b) as a function of the illumination wavelength and SiO2 thickness; calculated TCD contour maps for graphene (c) and MoS2 (d) as a function of the layer number and SiO2 thickness.

<!-- image -->

where X 0, Y 0 and Z 0 are tristimulus components of CIE standard illuminance; L * is called psychological lightness, and a * and b * are psychological chroma. X , Y and Z tristimulus components calculated by integrating its wavelength spectrum k multiplying the colormatching functions ( x ( l ), y ( l ), z ( l )) [46,49].

<!-- formula-not-decoded -->

where x ( l ), y ( l ) and z ( l ) are CIE color-matching functions [50] shown in Fig. 1(b); S ( l ) is the total light source power of incident light, which is the spectrum of light used in our optical microscope, as shown in Fig. 1(c).

In order to illustrate the visibility of 2D materials, TDC for graphene and MoS2 are shown in Fig. 2(c) and (d), respectively, which is a function of the layer count and SiO2 thickness. According to the National Bureau of Standards, the images can be distinguished if the value of TCD is more than 1.5, the threshold for visibility for naked eyes [31]. Based on this criterion, our numerical results suggest that the desirable thicknesses of SiO2 are 100 nm and

300 nm for observation of monolayer graphene and MoS2. Therefore, Si wafer with 90 nm, 100 nm, 270 nm and 300 nm SiO2 are chosen as the substrate of graphene and MoS2 for further study.

Red-Green-Blue Based Machine Learning . The total re /uniFB02 ected light could be transformed into a device-dependent color scheme, such as RGB, which is a color scheme of a computer monitor and can be directly observed by naked eyes, de /uniFB01 ned as [46,49].

<!-- formula-not-decoded -->

where M is a 3 /C2 3 transformation matrix that is generated by the monitor's phosphor chromaticity coordinates and the reference white of the light source [31,34]. The calculated RGB values for graphene and MoS2 deposited on four kinds of substrates are presented in Fig. S3. Because of the high /uniFB01 delity of RGB, it is almost impossible to mix one picture with another. Therefore, RGB is much suitable than grayscale in handling this problem.

Different kinds of RGB intensities related to different thickness of 2D material as described in previous section and thus make optical identi /uniFB01 cation possible. Machine learning algorithm based on k -means clustering and k -NN is then developed to carry out machine learning of 2D materials via RGB in two processes, as shown in Fig. 4, which are training and testing process. For the

Fig. 3. Calculated OC of graphene and MoS2 deposited on Si/SiO2 . Calculated OC between 1L and 8L graphene and Si/SiO2 substrate as functions of wavelength for 100 nm (a) and 270 nm (b) thickness of SiO2; and calculated OC between 1L and 8L MoS2 and Si/SiO2 substrate as functions of wavelength for 100 nm (c) and 300 nm (d) thickness of SiO2.

<!-- image -->

Fig. 4. The schematic illustration of the machine learning assisted identi /uniFB01 cation. In the training process, from optical image (a) of 2D material, the RGB values and coordinates of every pixel are obtained; analyzing RGB values by k -means clustering, 4 clusters (b) are deduced; combining corresponding AFM results (c) of the sample with the reconstructed optical image based on the clustered data, the training processes of 2D material (d) is completed. For the testing process, using optical image (e) of 2D material and above produced data bank, the layer identi /uniFB01 ed image (f) is obtained.

<!-- image -->

training process, the RGB values and coordinates are obtained for every pixel from optical image of 2D material, and the k -means clustering is employed to identify their similarities and characteristics, detailed in Material and Method part. As presented in Fig. 4(b), the RGB information of 2D material is clustered into four categories by k -means clustering. Combining the clustered categories RGB values with their corresponding coordinates, the classi /uniFB01 ed RGB of 2D material can be reconstructed. Simultaneously, based on the AFM results shown in Fig. 4(c), the areas of 2D material are linked to their thickness, as shown in Fig. 4(d). For the testing process, k -NN method is chosen to identify the layer counts of 2D materials because of its capacity of dealing with complex patterns. Based on the data bank of training process, the test sample, shown in Fig. 4(e), can be distinguish into two kinds of layer count, as shown in Fig. 4(f). These algorithms turn tiny optical difference among layers of 2D material into different categories and makes optical identi /uniFB01 cation more accurate and ef /uniFB01 cient.

Identi /uniFB01 cation of Graphene . The machine learning assisted identi /uniFB01 cation of graphene samples is shown in Fig. 5. To guide the experimental identi /uniFB01 cation, a colorbar representing 1L e 125L graphene and substrate is also given based on the numerically calculated RGB values. The colorbar for the graphene deposited on 100 nm Si/SiO2 is shown in Fig. 5(e), varying with layer counts from light to dark green for graphene from 1L to 20L, and from dark green to light yellow for graphene from 20L to 125L. In the training process, graphene are deposited on 100 nm Si/SiO2 substrates by mechanical exfoliation from highly oriented pyrolytic graphite, as detailed in Fig. S1, and the training optic image is shown in Fig. 5(a). Using optical image and AFM results (Fig. 5(c)), the training result demonstrated in Fig. 5(b) shows clear boundaries of 2D materials and present a one-to-one correspondence with optical image from 1L to 8L. Those results are processed as a database with different kinds of RGB intensity linked to the sample thickness and further analyzed by the k -NN algorithm to setup a training model. More training results about graphene deposited on 90 nm, 270 nm and 300 nm Si/SiO2 substrate can be found in Fig. S4. Though the contrast between 3L and 4L is rather dif /uniFB01 cult to distinguish in optical image by naked eyes (Fig. 5(a)), the machine learning is capable of recognizing this difference clearly, as shown in Fig. 5(b). This provide an ef /uniFB01 cient and accurate method to distinguish graphene deposited on Si/SiO2 substrate over large area. The corresponding Raman spectra are presented in Fig. 5(d) for 100 nm Si/ SiO2, which con /uniFB01 rm the thickness of graphene. For the testing process, a mixed-layer graphene sample is chosen as demonstrated in Fig. 5(f). Using k -NN method and the above-mentioned database, the photo is automatically referred to its layer by analyzing the RGB information, as shown in Fig. 5(g). And, the layer count can be /uniFB01 nd in corresponding AFM result shown in Fig. 5(h) and (i), which con /uniFB01 rm the machine learning results. Those results demonstrate that this machine learning method is effective to identify the layer count of graphene. What about other 2D materials?

Identi /uniFB01 cation of MoS2 . As a whole, this method works for all the case that different RGB intensity can re /uniFB02 ect different thickness information of 2D material. Therefore, it has also be employed to identify 2D MoS2, as shown in Fig. 6. The colorbar for the MoS2 deposited on 300 nm Si/SiO2 is also shown in Fig. 6(e), varying with layer counts from light to dark green for graphene from 1L to 20L, and from dark green to light yellow for graphene from 20L to 125L. In training process, the training result demonstrated in Fig. 6(b) shows that can distinguish 1L, 2L, 3L and 4L from each other based on optical image in Fig. 6(a) and AFM results (Fig. 6(c)). In particular, it is noted that in Fig. 6(a) the boundary of MoS2 is not very clear, yet the machine learning identify a clear boundary for the 2L and 4L MoS2. Also, the Raman spectra of multilayer MoS2 are also

Fig. 5. Identi /uniFB01 cation of graphene. Optical image of graphene (a) under white illumination, with the corresponding k -mean clustering training results (b), AFM topography (c), and Raman spectra (d) of multilayered graphene deposited on 100 nm Si/SiO2, respectively. Classical numerical colorbar (e) for graphene from 1L to 125L are also formed. Optical image of graphene (f) under white illumination for the test, with the related k -NN test results (g) based on the training results shown in (b) and AFM results (h) and (i) verify the accuracy of k -NN result.

<!-- image -->

Fig. 6. Identi /uniFB01 cation of MoS2 . Optical image of MoS2 (a) under white illumination, with the corresponding k -mean clustering training results (b), AFM topography (c), and Raman spectra (d) of multilayered MoS2 deposited on 300 nm Si/SiO2, respectively. Classical numerical colorbar (e) for MoS2 from 1L to 125L are also formed. Optical image of MoS2 (f) under white illumination for the test, with the related k -NN test results (g) based on the training results shown in (b) and AFM results (h) and (i) verify the accuracy of k -NN result.

<!-- image -->

presented to con /uniFB01 rm the layer counts in Fig. 6(d). Finally, the k -NN method and the above-mentioned data bank are used to identify the layer count of MoS2 via the RGB information, as shown in Fig. 6(g). And, the con /uniFB01 rmation of layer count can be /uniFB01 nd in corresponding AFM result shown in Fig. 6(h) and (i). More training results about MoS2 deposited on 90 nm,100 nm and 270 nm Si/SiO2 substrate can be found in Fig. S5.

## 4. Concluding remarks

In summary, we have successfully established the relationship among thickness, OC, RGB and TCD of 2D materials, taking graphene and MoS2 as examples, by using the re /uniFB02 ection model based on Fresnel law, and also employed the machine learning approach to analyze the optical images of 2D materials. The potential optimal Si substrates, which are 90 nm oxide layer, 100 nm oxide layer, 270nm oxide layer and 300 nm oxide layer, are given for graphene and MoS2 by comparing OC and TCD indexes. And then, experiments of graphene and MoS2 which deposited on four kinds of substrate are presented to con /uniFB01 rm the numerical results and carry out machine learning processes. The machine learning approach, which based on k -mean clustering, is employed to investigate the optical images of graphene and MoS2 and can distinguish tiny color difference among layers of 2D material and reconstruct the optical images via the intensity of RGB. Meanwhile, the AFM and Raman spectra results are also given to provide the thickness of 2D materials which makes up a data bank for determining layer count of 2D materials. Finally, the k -NN method is employed to identify the layer count of sample, automatically, based on the corresponding database of thickness of 2D material. By introducing machine learning approach into researching 2D materials, this machine learning assisted method can promote and accelerate the fundamental study and applications of 2D materials.

## Author contributions

R.W., C.G., and Y.L. conceived the project and designed the experiments. Y.K. and R.W. performed the material fabrication. Y.K. and Y.L. carried out the AFM experiments and analyses. Y.L., Y.K and P.L. carried out the Raman experiments and analyses. Y.L., C.Y. and Z.L. carried out all numerical calculations. J.P and Y.L carried out the machine learning calculation. Y.L., C.G. and R.W. co-wrote the paper, and all authors contributed to the discussions and preparation of the manuscript.

## Competing interests

The authors declare no competing interests.

## Acknowledgements

We acknowledge National Key Research and Development Program of China (2016YFA0201001), National Natural Science Foundation of China (11627801, 11472130, 11872203, and 11572276), Shenzhen Science and Technology Innovation Committee (JCYJ20170818160815002), Shenzhen Science and Technology Research Funding (JCYJ20160608141439330), Natural Science Foundation of Xinjiang (2017D01C055), and Wuhan University of Technology (2018-KF-14). We sincerely thanks Pro. Jiangyu Li from University of Washington for helpful discussion.

## Appendix A. Supplementary data

Supplementary data to this article can be found online at

https://doi.org/10.1016/j.jmat.2019.03.003.

## References

- [1] Novoselov KS, Geim AK, Morozov SV, Jiang D, Zhang Y, Dubonos SV, et al. Electric /uniFB01 eld effect in atomically thin carbon /uniFB01 lms. Science 2004;306:666 e 9. https://doi.org/10.1126/science.1102896.
- [2] Novoselov KS, Fal'ko VI, Colombo L, Gellert PR, Schwab MG, Kim K. A roadmap for graphene. Nature 2012;490:192 e 200. https://doi.org/10.1038/ nature11458.
- [3] Dean CR, Young AF, Meric I, Lee C, Wang L, Sorgenfrei S, et al. Boron nitride substrates for high-quality graphene electronics. Nat Nanotechnol 2010;5: 722 e 6. https://doi.org/10.1038/NNANO.2010.172.
- [4] Wang QH, Kalantar-Zadeh K, Kis A, Coleman JN, Strano MS. Electronics and optoelectronics of two-dimensional transition metal dichalcogenides. Nat Nanotechnol 2012;7:699 e 712. https://doi.org/10.1038/nnano.2012.193.
- [5] Yi Y, Yu X-F, Zhou W, Wang J, Chu PK. Two-dimensional black phosphorus: synthesis, modi /uniFB01 cation, properties, and applications. Mat. Sci. Eng. R 2017;120:1 e 33. https://doi.org/10.1016/j.mser.2017.08.001.
- [6] Tan C, Cao X, Wu XJ, He Q, Yang J, Zhang X, et al. Recent advances in ultrathin two-dimensional nanomaterials. Chem Rev 2017;117:6225 e 331. https://doi. org/10.1021/acs.chemrev.6b00558.
- [7] Bandurin DA, Tyurnina AV, Yu GL, Mishchenko A, Zolyomi V, Morozov SV, et al. High electron mobility, quantum Hall effect and anomalous optical response in atomically thin InSe. Nat Nanotechnol 2017;12:223 e 7. https:// doi.org/10.1038/nnano.2016.242.
- [8] Lee C, Wei X, Kysar JW, Hone J. Measurement of the elastic properties and intrinsic strength of monolayer graphene. Science 2008;321:385 e 8. https:// doi.org/10.1126/science.1157996.
- [9] Bertolazzi S, Brivio J, Kis A. Stretching and breaking of ultrathin MoS2. ACS Nano 2011;5:9703 e 9. https://doi.org/10.1021/nn203879f.
- [10] Zhang P, Ma L, Fan F, Zeng Z, Peng C, Loya PE, et al. Fracture toughness of graphene. Nat Commun 2014;5:3782. https://doi.org/10.1038/ncomms4782.
- [11] Wang K, De Greve K, Jauregui LA, Sushko A, High A, Zhou Y, et al. Electrical control of charged carriers and excitons in atomically thin materials. Nat Nanotechnol 2018;13:128 e 32. https://doi.org/10.1038/s41565-017-0030-x.
- [12] Xiao D, Liu GB, Feng W, Xu X, Yao W. Coupled spin and valley physics in monolayers of MoS2 and other group-VI dichalcogenides. Phys Rev Lett 2012;108:196802. https://doi.org/10.1103/PhysRevLett.108.196802.
- [13] Lee J, Mak KF, Shan J. Electrical control of the valley Hall effect in bilayer MoS2 transistors. Nat Nanotechnol 2016;11:421 e 5. https://doi.org/10.1038/nnano. 2015.337.
- [14] Kane CL, Mele EJ. Quantum spin Hall effect in graphene. Phys Rev Lett 2005;95:226801. https://doi.org/10.1103/PhysRevLett.95.226801.
- [15] Lee J, Wang Z, Xie H, Mak KF, Shan J. Valley magnetoelectricity in single-layer MoS2. Nat Mater 2017;16:887 e 91. https://doi.org/10.1038/nmat4931.
- [16] Song T, Cai X, Tu WY, Zhang X, Huang B, Wilson N, et al. Giant tunneling magnetoresistance in spin/uniFB01 lter van der Waals heterostructures. Science 2018;360:1214 e 8. https://doi.org/10.1126/science.aar4851.
- [17] Withers F, Del Pozo-Zamudio O, Mishchenko A, Rooney AP, Gholinia A, Watanabe K, et al. Light-emitting diodes by band-structure engineering in van der Waals heterostructures. Nat Mater 2015;14:301 e 6. https://doi.org/10. 1038/nmat4205.
- [18] Cao Y, Fatemi V, Fang S, Watanabe K, Taniguchi T, Kaxiras E, et al. Unconventional superconductivity in magic-angle graphene superlattices. Nature 2018;556:43 e 50. https://doi.org/10.1038/nature26160.
- [19] Wang Y, Zhang W, Chen L, Shi S, Liu J. Quantitative description on structureproperty relationships of Li-ion battery materials for high-throughput computations. Sci Technol Adv Mater 2017;18:134. https://doi.org/10.1080/ 14686996.2016.1277503.
- [20] Zhu C, Mu X, van Aken PA, Maier J, Yu Y. Fast Li storage in MoS2-graphenecarbon nanotube nanocomposites: advantageous functional integration of 0D, 1D, and 2D nanostructures. Adv. Energy Mater. 2015;5:1401170. https://doi. org/10.1002/aenm.201401170.
- [21] Shi S, Gao J, Liu Y, Zhao Y, Wu Q, Ju W, et al. Multi-scale computation methods: their applications in lithium-ion battery research and development. Chin Phys B 2016;25:018212. https://doi.org/10.1088/1674-1056/25/1/ 018212.
- [22] Chen X, Park YJ, Kang M, Kang S-K, Koo J, Shinde SM, et al. CVD-grown monolayer MoS2 in bioabsorbable electronics and biosensors. Nat Commun 2018;9:1690. https://doi.org/10.1038/s41467-018-03956-9.
- [23] Gong C, Li L, Li Z, Ji H, Stern A, Xia Y, et al. Discovery of intrinsic ferromagnetism in two-dimensional van der Waals crystals. Nature 2017;546:265 e 9. https://doi.org/10.1038/nature22060.
- [24] Huang B, Clark G, Navarro-Moratalla E, Klein DR, Cheng R, Seyler KL, et al. Layer-dependent ferromagnetism in a van der Waals crystal down to the monolayer limit. Nature 2017;546:270 e 3. https://doi.org/10.1038/ nature22391.
- [25] Esfahani EN, Li T, Huang B, Xu X, Li J. Piezoelectricity of atomically thin WSe2 via laterally excited scanning probe microscopy. Nanomater Energy 2018;52. 122-117, https://doi.org/10.1016/j.nanoen.2018.07.050.
- [26] Geim AK, Novoselov KS. The rise of graphene. Nat Mater 2007;6:183 e 91. https://doi.org/10.1038/nmat1849.
- [27] Son Y, Wang QH, Paulson JA, Shih CJ, Rajan AG, Tvrdy K, et al. Layer number
28. dependence of MoS2 photoconductivity using photocurrent spectral atomic force microscopic imaging. ACS Nano 2015;9:2843 e 55. https://doi.org/10. 1021/nn506924j.
- [28] S /C19 anchez-Royo JF, Mu ~ noz-Matutano G, Brotons-Gisbert M, Martínez-Pastor JP, Segura A, Cantarero A, et al. Electronic structure, optical properties, and lattice dynamics in atomically thin indium selenide /uniFB02 akes. Nano Res 2014;7: 1556 e 68. https://doi.org/10.1007/s12274-014-0516-x.
- [29] Lu W, Nan H, Hong J, Chen Y, Zhu C, Liang Z, et al. Plasma-assisted fabrication of monolayer phosphorene and its Raman characterization. Nano Res 2014;7: 853. https://doi.org/10.1007/s12274-014-0446-7.
- [30] Lei S, Ge L, Najmaei S, George A, Kappera R, Lou J, et al. Evolution of the electronic band structure and ef /uniFB01 cient photo-detection in atomic layers of InSe. ACS Nano 2014;8:1263 e 72. https://doi.org/10.1021/nn405036u.
- [31] Gao L, Ren W, Li F, Cheng HM. Total color difference for rapid and accurate identi /uniFB01 cation of graphene. ACS Nano 2008;2:1625 e 33. https://doi.org/10. 1021/nn800307s.
- [32] Benameur MM, Radisavljevic B, Heron JS, Sahoo S, Berger H, Kis A. Visibility of dichalcogenide nanolayers. Nanotechnology 2011;22:125706. https://doi.org/ 10.1088/0957-4484/22/12/125706.
- [33] Li H, Wu J, Huang X, Lu G, Yang J, Lu X, et al. Rapid and reliable thickness identi /uniFB01 cation of two-dimensional nanosheets using optical microscopy. ACS Nano 2013;7:10344 e 53. https://doi.org/10.1021/nn4047474.
- [34] Chen H, Fei W, Zhou J, Miao C, Guo W. Layer identi /uniFB01 cation of colorful black phosphorus. Small 2017;13:1602336. https://doi.org/10.1002/smll. 201602336.
- [35] Brotons-Gisbert M, S /C19 anchez-Royo JF, Martínez-Pastor JP. Thickness identi /uniFB01 -cation of atomically thin InSe nano /uniFB02 akes on SiO2/Si substrates by optical contrast analysis. Appl Surf Sci 2015;354:453 e 8. https://doi.org/10.1016/j. apsusc.2015.03.180.
- [36] Mao N, Tang J, Xie L, Wu J, Han B, Lin J, et al. Optical anisotropy of black phosphorus in the visible regime. J Am Chem Soc 2015;138:300 e 5. https:// doi.org/10.1021/jacs.5b10685.
- [37] Ramprasad R, Batra R, Pilania G, Mannodi-Kanakkithodi A, Kim C. Machine learning in materials informatics: recent applications and prospects. npj Comput. Mater. 2017;3:54. https://doi.org/10.1038/s41524-017-0056-5.
- [38] Butler KT, Davies DW, Cartwright H, Isayev O, Walsh A. Machine learning for molecular and materials science. Nature 2018;559:547 e 55. https://doi.org/ 10.1038/s41586-018-0337-2.
- [39] Ziatdinov M, Maksov A, Kalinin SV. Learning surface molecular structures via machine vision. npj Comput. Mater. 2017;3:31. https://doi.org/10.1038/ s41524-017-0038-7.
- [40] Liu Y, Zhao T, Ju W, Shi S. Materials discovery and design using machine learning. J. Materiomics 2017;3:159. https://doi.org/10.1016/j.jmat.2017.08. 002.
- [41] Lin X, Si Z, Fu W, Yang J, Guo S, Cao Y, et al. Intelligent identi /uniFB01 cation of twodimensional nanostructures by machine-learning optical microscopy. Nano Res 2018;52:12274. https://doi.org/10.1007/s12274-018-2155-0.
- [42] Altman N, Krzywinski M. Points of signi /uniFB01 cance: clustering. Nat Methods 2017;14:545 e 6. https://doi.org/10.1038/nmeth.4299.
- [43] Bzdok D, Krzywinski M, Altman N. Machine learning: supervised methods. Nat Methods 2018;15:5. https://doi.org/10.1038/nmeth.4551.
- [44] Blake P, Hill EW, Castro Neto AH, Novoselov KS, Jiang D, Yang R, et al. Making graphene visible. Appl Phys Lett 2007;91:063124. https://doi.org/10.1063/1. 2768624.
- [45] Castellanos-Gomez A, Agraït N, Rubio-Bollinger G. Optical identi /uniFB01 cation of atomically thin dichalcogenide crystals. Appl Phys Lett 2010;96:213116. https://doi.org/10.1063/1.3442495.
- [46] Macleod HA. Thin/uniFB01 lm optical /uniFB01 lters. Boca Raton: CRC Press; 2001.
- [47] Michelson AA. Studies in optics. Chicago: University of Chicago Press; 1927.
- [48] Wald G. Human vision and the spectrum. Science 1945;101:653 e 8. https:// doi.org/10.1126/science.101.2635.653.
- [49] Henrie J, Kellis S, Schultz SM, Hawkins A. Electronic color charts for dielectric /uniFB01 lms on silicon. Optic Express 2004;12:1464. https://doi.org/10.1364/opex.12. 001464.
- [50] Jung I, Pelton M, Piner R, Dikin D A, Stankovich S, Watcharotone S, et al. Simple approach for high-contrast optical imaging and characterization of graphene-based sheets. Nano Lett 2007;7:3569 e 75. https://doi.org/10.1021/ nl0714177.

<!-- image -->

Yuhao Li received his B.E. degree from Ningbo University in 2014. He is currently a Ph.D. candidate under the guidance of Prof. Cun-Fa Gao at Nanjing University of Aeronautics &amp; Astronautics. His research interests focus on electromechanics of atomically thin 2D materials and heterostructures.

<!-- image -->

<!-- image -->

Yangyang Kong received his B.E. degree from Min Jiang University in 2016. He is currently a M.S. degree candidate under the guidance of Prof. Wu Rong at Xin Jiang University of Physics Science and Technology. His research interests focus on 2D material preparation and characterization.

Jinlin Peng received his B.S. degree from Xiangtan University in 2015. He is currently a Ph.D. candidate under the guidance of Prof. Yunya Liu at Xiangtan University. His research interests focus on phenomenological theory and machine-learning of ferroelectrics.

<!-- image -->

<!-- image -->

Cun-Fa Gao is a professor of Nanjing University of Aeronautics &amp; Astronautics, Nanjing, China. He received his Ph.D. in Aircraft Design from Nanjing University of Aeronautics &amp; Astronautics, Nanjing, China in 1998. He was a postdoctoral research fellow in Mechanics at Peking University from 1998 to 2000, and continued a Humboldt scholar in Dresden University of Technology from 2000 to 2002. From 2002 to 2005, he was a visiting scholar in Hong Kong University of Science and Technology, Shizuoka University (Japan) and University of Sydney, respectively. Since 2005, he was a distinguished Professor of Nanjing University of Aeronautics &amp; Astronautics. His research interests include the fracture problems of multifunctional material under multi/uniFB01 eld loadings.

Rong Wu is a professor of Xinjiang University, China. She received her M.S. degree in engineering from Xinjiang University in July 2004. She was a member of the State Key Laboratory of Surface Physics at Fudan University during his Ph.D. in 2004 e 2007. Since 2007, she worked in Xinjiang University. From 2010 to 2012, he was a postdoctoral fellow at the Institute of Physics and Chemistry of the Chinese Academy of Sciences. From 2015 to 2016, she was a visiting scholar in the Department of Mechanical Engineering at the University of Washington. Her main research interest is the growth preparation and physical properties of wide-bandgap semiconductor optoelectronic materials.