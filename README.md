# DHMOC: a data-driven hierarchical learning framework for multi-class omics data classification

*Linping Wang, Jiemin Xie, Lei Ba, Jiarong Wu, Keyi Li, Xuemei Liu, Yunhui Xiong, Li C. Xia,***

## Summary
We proposed the method of data-driven hierarchical learning framework for multi-class omics data classification (DHMOC), interlacing classification error minimization with successive label clustering, thus enables automatic and simultaneous learning of both class hierarchy and sample labels. We evaluated DHMOC on a number of simulated and real-world multi-omics datasets, including complex disease, microbiome, single-cell and spatial transcriptomics data. The benchmark demonstrated DHMOC’s high performance in classification accuracy, as well in inferring the true biological hierarchy.

The repository contains all the data and code (https://github.com/labxscut/DHMOC/tree/DHMOC/code) used in this study, as well as some important results (please refer to the figures(https://github.com/labxscut/DHMOC/tree/main/Figures).

![image](Figures/Figure1_Study_framework.png)


## Installation

*******

## 	Simulation Data 
To evaluate the robustness and effectiveness of the DHMOC method, simulation datasets were generated under controlled conditions, designed to reflect diverse real-world scenarios. The simulation framework systematically varied sample sizes, feature dimensions, class counts, and noise distributions to ensure comprehensive testing across different contexts. The parameters considered for the simulations included sample sizes (500, 1000, 5000, and 10,000), feature dimensions (500, 1000, 5000, and 10,000), and class counts (3, 5, 7, and 10). Additionally, noise levels were modeled using standard deviations of (1/100, 1/10, 1/6, 1/4, 1/3, 1, 3), which can also be expressed as coefficients of variation 𝜆=𝜎/𝜇. Here, 𝜎 is the standard deviation of the noise, which controls the degree of dispersion of the data points. Meanwhile, 𝜇 serves as a key parameter to regulate the generation of class centers and the overall distribution characteristics of the data. Three statistical distributions were used to generate feature values, each paired with appropriate noise models to simulate variability:
* 1.	Poisson Distribution with Poisson Noise: feature values were sampled from Poisson distributions with class-specific means, and Poisson noise was added to reflect variability in count-based data, such as RNA sequencing.
* 2.	Uniform Distribution with White Noise: feature values were uniformly sampled, with Gaussian white noise added to mimic experimental measurement errors.
* 3.	Negative Binomial Distribution with Poisson Noise: feature values were generated from negative binomial distributions to model over-dispersed count data, combined with Poisson noise to simulate additional variability. This approach is particularly suitable for biological data with high replicability variance.

##  Real-world data 

Details of the real-world datasets are given below:

* ① The Fecal microbiome dataset was obtained from the ENA database (PRJEB7774) and contains microbiome profiles associated with colorectal cancer.
* ② The C. elegans Tintori stage and lineage datasets capture the transcriptomic dynamics during the development of Caenorhabditis elegans (https://github.com/qinzhu/Celegans.Tintori.Cello), a model organism with an invariant cell lineage. These datasets are ideal for studying cell differentiation and developmental processes.
* ③ Cancer-related datasets included the HCC dataset, a spatial transcriptomics dataset of hepatocellular carcinoma sourced from the HCCDB database, which integrates molecular data for liver cancer.
* ④ Breast and gastric cancer data were downloaded from The Cancer Genome Atlas (TCGA) and the Molecular Taxonomy of Breast Cancer International Consortium (METABRIC), including mutation, copy number aberration and methylation data, and through both the cBioPortal (https://www.cbioportal.org/) and the TCGA data portal.

* ⑤ Gastric cancer cell line NCI-N87 scRNA-seq data was downloaded from Gene Expression Omnibus (GSE142750) and National Institute of Health’s SRA (PRJNA498809).

* ⑥ Lymphoid cell scRNA-seq data was downloaded from human Ensemble Cell Atlas (hECA) system.

* ⑦ Two datasets from mouse models were incorporated. The Quake Smart-seq2 Limb Muscle dataset includes single-cell transcriptomic data from limb muscle tissue and is part of a larger dataset covering over 100,000 cells from 20 organs and tissues, providing a comprehensive atlas of mouse transcriptomics. The Adam dataset features single-cell RNA-seq data from 20,424 cells of newborn mouse kidneys. This dataset utilized a method designed to minimize gene expression artifacts, offering high-resolution insights into kidney development during the active nephrogenesis phase.



## Code

The dependencies required are python. All of the codes can be found https://github.com/labxscut/DHMOC/tree/DHMOC/simulate and  https://github.com/labxscut/DHMOC/tree/DHMOC/code.

The code base structure is explained below:

* **[DHMOC_simulation_all_poisson.py](https://github.com/labxscut/DHMOC/blob/DHMOC/simulate/DHMOC_simulation_all_poisson.py)**: Poisson Distribution with Poisson Noise: feature values were sampled from Poisson distributions with class-specific means, and Poisson noise was added to reflect variability in count-based data, such as RNA sequencing.
* **[DHMOC_Simulation_Data_Generation_unif.py](https://github.com/labxscut/DHMOC/blob/DHMOC/simulate/DHMOC_Simulation_Data_Generation_unif.py)**: Uniform Distribution with White Noise: feature values were uniformly sampled, with Gaussian white noise added to mimic experimental measurement errors.
* **[bin_simulation.py](https://github.com/labxscut/DHMOC/blob/DHMOC/simulate/bin_simulation.py)**: Negative Binomial Distribution with Poisson Noise: feature values were generated from negative binomial distributions to model over-dispersed count data, combined with Poisson noise to simulate additional variability. This approach is particularly suitable for biological data with high replicability variance.
*  **[DHMOC_select_classifier.py](https://github.com/labxscut/DHMOC/blob/DHMOC/code/DHMOC_select_classifier.py)**: Select the appropriate classifier for a data set.
*  **[DHMOC_appoint_classifer_.py](https://github.com/labxscut/DHMOC/blob/DHMOC/code/DHMOC_appoint_classifer_.py)**: Specify the appropriate classifier according to the data set characteristics.



# Contact & Support:

* Li C. Xia: email: [lcxia@scut.edu.cn](mailto:lcxia@scut.edu.cn)
* Linping Wang: email: [wlp.scut@outlook.com](mailto:wlp.scut@outlook.com)
