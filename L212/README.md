# Data

This data corresponds to section 4.2.3 of Integration of a GPU-accelerated 3D fuzzy filtering method with low-dose CT reconstruction, and it consists of  images reconstructed from parallel3D projections generated from CT images,  and results of processing with different denoising methods. It has been uploaded to Zenodo: 10.5281/zenodo.18312533

This work includes reconstructed and filtered images derived from the Low Dose CT Image and Projection Data (LDCT-and-Projection-Data) dataset. Modifications include preprocessing, denoising, reconstruction, and visualization.

The original dataset is available from The Cancer Imaging Archive (TCIA) under the Creative Commons Attribution 4.0 International License (CC BY 4.0). Users of this work should cite the following reference:

McCollough, C., Chen, B., Holmes III, D., Duan, X., Yu, Z., Yu, L., Leng, S., Fletcher, J. (2020). Low Dose CT Image and Projection Data (LDCT-and-Projection-data) (Version 7) [dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/9npb-2637



# File description

The MAT files contain the following volumes, the projections reconstructed were obtained from the forward-projection of L212 volume to a Parallel3D geometry: 

LOW_NORM: Image volume reconstructed from the Parallel3D with back-projection from a quarter of the full range of projections.
REFHU_NORM: Image volume reconstructed from the Parallel3D with back-projection from the full range of projections.
REDCNN_NORM: Low-dose images denoised with the RED-CNN method.
CNCL_NORM: Low-dose images denoised with the CNCL method.
FGI_NORM: Low-dose images denoised with the F3D-FGI method, with different parameters. The volume presented in the paper is FGI_NORM_3.
FGISIRT_NORM: Image volume reconstructed from the Parallel3D with the iterative SIRT3D method regularized with F3D-FGI from a quarter of the full range of projections. The iterative method was run for 500 iterations, applying F3D-FGI every 100 iterations.

This data is presented in the HU range of [-360,3000]
