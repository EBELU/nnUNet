# nnU-Net_pediatric_segmentation
This repo contains a modified version of nnU-Net used to build a segmentation model for organs at risk (OAR) in T1-weighted MRI images of pediatric patients. Model weights can be downloaded from [Google Drive](https://drive.google.com/file/d/1t9bPWfXj-fs1r2-PGr9tFGQOiJ_EI5eh/view?usp=drive_link) and the model architecture with modifications can be installed using `pip install .`. The model can segment the following structures:
- brainstem
- cerebellum
- temporal lobes
- frontal lobes 
- hippocampus
- thalamus
- hypothalamus
 
Two models are required to segment all OAR due to overlapping voxels, these models are run by calling predict with dataset keys
```bash
"Dataset009_SFrontalLobeT+STemporalLobeT+SBrainStem+SCerebellum-MROnly"
"Dataset010_SHypothalamusT+SThalamusT+SBrainStem+SHippocampusT+SCerebellum-MROnly"
```
trainer and planner
```python
plans = "nnUNetCBAMAttentionResUNetPlans"
trainer = "nnUNetTrainerTverskyDiceCELoss_a03b07g1_1000"
```
The components of the integer masks are mapped as
```python
dataset9 = {'background': 0, 'FrontalLobe_T': 1, 'TemporalLobes_T': 2, 'BrainStem': 3, 'Cerebellum': 4}
dataset10 = {'background': 0, 'Hypothalamus_T': 1, 'Thalamus_T': 2, 'BrainStem': 3, 'Hippocampus_T': 4, 'Cerebellum': 5}
```

## References
nnUNet is published in [following paper](https://www.google.com/url?q=https://www.nature.com/articles/s41592-020-01008-z&sa=D&source=docs&ust=1677235958581755&usg=AOvVaw3dWL0SrITLhCJUBiNIHCQO):

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.

### Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).
