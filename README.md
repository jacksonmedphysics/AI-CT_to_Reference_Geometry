# AI-CT_to_Reference_Geometry
Neural Network to CT slice appearance with Z-axis location according to a standard reference goemetry

Requirements: python, tensorflow 2+, SimpleITK, pandas, scipy, matplotlib

Trained model provided to associate directory of dicom files with corresponding anatomical z-axis loations based on several landmarks: femoral heads (0mm), kidneys (+246mm), liver dome (+376), shoulder (+541mm), and brain (+748mm). Intermediate values are interpolated linearly.

Additionally, a script is provided to perform initial course alignment by AI detection of z-axis coordinates and X/Y center-of-mass.

[zval_prediction_alignment.png]

Rigid registration then completed with SimpleITK and threshold-derived body mask

[coregistration_sample.png]
