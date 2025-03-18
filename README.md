# kitchen-surfaces-AD
Anomaly Detection on Diverse Kitchen Surfaces: A Case Study Using Industry Data

##Abstract

Anomaly detection is crucial for maintaining quality control in industrial manufacturing,
ensuring that defective products are identified before reaching consumers.
Deep learning-based unsupervised anomaly detection methods, such as PatchCore
by Amazon AWS and CutPaste by Google Cloud AI Research, have demonstrated
strong performance on benchmark datasets like MVTec. However, their effectiveness
in real-world industrial settings remains an open challenge. This thesis evaluates
and compares these two methods for detecting surface defects in kitchen boards
using high-resolution images from an industrial production environment.
The dataset consists of images from a kitchen carpentry production facility. The
analysis focuses on two categories, White and White with Edges. To process the
data efficiently, a patch-based approach was implemented, extracting and normalizing
image patches to enable localized anomaly detection. PatchCore utilizes a memory
bank approach for anomaly detection, while CutPaste employs self-supervised
learning with synthetic anomaly generation. Both methods were modified to accommodate
the industrial dataset, incorporating custom data loaders, augmented
training strategies, and visualization techniques for qualitative analysis.
Experiments were conducted using ResNet18 and ResNet50 backbones with and
without blur preprocessing. Results indicate that PatchCore achieved higher ROCAUC
(up to 89.06%) and precision, making it preferable for high-precision defect
detection. In contrast, CutPaste exhibited higher recall, detecting a larger proportion
of anomalies but at the cost of increased false positives.
The findings demonstrate that PatchCore is better suited for applications where false
positives must be minimized, while CutPaste is more effective in recall-prioritized
scenarios. This work bridges the gap between cutting-edge anomaly detection research
and real-world industrial applications, providing insights into the deployment
of deep learning-based quality control solutions in manufacturing.

## Acknowledgments

This project is based on the following repositories:

- [CutPaste: Self-Supervised Learning for Anomaly Detection](https://github.com/LilitYolyan/CutPaste)
- [ind_knn_ad: Anomaly Detection using k-NN](https://github.com/rvorias/ind_knn_ad/tree/master) 

Special thanks to the original authors for their contributions!
