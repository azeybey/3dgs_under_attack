# 3dgs_under_attack

While adversarial attacks on object classification algorithms have been
extensively studied in the context of 2D images, their impact on 3D models remains
relatively unexplored. This work presents the Masked Iterative Fast Gradient Sign
Method (M-IFGSM), a novel approach for generating adversarial noise specifically
targeted at 3D object detection systems. This research aims to fill this gap by
investigating the vulnerabilities of 3D models created using 3D Gaussian Splatting
techniques when subjected to adversarial noise.

Our proposed M-IFGSM method generates adversarial noise focused on masked
regions of input images, ensuring that only the object of interest is perturbed. These
adversarially modified images are then utilized to construct 3D Gaussian Splatting
models. Experimental results demonstrate that the M-IFGSM effectively degrades the
performance of object classification algorithms when applied to 3D models, highlighting
significant risks for applications such as autonomous driving, robotics, and surveillance.


Click the links below to download the checkpoint to sam_ckpt/ : 

https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
