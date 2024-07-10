# 3dgs_under_attack

3D Gaussian Splatting (3DGS) has recently revolutionized the field of radiance
field reconstruction, achieving high-quality novel view synthesis and fast rendering
speeds in 3D modeling. While adversarial attacks on object classification algorithms
have been extensively studied in the context of 2D images, their impact on 3D models
remains relatively unexplored. This thesis presents the Masked Iterative Fast Gradient
Sign Method (M-IFGSM), a novel approach for generating adversarial noise specifically
targeted at 3D object detection systems. This research aims to fill this gap by
investigating the vulnerabilities of 3D models created using 3D Gaussian Splatting
techniques when subjected to adversarial noise.
Our proposed M-IFGSM method generates adversarial noise focused on masked
regions of input images, ensuring that only the object of interest is perturbed. These
adversarially modified images are then utilized to construct 3D Gaussian Splatting
models. Experimental results demonstrate that the M-IFGSM effectively degrades the
performance of object classification algorithms when applied to 3D models, highlighting
significant risks for applications such as autonomous driving, robotics, and surveillance.
We selected eight different objects from the Common Objects 3D (CO3D)
dataset. We attack the provided 2D images from different angles and create 3D objects
from these noisy views. We then render the attacked 3D model from different angles to
test the effectiveness of our attacks. The results demonstrate that the M-IFGSM
method effectively generates adversarial noise that significantly reduces the confidence
and accuracy of the object detection model while being barely noticeable to the human
eye. The average confidence levels for train and test datasets drop drastically when
adversarial perturbations are introduced, highlighting the attacks’ success rate on
rendered 3D objects. Our results indicate that for train images, the average confidence
level decreases from 73% to 7%, and when the adversarial noise is added to unseen
renders, target class confidence decreases to 27%.


Click the links below to download the checkpoint to sam_ckpt/ : 

https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
