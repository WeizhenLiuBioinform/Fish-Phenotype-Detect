# Benchmarking Fish Dataset and Evaluation Metric in Keypoint Detection -Towards Precise Fish Morphological Assessment in Aquaculture Breeding
 [![IJCAI](https://img.shields.io/badge/IJCAI-paper-brightgreen)](https://arxiv.org/abs/2405.12476)
 [![arXiv](https://img.shields.io/badge/arXiv-paper-purple?link=https%3A%2F%2Farxiv.org%2Fabs%2F2405.12476)](https://arxiv.org/abs/2405.12476)
 [![fish](https://img.shields.io/badge/fish-Image-yellow?link=https%3A%2F%2Farxiv.org%2Fabs%2F2405.12476)](#Dataset)
 [![Poster](https://img.shields.io/badge/Poster-Presentation-cyan)]()

This repository contains the code and dataset for the paper "Benchmarking Fish Dataset and Evaluation Metric in Keypoint Detection -Towards Precise Fish Morphological Assessment in Aquaculture Breeding," accepted at IJCAI2024 in the AI and Social Good track.

[Weizhen Liu<sup>1</sup>](https://www.researchgate.net/profile/Weizhen-Liu), [Jiayu Tan<sup>1</sup>](https://arxiv.org/search/cs?searchtype=author&query=Tan,+J), [Guangyu Lan<sup>1</sup>](https://arxiv.org/search/cs?searchtype=author&query=Lan,+G), [Ao Li<sup>1</sup>](https://arxiv.org/search/cs?searchtype=author&query=Li,+A), [Dongye Li<sup>4</sup>](https://arxiv.org/search/cs?searchtype=author&query=Li,+D), [Le Zhao<sup>1</sup>](https://arxiv.org/search/cs?searchtype=author&query=Zhao,+L), [Xiaohui Yuan<sup>1,3 *</sup>](https://arxiv.org/search/cs?searchtype=author&query=Yuan,+X), [Nanqing Dong<sup>2 *</sup>](https://eveningdong.github.io/)

<p align="center"><sup>1</sup>School of Computer Science and Artificial Intelligence, Wuhan University of Technology</p>
<p align="center"><sup>2</sup>Shanghai Artificial Intelligence Laboratory</p>
<p align="center"><sup>3</sup>Yazhouwan National Laboratory</p>
<p align="center"><sup>4</sup>Sanya Boruiyuan Technology Co. Ltd</p>
<p align="center"><sup>{liuweizhen, tjy2023305211,yuanxiaohui}@whut.edu.cn, dongnanqing@pjlab.org.cn</sup></p>

<div style="text-align:center">
<img src="assets/figure.png" width="800" alt="" class="img-responsive">
</div>



## Overview

- [Motivation and design](#motivation-and-design)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Citations](#citations)




##  Motivation and design

Accurate phenotypic analysis in aquaculture breeding necessitates detailed morphological data, yet current datasets are small-scale, species-limited, and lack comprehensive annotations. To address these shortcomings, we introduce FishPhenoKey, a dataset of 23,331 high-resolution images across six fish species, annotated with 22 phenotype-oriented keypoints. This dataset facilitates precise measurement of complex morphological features critical for advanced aquaculture research.

FishPhenoKey is designed to support diverse research needs with flexibility and scalability. By integrating high-resolution imaging and robust annotation protocols, it provides the necessary diversity and detail for accurate phenotypic analysis. The modular design of FishPhenoKey ensures compatibility with various evaluation metrics and keypoint detection models, enabling continuous updates and expansion to accommodate future advancements in aquaculture and genetic studies.

## Dataset

The FishPhenoKey dataset includes high-resolution JPG images of six fish species, each captured from two different viewpoints. Under the guidance of fishery experts, we have precisely annotated four of these species to support the task of measuring subtle morphological phenotypes in fish body parts. The definitions of the 22 keypoints are detailed as follows: 1: tip of the snout, 2: posterior end of the operculum, 3: top end of the head, 4: isthmus, 5: dorsal apex, 6: bottom end of the ventral margin, 7: top end of the caudal peduncle, 8: bottom end of the caudal peduncle, 9: posterior end of the tail fin, 10: posterior end of the caudal vertebrae, 11: anterior end of the eye, 12: posterior end of the eye, 13: anterior end of the pectoral fin, 14: posterior end of the pectoral fin, 15: anterior end of the pelvic fin, 16: posterior end of the pelvic fin, 17: anterior end of the anal fin, 18: posterior end of the anal fin, 19: outer margin of the anal fin, 20: anterior end of the dorsal fin, 21: posterior end of the dorsal fin, 22: outer margin of the dorsal fin.

<div style="display: flex; justify-content: center;">
  <img src="assets/fish.png" width="400" alt="" style="margin-right: 10px;">
  <img src="assets/fish.gif" width="400" alt="">
</div>

If you need the complete dataset, please download the [FishPhenoKey Dataset User Agreement](./FishPhenoKey Dataset User Agreement.docx) and read the relevant regulations. If you agree to the regulations, please fill in the relevant user information in the user agreement, [authorization date], and [electronic signature] at the end of the agreement. Send the PDF format of the user agreement to the email **[liuweizhen@whut.edu.cn](mailto:liuweizhen@whut.edu.cn)**. After review, we will send the download link for the complete dataset via email.



## Environment Setup

* Python 3.6/3.7/3.8
* Pytorch 1.10 or above
* pycocotools (Linux: `pip install pycocotools`; Windows: `pip install pycocotools-windows` (no need to install VS separately))
* Ubuntu or CentOS (Windows not recommended)
* Preferably use GPU for training
* For detailed environment setup, refer to `requirements.txt`

## Citations
Please cite our paper in your publications if our methods and dataset are helpful to your research. The BibTeX is as follows:

~~~
@article{liu2024benchmarking,
  title={Benchmarking Fish Dataset and Evaluation Metric in Keypoint Detection-Towards Precise Fish Morphological Assessment in Aquaculture Breeding},
  author={Liu, Weizhen and Tan, Jiayu and Lan, Guangyu and Li, Ao and Li, Dongye and Zhao, Le and Yuan, Xiaohui and Dong, Nanqing},
  journal={arXiv preprint arXiv:2405.12476},
  year={2024}
}
~~~

