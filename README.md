This repository contains code demonstrating the Co-learn++ method in our IJCV paper [Source-Free Domain Adaptation Guided by Vision and Vision-Language Pre-Training](https://arxiv.org/pdf/2405.02954). This is an extension of the Co-learn method in our ICCV 2023 paper [Rethinking the Role of Pre-Trained Networks in Source-Free Domain Adaptation](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Rethinking_the_Role_of_Pre-Trained_Networks_in_Source-Free_Domain_Adaptation_ICCV_2023_paper.pdf).

### Prerequisites:

We used NVIDIA container image for PyTorch, release 20.12, to run experiments.

Install additional libraries with `pip install -r requirements.txt`.

### Dataset:

- Please manually download the datasets [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification), [DomainNet](https://ai.bu.edu/M3SDA/#dataset) from the official websites, and modify the path of images in each '.txt' under the folder './code/data/'. Scripts to generate the txt files are in the respective data folders.

### Training:

- Training scripts in './code/uda/scripts'. Run 'eval_target_zeroshot.sh' for zero-shot CLIP and 'train_target_two_branch.sh' for co-learning with CLIP encoder.
- Results consolidation scripts in './code/uda/consolidation_scripts'.

## Citation

```
@article{zhang2024colearnplus,
    author = {Zhang, Wenyu and Shen, Li and Foo, Chuan-Sheng},
    year = {2024},
    month = {08},
    pages = {1-23},
    title = {Source-Free Domain Adaptation Guided by Vision and Vision-Language Pre-training},
    journal = {International Journal of Computer Vision},
    doi = {10.1007/s11263-024-02215-3}
}

@inproceedings{zhang2023colearn,
    author = {Zhang, Wenyu and Shen, Li and Foo, Chuan-Sheng},
    booktitle = {2023 IEEE/CVF International Conference on Computer Vision (ICCV)},
    title = {Rethinking the Role of Pre-Trained Networks in Source-Free Domain Adaptation},
    year = {2023},
    volume = {},
    issn = {},
    pages = {18795-18805},
    doi = {10.1109/ICCV51070.2023.01727},
    url = {https://doi.ieeecomputersociety.org/10.1109/ICCV51070.2023.01727},
    publisher = {IEEE Computer Society},
    address = {Los Alamitos, CA, USA},
    month = {oct}
}
```

## Acknowledgements

Our implementation is based off [SHOT++](https://github.com/tim-learn/SHOT-plus). Thanks to the SHOT++ implementation.
