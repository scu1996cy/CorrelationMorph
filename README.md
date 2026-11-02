# CorrMorph: Unsupervised Deformable Brain MRI Registration Based on Correlation Mining

This is the official pytorch implementation of the paper 'CorrMorph: Unsupervised Deformable Brain MRI Registration Based on Correlation Mining,' published in the IEEE Journal of Biomedical and Health Informatics. [Paper](https://ieeexplore.ieee.org/document/10771681)

---

# Dataset

Experiments are conducted on four publicly brain MRI datasets: [Mindboggle](https://osf.io/nhtur/), [LONI Probabilistic Brain Atlas (LPBA)](https://resource.loni.usc.edu/resources/atlases-downloads/),  and [Open Access Series of Imaging Studies (OASIS)](https://sites.wustl.edu/oasisbrains/]).

---

# Train and Infer

Step 1: Before executing training or inference using commands, please ensure that the dataset address has been modified: ``` train_dir = 'Path_to_OASIS_data/Train/' ```, ``` val_dir = 'Path_to_OASIS_data/Val/' ```, and ``` test_dir = 'Path_to_OASIS_data/Test/' ```.

Step 2: ```python train_CorrelationMorph.py``` to train the model. 

Step 3: ```python infer_CorrelationMorph.py``` to test the model.

Modifying "batch_size," "lr," and "weights" in the training code can adjust the batch size, learning rate, and loss weights used for image similarity measure loss and deformation field regularization loss.

---


## Contact

If you have any questions, feel free to contact changyuan@stu.scu.edu.cn

---

## Citation

If you find this work useful, please cite:

```bibtex
@ARTICLE{10771681,
  author={Chang, Yuan and Li, Zheng and Yang, Ning},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={CorrMorph: Unsupervised Deformable Brain MRI Registration Based on Correlation Mining}, 
  year={2025},
  volume={29},
  number={4},
  pages={2798-2807},
  doi={10.1109/JBHI.2024.3508719}
}
```

---

## Reference and Acknowledgments

[VoxelMorph](https://github.com/voxelmorph/voxelmorph)

[TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration)
