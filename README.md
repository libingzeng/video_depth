 # Test-Time Optimization for Video Depth Estimation Using Pseudo Reference Depth
 
 ### [[Paper](https://libingzeng.github.io/projects/depth/depth/depth_cgf.pdf)] [[Project Website](https://libingzeng.github.io/projects/depth/depth.html)]

<p align='center'>
<img src="teaser.gif" width='100%'/>
</p>

In this paper, we propose a learning-based test-time optimization approach for reconstructing geometrically consistent depth
maps from a monocular video. Specifically, we optimize an existing single image depth estimation network on the test example
at hand. We do so by introducing pseudo reference depth maps which are computed based on the observation that the optical
flow displacement for an image pair should be consistent with the displacement obtained by depth-reprojection. Additionally, we
discard inaccurate pseudo reference depth maps using a simple median strategy and propose a way to compute a confidence map
for the reference depth. We use our pseudo reference depth and the confidence map to formulate a loss function for performing
the test-time optimization in an efficient and effective manner. We compare our approach against the state-of-the-art methods
on various scenes both visually and numerically. Our approach is on average 2.5X faster than the state of the art and produces
depth maps with higher quality.
<br/>

**Consistent Video Despth Estimation**
<br/>
[Libing Zeng](https://libingzeng.github.io/), 
[Nima Khademi Kalantari](http://faculty.cs.tamu.edu/nimak/)
<br/>
In Computer Graphics Forum (Proceedings of Eurographics 2023).

 
# Prerequisite
- Download third-party packages from the following link.
  ```
  https://drive.google.com/drive/folders/1JoeeI6aV2zR4doU-2c0fHxFG_13SYu1T?usp=drive_link
  ```
- Install python packages.
  ```
  conda env create -f environment.yml
  conda activate consistent_depth
  ```
- Intall FFmpeg and COLMAP following intructions from [Consistent Video Despth Estimation](https://github.com/facebookresearch/consistent_depth#readme).


# Quick Start
You can run the following demo **without** installing **COLMAP**.
- Download the precomputed COLMAP results of the demo video from data/video/ayush.mp4 from the following link.
  ```
  https://drive.google.com/drive/folders/1vsNU_qsqrkRzM5LRD0Ssjkdy22iOAePI?usp=drive_link
  ```
- Run
  ```
  ./shell/run_ayush.sh
  ```
- You can find some intermediate visual results as below.
  ```
  flow # optical flow
  mask_rectified_eps2_2e-06_gdt_10.0_dmm_0.0 # per-pair mask
  mask_rectified_median_eps2_2e-06_gdt_10.0_dmm_0.0 # per-frame mask
  depth_gt_predicted_png_eps2_2e-06_gdt_10.0_dmm_0.0_prepro1 # per-pair and per-frame depth maps
  depth_median_confidence_map_eps2_2e-06_gdt_10.0_dmm_0.0_ctm0.1_normalized_png # confidence map
  ```
- You can find final results as below.
  ```
  results/ayush_test/R_hierarchical2_mc
  ```

# Customized Run:
  Please follow intructions from [Consistent Video Despth Estimation](https://github.com/facebookresearch/consistent_depth#readme).

# Citation
If you find our code useful, please consider citing our paper:
```
  @article{Zeng_2023_depth,
      author = {Zeng, Libing and Kalantari, Nima Khademi},
      title = {Test-Time Optimization for Video Depth Estimation Using Pseudo Reference Depth},
      journal = {Computer Graphics Forum},
      doi = {https://doi.org/10.1111/cgf.14729},
      url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.14729},
  }
              
```

# License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details.

# Acknowledgments
We thank the reviewers for their insightful comments. We also thank Brennen Taylor for capturing the input sequences.
Additionally, this implementation is based on [Consistent Video Despth Estimation](https://github.com/facebookresearch/consistent_depth#readme).