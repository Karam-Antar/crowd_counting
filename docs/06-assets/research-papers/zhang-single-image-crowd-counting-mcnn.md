# Single-Image Crowd Counting via MCNN

## Citation

Yingying Zhang, Desen Zhou, Siqin Chen, Shenghua Gao, and Yi Ma. "Single-Image Crowd Counting via Multi-Column Convolutional Neural Network." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 589-597.

- [Official CVF Open Access page](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.html)
- [Official PDF](https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)

## Relevance to this project

This paper maps a crowd image to a density map with a Multi-Column Convolutional Neural Network (MCNN). Its multi-column receptive fields address changes in head size caused by perspective and image resolution. It also describes geometry-adaptive kernels for constructing density-map targets.

These ideas are relevant to this project because the project:

- treats crowd counting as spatial density-map regression,
- uses geometry-aware density targets,
- evaluates the integrated density map as a crowd count,
- addresses severe scale variation in crowded scenes.

The project later moved from the paper's MCNN architecture to a pretrained encoder-decoder architecture, as recorded in the decision records. The paper is therefore a foundational reference for the problem formulation and target design, not a claim that the current model is an MCNN implementation.

## Related project documentation

- [Density-map regression decision](../../04-decisions/01-adrs/0001-use-density-map-regression.md)
- [Geometry-adaptive targets decision](../../04-decisions/01-adrs/0002-use-geometry-adaptive-targets.md)
- [Dataset and density-map contract](../../05-data/01-datasets.md)
