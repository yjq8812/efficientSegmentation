# efficientSegmentation

An implementation for an efficient segmentation with a supermodular loss function. This implementation also allows user-defined loss functions and user-defined optimisations for the relative loss maximization and inference procedure. 

For more details: 
Yu, J. and M. B. Blaschko: Efficient Learning for Discriminative Segmentation with Supermodular Losses. BMVC, 2016.

1. Usage

(1) Dataset: 

- This implementation uses the dataset provided by:
- 
V. Gulshan, C. Rother, A. Criminisi, A. Blake and A. Zisserman
Geodesic Star Convexity for Interactive Image Segmentation, CVPR, 2010. 
Run directly main.m first. It by defaut runs on a preprocessed and downsampled dataset "Sampled690.mat". 

- One can change the dataset in mainInit.m in order to use the OriginalData.mat, which takes longer time.

(2) Loss function and its optimization:

- Define your loss function in customLossFunction.m following the specific format.
- Use your loss function in mainInit.m with the name your defined.
- '8connected' is the loss in the paper.

2. Rerun the prepocessing

Download the dataset:

- Images: http://www.robots.ox.ac.uk/~vgg/data/iseg/data/images.tgz
- Ground truth: http://www.robots.ox.ac.uk/~vgg/data/iseg/data/images-gt.tgz
- Brush strokes: http://www.robots.ox.ac.uk/~vgg/data/iseg/data/images-labels.tgz

Add the dataset at "./generateUnary/Dataset/" respectively.
run ./generateUnary/main.m and save the output X and Y.

3. Reference

This uses several third-party packages. Please consider citing :

(1) SFO toolbox
Andreas Krause. SFO: A toolbox for submodular function optimization. JMLR, 11:1141–1144, 2010.

(2) GCMex & BKMatlab - MATLAB wrapper for the Boykov-Kolmogorov graph cuts
- Yuri Boykov and Vladimir Kolmogorov. An experimental comparison of min-cut/maxflow
algorithms for energy minimization in vision. T-PAMI, 26(9):1124–1137, 2004.
- Yuri Boykov, Olga Veksler, and Ramin Zabih. Efficient approximate energy minimization
via graph cuts. T-PAMI, 20(12):1222–1239, 2001.
- B. Fulkerson, A. Vedaldi, and S. Soatto. Class segmentation and object localization
with superpixel neighborhoods. In ICCV, 2009.

(3) Geodesic star convexity 

V. Gulshan, C. Rother, A. Criminisi, A. Blake and A. Zisserman,
Geodesic star convexity for interactive image segmentation. 
In Proceedings of Conference on Vision and Pattern Recognition (CVPR 2010).


--
Jiaqian Yu @ 2016
