Until now, the wind neural network has demonstrated good accuracy in the dataset, as Calafell et al.<sup>1</sup>  describe in their publication. Furthermore, 
The model has proven to be easy and adaptable to different geometries and contextual windows.

As part of the new extension of the research and applications, we have found that the Neural Network can make inferences on much bigger domains than 
the ones it was trained on. As shown in the next picture.

<img width="486" height="390" alt="image" src="https://github.com/user-attachments/assets/0ae35912-8368-4222-9311-be7a9e5ce450" />



Furthermore, a comparison was made the comparission between high-fidelity simulations performed with SOD2D<sup>2</sup>.
<img width="1790" height="427" alt="image" src="https://github.com/user-attachments/assets/7e04ff63-facd-4ece-aea3-a28974124ec8" />

Even though the structure of the field seems very similar, the average error is ~40%, and the error is attributed to the complete change in the geometry,
as the unseen geometry differs from the geometries on which the model was trained. Currently, some extensive work is being done on improving the
dataset and obtaining a more comprehensive dataset.



[1] Calafell Sandiumenge, Joan and Bustillo, Jaime and Mateu Armengol, Jan and Gómez, Samuel and Ramírez Jávega, Francisco and Lehmkuhl, Oriol, 
Building a General and Data-Efficient Convolutional Neural Network-Based Model for Fast Urban Flow Estimation. 
Available at SSRN: https://ssrn.com/abstract=5970896 or http://dx.doi.org/10.2139/ssrn.5970896
[2] Gasparino, L., Spiga, F., & Lehmkuhl, O. (2024). SOD2D: A GPU-enabled spectral finite elements method for compressible scale-resolving simulations. 
Computer Physics Communications, 297, 109067.
