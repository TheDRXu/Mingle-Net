<h1>MingleNet: A Novel Dual Stacking Approach for Medical Image Segmentation</h1>
Medical image segmentation is important for disease diagnosis and treatment planning. 
</br>
Ensemble learning, which combines multiple models or predictions, can improve accuracy and performance in medical image segmentation. We propose MingleNet, which uses multiple layers of ensemble learning. 
</br>
MingleNet uses double-stacking of models, such as DoubleU-Net, DeepLabv3+, U-Net, and DeepLab, to produce masks.


<h2>MingleNet Architecture</h2>
<img src="https://github.com/TheDRXu/Mingle-Net/assets/101695920/19e8c9a7-ce59-4fc8-93bd-0c098cc7022c" width=50% height=50%>

<h2>Dataset</h2>
<ul>
  <li>Kvasir SEG = [here](https://datasets.simula.no/kvasir-seg/)</li>
  <li>CVC-ClinicDB = [here](https://datasetninja.com/cvc-612)</li>
  <li>CVC-ColonDB</li>
</ul>

#Installations
Install all the required libraries using the command:
```
pip install -r requirements.txt
```
Note that other versions of the libraries may also work. This setup was tested with an RTX 3080 TI GPU and 32GB of RAM.

<h2>How to run</h2>
This Project can be run using the ```Notebook.ipynb``` file.
