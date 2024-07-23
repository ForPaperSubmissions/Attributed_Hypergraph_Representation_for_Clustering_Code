<meta name="robots" content="noindex">

Notes: The code for contrastive learning and the detailed report will be uploaded soon.<br/>

Code: code is available under the folder code/.<br/>
Data: data is available under the folder code/data/.<br/>

----------------------------------------------
Requirements:<br/>
Python 3.9.19<br/>
Numpy 1.26.4<br/>
Scipy 1.12.0<br/>
Scikit-network 0.32.1<br/>
Scikit-learn 1.4.2<br/>
Cython 3.0.10<br/>
Psutil 6.0.0<br/>
PyYAML 6.0.1<br/>
Tqdm 4.65.0<br/>
PyTorch 1.11.0<br/>
Torch-geometric 2.5.3<br/>
Run python setup.py build_ext --inplace to setup module named 'spanning_tree'<br/>

----------------------------------------------
Usage:<br/>
Run the algorithm AHRC by executing<br/>
python AHRC.py --dataset coau_cora<br/>
The --dataset argument should be one of the available datasets: [coau_cora,coci_cora].<br/>

Other parameters are optional:<br/>
--alpha: The restart probability in the $\alpha, \gamma$-hypergraph random walk. Default value is 0.2.<br/>
--gamma: The maximum length of $\alpha, \gamma$-hypergraph random walk. Default value is 2.<br/>
--tau: The number of spanning forest sparsification iterations. Default value is 3.<br/>
--timer: A boolean variable. Set to True to calculate the average running time of the algorithm. Default value is False.<br/>

----------------------------------------------
Examples:<br/>

Example-1<br/>
Evaluate the clustering results of AHRC on dataset coau_cora<br/>
python AHRC.py --dataset coau_cora<br/>

Output:<br/>
The metric scores obtained would be saved in file output_metrics.txt under the folder code/.<br/>

Example-2:<br/>
Calculate the average running time of AHRC on dataset coau_cora.<br/>
python AHRC.py --dataset coau_cora --timer True<br/>

Output:<br/>
The running time would be saved in file output_time.txt under the folder code/.<br/>
