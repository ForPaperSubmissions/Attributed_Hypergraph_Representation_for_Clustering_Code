<meta name="robots" content="noindex">

Code: code is available under the folder code/.<br/>
Data: data is available under the folder code/data/.<br/>

----------------------------------------------
Requirements:<br/>
Python 3.9.19<br/>
Numpy 1.26.4<br/>
Scipy 1.12.0<br/>
Scikit-network 0.32.1<br/>
Scikit-learn 1.4.2<br/>
Psutil 6.0.0<br/>
PyYAML 6.0.1<br/>
Tqdm 4.65.0<br/>
PyTorch 1.11.0<br/>
Torch-geometric 2.5.3<br/>
Python file setup.py is also provided to setup module 'spanning_tree'<br/>

----------------------------------------------
Usage:<br/>
Run the method AHRC by executing<br/>
python AHRC.py --dataset coau_cora<br/>
The --dataset argument should be one of the available datasets.<br/>

Other parameters are optional:<br/>
--alpha: The restart probability in the $\alpha, \gamma$-hypergraph random walk. The default value is 0.2.<br/>
--gamma: The maximum length of $\alpha, \gamma$-hypergraph random walk. The default value is 2.<br/>
--tau: The number of spanning forest sparsification iterations. The default value is 3.<br/>
--timer: A boolean variable. Set to True to calculate the average running time of the algorithm. The default value is False.<br/>

Run the method GRC+ by going to the folder GRCPLUS/ then executing<br/>
python train.py --dataset coau_cora<br/>
The --dataset argument should be one of the available datasets.<br/>

Other hyperparameters are optional:<br/>
--p_e: The probability of each edge being removed.<br/>
--p_a: The probability of each attribute being removed.<br/>
--lr: The learning rate.<br/>

Run the method TCL+ by going to the folder TCLPLUS/ then executing<br/>
python train.py --dataset coau_cora<br/>
The --dataset argument should be one of the available datasets.<br/>

Other hyperparameters are optional:<br/>
--p_d: The probability of each entry in ISM being set to zero.<br/>
--lr: The learning rate.<br/>

----------------------------------------------
Examples:<br/>

Example-1<br/>
Evaluate the clustering results of AHRC on dataset coau_cora<br/>
python AHRC.py --dataset coau_cora<br/>

Output:<br/>
The metric scores obtained would be saved in the file output_metrics.txt under the folder code/.<br/>


Example-2<br/>
Evaluate the clustering results of AHRC on dataset coci_cora<br/>
python AHRC.py --dataset coci_cora<br/>

Output:<br/>
The metric scores obtained would be saved in the file output_metrics.txt under the folder code/.<br/>


Example-3:<br/>
Calculate the average running time of AHRC on dataset coau_cora.<br/>
python AHRC.py --dataset coau_cora --timer True<br/>

Output:<br/>
The running time would be saved in the file output_time.txt under the folder code/.<br/>

Example-4:<br/>
Evaluate the clustering results of GRC+ on dataset coau_cora<br/>
python train.py --dataset coau_cora<br/>

Output:<br/>
The metric scores obtained would be saved in the file GRCPLUS.txt under the folder data/coau_cora/results/.<br/>

Example-5:<br/>
Evaluate the clustering results of TCL+ on dataset coau_cora<br/>
python train.py --dataset coau_cora<br/>

Output:<br/>
The metric scores obtained would be saved in the file TCLPLUS.txt under the folder data/coau_cora/results/.<br/>
