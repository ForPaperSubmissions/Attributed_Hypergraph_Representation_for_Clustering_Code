<meta name="robots" content="noindex">

Code: code is available under the folder code/.<br/>
Data: data is available under the folder code/data/.<br/>

----------------------------------------------
Requirements:<br/>
numpy, scipy, scikit-learn, scikit-network<br/>

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
