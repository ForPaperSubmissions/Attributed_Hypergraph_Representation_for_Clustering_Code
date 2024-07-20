<meta name="robots" content="noindex">

Code: code is available under the folder code/.<br/>
Data: data is available under the folder code/data/.<br/>

Requirements:<br/>
numpy, scipy, scikit-learn, scikit-network<br/>

Usage:<br/>
Run the algorithm AHRC by executing<br/>
python AHRC.py --dataset coau_cora<br/>
The --dataset argument should be one of the names of available datasets [coau_cora,coci_cora].<br/>

Other parameters are optional:<br/>
--alpha: the restart probability $\alpha$ in the $\alpha, \gamma$-hypergraph random walk. The default value is 0.2.<br/>
--gamma: the maximum length of $\alpha, \gamma$-hypergraph random walk. The default value is 2.<br/>
--tau: the number of spanning forest sparsification iterations. The default value is 3.<br/>
--timer: a boolean variable, set to be True if calculate the average running time of the algorithm. The default value is False.<br/>

Examples:<br/>

