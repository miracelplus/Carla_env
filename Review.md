# CEE 554 Data Mining in Transportation



### Perceptron Learning Algorithm:

$$
h(X)=sign[\sum_{i=1}^{d}{w_ix_i-threshould}]\\
h(X)=sign[\sum_{i=1}^{d}{w_ix_i-w_0}]\\
$$

The feature vector will add a constant number, for convenience, we use 1
$$
X = [x_1,\cdots,x_d]\rightarrow X = [1,x_1,\cdots,x_d]\\
h(X)=sign[\sum_{i=0}^{d}{w_ix_i}]
$$
Vector form:
$$
h(X) = sign[W^TX]
$$
The algorithm diagram:

<img src="/Users/Zhang/Library/Application Support/typora-user-images/image-20200304151534649.png" alt="image-20200304151534649" style="zoom: 50%;" />



<u>**PLA for linearly separable data**</u>

​			$\bullet\  $	Start with a weight vector W

​    		$\bullet$     $\forall X_i\in X_{train}$ : If data point i is misclassified, $W\leftarrow W+y_iX_i$

​            $\bullet$     If the classification error $􏰈\sum_{i=1}^{N}{\frac{\delta[y_i\neq g_i]}{N}}$ is greater than zero, go to step 2. Otherwise, terminate.



##### Intuition behind the update rule of PLA

<img src="/Users/Zhang/Library/Application Support/typora-user-images/image-20200304154259166.png" alt="image-20200304154259166" style="zoom:50%;" />

##### Logistic Regression

Instead of using a hard threshould, we now use a smooth function and give it the meaning as "probability".

<img src="/Users/Zhang/Library/Application Support/typora-user-images/image-20200304154445134.png" alt="image-20200304154445134" style="zoom: 67%;" />
$$
\sigma(z)=\frac{1}{1+e^{-z}}
\\
$$

$$
\\P(y_i=1)
$$

<u>Learning Model</u>

$\bullet$    Hypothesis:
$$
h(X)=\begin{cases}
1& \delta(W^TX)\geq0.5\\
-1& \delta(W^TX)<0.5
\end{cases}
$$
$\bullet$    Mean Squared Error (MSE)
$$
C(W) = \frac{1}{N}\sum_{i=1}^{N}{(\delta(W^TX_i)-y_i)^2}
$$
​	  We want to let $W^*=argmin_W{C(W)}$



### Batch Gradient Descent

​		$C(W) = \frac{1}{N}\sum_{i=1}^{N}{(\delta(W^TX_i)-y_i)^2}$

​		$w\leftarrow w-\lambda\frac{\partial}{\partial w}C(W)$

​		$\frac{\partial}{\partial w_j}C(W) = \frac{2}{N}\sum_{i=1}^{N}{(\delta(W^TX_i)-y_i)\cdot\partial\sigma(W^TX_i)\cdot X_i}$

​	

​		$\bullet\  $	Select a step size $\lambda$

​		$\bullet$    Updata the weight vector $W:w\leftarrow w-\lambda\frac{\partial}{\partial w}C(W)$

​        $\bullet$     If $W$ converges or the error rate is smaller than requirement, then stop. Otherwise, adjust $\lambda$ and go to step 2.



### Stochastic/Online Gradient Descent

Update rule is the same.

​		$\bullet$     Shuffle the data

​		$\bullet\  $	Select a step size $\lambda$

​		$\bullet$    Updata the weight vector $W:w\leftarrow w-\lambda\frac{\partial}{\partial w}C(W,X_i)$ for each data point

​        $\bullet$     Stop if the number of iterations reaches a pre-set number, or the change in C between two consecutive interations is less than a pre-determined tolerance; Otherwise, go to step 2.

<img src="/Users/Zhang/Library/Application Support/typora-user-images/image-20200304161735710.png" alt="image-20200304161735710" style="zoom:50%;" />



### Learning Model for Logistic Regression

Hypothesis Set: $h(X)=\sigma(W^TX)$

Cost Function: Take the log of preious MSE and then convert to a convex function, which means we can find the global optimum.
$$
C(W)=\frac{1}{N}\sum_{i=1}^{N}-y_ilog(\sigma)-(1-y_i)log(1-\sigma)
$$

$$
W^* = argmin_W{C(W)}
$$

### Linear Regression

For Linear Regression, we can derive the closed form of the $W^*$ as follows:

Let
$$
\nabla_W(C)=\frac{2}{N}X^T(XW-Y)=0
$$

$$
W^*=(X^TX)^{-1}X^TY
$$





### Hoeffding's Inequality

Is learning feasible? May be and may be not!
$$
P(|E_{in}(h)-E_{out}(h)|\geq\epsilon)\leq2e^{-2\epsilon^2N}
$$
It is a verification of hypothesis set $h$ if it is feasible given $N$ data and tolerance $\epsilon$, not learning !

We can move even further, since we know for a hypothesis set $h$, suppose it contains $M$ hypothesis, then for our final hypothesis $g$ , by using union bound, 
$$
P(|E_{in}(g)-E_{out}(g)|\geq\epsilon)\leq \sum_{i=1}^{M}P(|E_{in}(h_i)-E_{out}(h_i)|\geq\epsilon)\\
\qquad =2Me^{-2\epsilon^2N}
$$

$$
\Rightarrow P(|E_{in}(g)-E_{out}(g)|\geq\epsilon)\leq 2Me^{-2\epsilon^2N}
$$

Next we need to figure what $M$ is. $M$ is the number of distinct hypothesis $h_i$, here "distinct" means $h_i$ and $h_j$ have different $E_{in}$ and $E_{out}$. A natural understanding is that while there are plenty of hypothesis, some of them, in fact, are pretty similar with same $E_{in}$ and $\Delta E_{out}(h_i,h_j)$ is really small. A straightforward understanding is follows:

<img src="/Users/Zhang/Library/Application Support/typora-user-images/image-20200304165603671.png" alt="image-20200304165603671" style="zoom:50%;" />

We can consider the three figures in the first row as the same hypothesis and the three figures in the second row as another hypothesis. By this simplification, we can now replace $M$ with the number of dichotomies, making uncountable countable.

### Growth Function

Definition: the **<u>growth function</u>** counts the most dichotomies on any $N$ points.
$$
m_\cal{H}(\it{N})=\mathop{\max}_{X_1,X_2,\cdots,X_N\in \cal{X}}|\cal{H}(\it(X_1,X_2,\cdots,X_N)|
$$
Notice:                                    $m_\cal{H}(\it{N})\leq \rm{2^N}$



Three Growth Functions:

$\bullet$    Positive Rays:		$m_\cal{H}(\it{N})= N+1$

$\bullet$    Positive Intervals:		$m_\cal{H}(\it{N})= \rm{\frac{1}{2}N^2+\frac{1}{2}N+1}$

$\bullet$    Convex Sets:		$m_\cal{H}(\it{N})= \rm{2^\it{N}}$



With Growth function, we can rewrite the above inequality as:
$$
P(|E_{in}(g)-E_{out}(g)|\geq\epsilon)\leq 2m_\cal{H}(\it{N})e^{-\rm{2}\epsilon^2N}
$$

##### Upper Bound of Growth Function

For a fixed $\cal{H}$ and a fixed break point $k$
$$
m_\cal{H}(\it{N})\leq \sum_{i=0}^{k-1}\pmatrix{N\\i}
$$


### Break Point

Definition: If no dataset of size $k$ can be shattered by $\cal{H}$, then $k$ is the brak poing of $\cal{H}$

In other words, we only need to check if $m_\cal{H}(\it{N})= \rm{2^N}$ at which number of $N$

$\bullet$    2D Perceptron:		$k=4$ 

$\bullet$    Positive Rays:		$m_\cal{H}(\it{N})= N+1$ , $k=2$ 

$\bullet$    Positive Intervals:		$m_\cal{H}(\it{N})= \rm{\frac{1}{2}N^2+\frac{1}{2}N+1}$ , $k=3$ 

$\bullet$    Convex Sets:		$m_\cal{H}(\it{N})= \rm{2^N}$ , $k=N$



### V-C Inequality and V-C Dimension

After figuring out what growth function is, we can get :


$$
P(|E_{in}(g)-E_{out}(g)|\geq\epsilon)\leq 2Me^{-2\epsilon^2N}
$$

$$
\Rightarrow P(|E_{in}(g)-E_{out}(g)|\geq\epsilon)\leq 4m_{\cal{H}}(2N)e^{-\frac{1}{8}\epsilon^2N}
$$



Definition: A quantity that is defined for a hypothesis $\cal{H}$ and denoted by $d_{VC}(\cal{H})$ . Also, it equals the most points $\cal{H}$ can shatter.

$N\leq d_{VC}(\cal{H})$ : there $\color{red}{exists}$ $N$ points than $\cal{H}$ can shatter

$N>d_{VC}(\cal{H})$ : $N$ is a break point for $\cal{H}$ 

Anythng above the VC dimension is a break point, and for any $N$ below the VC dimension, there is a constellation of points that we can shatter.



Now, we can represent the growth function from:
$$
m_\cal{H}(\it{N})\leq \sum_{i=0}^{k-1}\pmatrix{N\\i}  
$$
To: 
$$
m_\cal{H}(\it{N})\leq \sum_{i=0}^{d_{VC}}\pmatrix{N\\i}
$$
And the order of the polynomial that bounds the growth function is $N^{d_{VC}}$

$\bullet$    Positive Rays:		$d_{VC}=1$

$\bullet$    2D perceptron:		$d_{VC}=3$

$\bullet$    Convex Sets:		$d_{VC}=\infty$ 



**VC-Dimension is independent of the <u>learning algorithm,</u> <u>input distribution</u> and <u>target function</u>, only dependent on the learning model (i.e. $2^{nd}$ order linear model)**



For a $d$ dimensional perceptron, 
$$
\color{red}{d_{VC} =d+1}
$$


##### Number of Data Points Needed

We denote the right hand side of V-C Inequality as $\delta(N,d,\epsilon)$ .If we fixed $\epsilon$ , $\delta$ will become a function of $N$ and $d$ . To capture this trade-off, we use $N^de^{-N}$ as a rough estimation of $\delta$ .
$$
P(|E_{in}(g)-E_{out}(g)|\geq\epsilon)\leq N^de^{-N}
$$
In reality, we always need 1,000 times of VC-dimension datat points, that is $N\geq 10d_{VC}$ 

##### Performance

Given two different models with same $\epsilon$ , $N$ , different $\cal{H}$ with different VC dimension. Which model is better?

Answer: The model with lower VC dimension has a better <u>bound</u> on generalization performance since the model is simplier and it can fit more cases. To obtain the same generalization performance for a model with a higher VC dimension, we need more data points.

##### Approximation vs Generalization

From
$$
P(|E_{in}(g)-E_{out}(g)|\geq\epsilon)\leq 4m_{\cal{H}}(2N)e^{-\frac{1}{8}\epsilon^2N}
$$
We denote right hand side as $\delta$ ,

$\Rightarrow$  
$$
\Omega(N,\cal{H},\delta)=\epsilon=\sqrt{\frac{8}{\it{N}}\ln\frac{4m_{\cal{H}}(\rm2\it{N})}{\delta}}
$$
For good event, $P(Good\;Event)\geq 1-\delta$ , $|E_{out}-E_{in}|\leq \Omega(N,\cal{H},\delta)$. We can always say $E_{out}\geq E_{in}$ , then we get 
$$
E_{out}\leq E_{in}+\Omega
$$
where $E_{in}$ means "Approximation Performance", $\Omega$ means "Generalization Performance".

<img src="/Users/Zhang/Library/Application Support/typora-user-images/image-20200304190521004.png" alt="image-20200304190521004" style="zoom: 50%;" />

More complex $\cal{H}\quad \rightarrow$ Better chance of having $f$ in $\cal{H}$ : Approximation

Less complex  $\cal{H}\quad \rightarrow$ Better chance of finding $f$ : Generalization



### Quantify the trade-off

VC-dimension is one approach:       $E_{out}\leq E_{in}+\Omega$ 



Bias-Variance decomposition is another approach:

​						<u>Bias</u>: How well can $\cal{H}$ approximate $f$  overall; the probability to have $g$ 

​						<u>Variance</u>:  How well can we zoom in on a good $h\in \cal{H}$ ; the probability to find $g$ in pocket

<u>Bias</u> : $E[((\overline{g}(x)-f(x))^2] $ 

<u>Variance</u>: $E_x[E_D[(g^{(D)}(x)-\overline{g}(x))^2]]$ 



Bigger $\cal{H}\;\rightarrow$ the **<u>bias</u>** goes down, and the **<u>var</u>** goes up (with bigger $\cal{H}$ we can have better approximation of $f$ while the uncertainty about which $h$ to pick increases)

<img src="/Users/Zhang/Library/Application Support/typora-user-images/image-20200304213316939.png" alt="image-20200304213316939" style="zoom:50%;" />
$$
\color{red}{E_{out}=bias+var}
$$
<img src="/Users/Zhang/Library/Application Support/typora-user-images/image-20200304213838170.png" alt="image-20200304213838170" style="zoom:50%;" />

Here, the distribution in training sets and outside data should be the same.

##### Target Distribution

<img src="/Users/Zhang/Library/Application Support/typora-user-images/image-20200304214029832.png" alt="image-20200304214029832" style="zoom:50%;" />

$P(y|X)$ : Target distribution $\rightarrow$  We are trying to learn

$P(X)$ : The input distribution $\rightarrow$ We are not trying to learn 



### Decision Tree

Measure of selecting the best split:

<img src="/Users/Zhang/Library/Application Support/typora-user-images/image-20200304220248260.png" alt="image-20200304220248260" style="zoom: 67%;" />

How to build a tree? We are selecting a variable to split on and **<u>we select the one that leads to highest reduction in entropy</u>**.

For a parant node, we calculate the Entropy
$$
Entropy=E[\log_2\frac{1}{x}]=\sum_{x}{x\log_2\frac{1}{x}}
$$
where $x$ is the frequency (probability) of each variable.

<img src="/Users/Zhang/Library/Application Support/typora-user-images/image-20200304220754068.png" alt="image-20200304220754068" style="zoom:50%;" />
$$
\Delta_{info}=Entropy(paraent)-\sum_{i=1}Entropy(i)\cdot p_i\\
s.t.\; \sum_{i=1}p_i=1
$$
The best split is to $argmax\Delta_{info}$ 

In fact, we are trying to find the best split that maximizes the change of impurity. When th impurity measure is entropy, the difference in entropy is known as the **Information Gain**, $\Delta_{info}$.
$$
\Delta=I(parent)-\sum_{i=1}^{k}\frac{N(v_j)}{N}I(v_j)
$$

##### Reduced Error Pruning

<u>Algorithm</u>

$\bullet$ 		Classify examples in validation set

$\bullet$ 		For each internal node: Calculate the error of the sub-tree if converted to a leaf with majority class label

$\bullet$ 		Prune node with highest reduction in error on  $\color{red}{Test\;Set}$ 



How to resolve the issue of over-fitting in decision trees:

$\bullet$ 		Pre-pruning: Stop at some point and take the majority vote

$\bullet$ 		Post-pruning: Grow a full-length tree until you get all pure classes, and then start pruning the tree using a validation dataset





### Neural Networks

Number of layers: Hidden layers and Output layers, not including Input layer.

Multi-Layer Perceptron (MLA) is a genrealization of the simple perceptron. And we use <u>**Stochastic Gradient Descent**</u> if we use soft threshould that is smooth and differentiable.

Learning Model: 

$\bullet$ 		Hypothesis Set
$$
h(X)=x_1^{(L)}
$$


$\bullet$ 		Learning Algorithm

​			$\bullet$ 	Error measure
$$
E_{in}=\frac{1}{N}\sum_{i=1}^{N}e_i
$$
​			$\bullet$ 	Searching for the best hypothesis: Stochastic Gradient Descent (SGD) 
$$
w\leftarrow\;w-\lambda\Delta_{in}(w)
$$

##### Back Propogation

<img src="/Users/Zhang/Library/Application Support/typora-user-images/image-20200304234909104.png" alt="image-20200304234909104" style="zoom: 67%;" />

Gradient for $w_{ij}^{(l)}$ :
$$
\frac{\partial e}{\partial w_{ij}^{(l)}} = \frac{\partial e}{\partial s_j^{(l)}}\times \frac{\partial s_j^{(l)}}{\partial w_{ij}^{(l)}}=\delta^{(l)}\times x_i^{(l-1)}
$$
where $\delta^{(l)}=\frac{\partial e}{\partial \theta}\times \frac{\partial \theta}{\partial s_j^{(l)}}$ 

For the previous layer $(l-1)$ ,
$$
\delta_i^{(l-1)}=\frac{\partial e}{\partial s_i^{(l-1)}}=\sum_{j=1}^{d^{(l)}}\frac{\partial e}{\partial s_j^{(l)}}\times\frac{\partial s_j^{(l)}}{\partial x_i^{(l-1)}}\times\frac{\partial x_i^{(l-1)}}{\partial s_i^{(l-1)}}\\
=\sum_{j=1}^{d^{(l)}}\delta_j^{(l)}\times w_{ij}^{(l)}\times \theta'(s_i^{(l-1)})\\
=\theta'(s_i^{(l-1)})\times \sum_{j=1}^{d^{(l)}}\delta_j^{(l)}\times w_{ij}^{(l)}
$$
Within one back and forth, the system can calculate and update $x_i^{(l)}$and $\delta_j^{(l)}$ once.



Notice: When $W$ is too large (positive or negative), 



