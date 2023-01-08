# Numerical Analysis

## Chap1. Mathematical Preliminaries

### 1.1 Basic Concepts and Taylor's Theorem

Limit, Continuity, Derivative

- Taylor's Theorem
  *If $f\in C^n[a,b]$ and if $f^{(n+1)}$ exists on the open interval $(a,b)$ , then for any points $c$ and $x$ in the closed interval $[a,b]$,*  

$$
f(x)=\sum_{k=0}^n\frac1kf^{(k)}(c)(x-c)^k+E_n(x)
$$

where, for some point $\xi$ between $c$ and $x$, the error term is 
$$
E_n(x)=\frac1{(n+1)!}f^{(n+1)}(\xi)(x-c)^{(n+1)}
$$

### 1.2 Orders of Convergence and Additional Basic Concepts

We say that the rate of convergence is at least **linear** if there is a constant $c<1$ and an integer $N$ such that
$$
|x_{n+1}-x^*|\leq c|x_{n}-x^*|\ \ (n\leq N)
$$
We say that the rate of convergence is at least **superlinear** if there exiost a sequence $\varepsilon_n$ tending to $0$ and an integer $N$ such that
$$
|x_{n+1}-x^*|\leq \varepsilon_n|x_{n}-x^*|\ \ (n\leq N)
$$
We say that the rate of convergence is at least **quadratic** if there is a constant $C$ and an integer $N$ such that
$$
|x_{n+1}-x^*|\leq C|x_{n}-x^*|^2\ \ (n\leq N)
$$
We say that the rate of convergence is of  **order** $\alpha$ at least if there is a constant $C$ and $\alpha$ and an integer $N$ such that
$$
|x_{n+1}-x^*|\leq C|x_{n}-x^*|^\alpha\ \ (n\leq N)
$$

### 1.3 Difference Equation

- Theorem of Null Space
  *If $p$ is a polynomial and $\lambda$ is a root of $p$, then one solution of the difference equation $p(E)x=0$ is $[\lambda,\lambda^2,\lambda^3,\dots]$. If all the roots of $p$ are simple and nonzero, then each solution of the difference equation is a **linear combination** of such special solutions.*

- Theorem on Basis for Null Space
  *If $p$ is a polynomial ,satisfying $p(0)\neq0$. Then a basis for the null space of $p(E)$ is obtained as follows: With each root $\lambda$ of $p$ hacing multiplicity $k$, associate the $k$ basic solutions $x(\lambda), x'(\lambda),\dots,x^{(k-1)}(\lambda)$, where $x(\lambda) = [\lambda,\lambda^2,\lambda^3,\dots]$.*

- Theorem on Stable Difference Equations
  *For aa polynomial $p$ satisfying $p(0)\neq0$, these properties are equivalent:*
  1. *The difference equation $p(E)x = 0$ is stable.*
  2. *All roots of $p$ satisfy $|z|\leq 1$, adn all multiple roots satisfy $|z|<1$.*

## Chap2. Computer Arithmetic

### 2.1 Floating-Point numbers and Roundoff Errors

correctly rounded 四舍五入

chopping/  truncated 截断

If $x$ is  **rounded** so that $\tilde{x}$ is the n-digit approximation to it, then
$$
|x-\tilde x|\leq\frac12\times10^{-n}
$$

- Marc-32
  sign of the real number $x$ 1bit
  biased exponent (integer e) 8 bits
  mantissa part (real number f) 23 bits
  $$
  x=(-1)^kq\times 2^m
  $$
  where
  $$
  q=(1.f)_2\ and\ m=e-127
  $$

- 

​		the restriction of e is $0<e<(11111111)_2=2^8-1=255$
​		[00000000] means +0, [80000000] means -0.
​		[7F800000] and [FF800000] , means $+\infty, -\infty$ respectively.
​		NaN means Not a Number and are represented by computer words with $e=255$ and $f\neq 0$

- We write $x=(1.a_1a_2\dots a_{23}a_{24}a_{25}\dots)_2\times 2^m$, in which each $a_i$ is either 0 or 1. One nearby machine number is obtained by chopping and the result is $x_-=(1.a_1a_2\dots a_{23})_2\times 2^m$; another nearby machine number is obtained by rounding up and the result is $x_+=((1.a_1a_2\dots a_{23})_2+2^{-23})\times 2^m$
  $|x-x_-|\leq \frac12|x_+-x_-|=\frac12\times 2^{m-23}=2^{m-24}$, $|\frac{x-x_-}{x}|\leq\frac{2^{m-24}}{q\times 2^m}=\frac1q\times2^{-24}\leq2^{-24}$
  By letting $\delta =(x^*-x)/x$, we can write this inequality in the form $fl(x)=x(1+\delta)\ \ |\delta|\leq2^{-24}$

### 2.2 Absolute and Relative Errors: Loss of Signifgicance

- absolute error $|x-x^*|$

- relative error $|\frac{x-x^*}{x}|$

- *If $x$ and $y$ are positive normalized floarting-point binary machine numbers such that $x>y$ and*
  $$
  2^{-q}\leq 1-\frac yx\leq 2^{-p}
  $$
  then at most $q$ and at least $p$ significant binary bits are lost in the subtraction $x-y$.

### 2.3 Stable and Unstable Computations: Conditioning



## Chap3. Solution of Nonlinear Equation

### 3.1 Bisection Method

- Algorithm
  **input** $a, b, M,$ $\delta,$ $\varepsilon$
  $u\longleftarrow f(a)$ 
  $v\longleftarrow f(b)$ 
  $e\longleftarrow b-a$
  **output** $a, b, u, v$

- error analysis
  $$
  |r-c_n|\leq\frac12(b_n-a_n)=2^{-(n+1)}(b_0-a_0)
  $$
  **order: Linear**

  
  

### 3.2 Newton's Method

- Newton's Method begins with an estimate $x_0$ of $r$ and them defines inductively
  $$
  x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}\ \ \ \ (n\geq 0)
  $$

- Algorithm
  **input** $x, M$ 
  $y\longleftarrow f(x)$
  **output** $0, x, y$
  **for** $k=1$ **to** $M$ do
      $x\longleftarrow x-f(y)/f'(x)$
      $y\longleftarrow f(x)$
      **output** $k, x, y$
  **end do**

- error analysis
  $$
  \begin{split}
  e_{n+1}&=x_{n+1}-r=x_n-\frac{f(x_n)}{f'(x_{n})}-r\\
  &=e_n-\frac{f(x_n)}{f'(x_{n})}=\frac{e_nf'(x_n)-f(x_n)}{f'(x_n)}
  \end{split}
  $$
  because $0=f(r)=f(x_n-e_n)=f(x_n)-e_nf'(x_n)+\frac12e_n^2f''(\xi_n)$,
  $$
  e_nf'(x_n)-f(x_n)=\frac12f''(\xi_n)e_n^2, e_{n+1}=\frac12\frac{f''(\xi_n)}{f'(x_n)}\approx\frac12\frac{f''(r)}{f'(r)}e_n^2=Ce_n^2
  $$
  **order: Quadratic**

- Let $f''$ be continuous and let $r$ be a simple zero of $f$. Then there is a neighborhood of $r$ and a constant $C$ such that if Newton's method is started in that neighborhood, the successive points become steadily closer to $r$ and satisfy
  $$
  |x_{n+1}-r|\leq C(x_n-r)^2\ \ \ \ (n\geq 0)
  $$

- 

- *If $f$ belongs to $C^2(\mathbb R)$, is increasing, is convex, and has a zero, then the zero is unique, and the Newron iteration will converge to it from any starting point.*

  - For implicit function, we have equation $G(x,y)=0$, defines $y$ as a function of $x$. If $x$ is prescribed, then the equation$G(x,y) = 0$ can be solved for $y$ using Newrton's method.
    $$
    y_{k+1}=y_k-G(x, y_k)/\frac{\partial G}{\partial y}(x,y_k)
    $$
    

### 3.3 Secant Method

Steffensen's iteration
$$
x_{n+1}=x_n-\frac{[f(x_n)]^2}{f(x_n+f(x_n))-f(x_n)}
$$
We use **difference quotient** to replace $f'(x)$ in the Newton's method,
$$
f'(x_n)\approx \frac{f(x_n)-f(x_{n-1})}{x_n-x_{n-1}}
$$
Then, we got **secant method**, the fomula is
$$
x_{n+1}=x_n-f(x_n)[\frac{x_n-x_{n-1}}{f(x_n)-f(x_{n-1})}]\ \ \ (n\geq 1)
$$

- Algorithm
  **input** $a,b, M,\delta, \varepsilon$ 
  $fa\longleftarrow f(a);\ fb\longleftarrow f(b)$
  **output** $0, a, fa$
  **output** $1, b, fb$
  **for** $k=2$ **to** $M$ do
      if $|fa|>|fb|$ then
  			$a\leftrightarrow b; fa\leftrightarrow fb$
      end if
      $s\longleftarrow (b-1)/(fb-fa)$
      $b\longleftarrow a$
      $fb\longleftarrow fa$
      $a\longleftarrow a- fa*s$
      $fa\longleftarrow f(a)$

  ​    **output** $k, a, fa$
  ​    **if** $|fa|<\varepsilon$ **or** $|b-a|<\delta$ **then stop**
  **end do**

- error analysis
  $$
  \begin{split}
  e_{n+1}&=x_{n+1}-r=[f(x_n)x_{n-1}-f(x_{n-1})x_n]/[f(x_n)-f(x_{n-1})]-r\\
  &=[f(x_n)e_{n-1}-f(x_{n-1})e_n]/[f(x_n)-f(x_{n-1})]\\
  e_{n+1}&=[\frac{x_n-x_{n-1}}{f(x_n)-f(x_{n-1})}][\frac{f(x_n)/e_n-f(x_{n-1})/e_{n-1}}{x_n-x_{n-1}}]e_ne_{n-1}\\
  &=\frac1{f'(r)}\frac{\frac12(e_n-e_{n-1})f''(r)+O(e_{n-1})^2}{e_n-e_{n-1}}\\
  &\approx \frac12\frac{f''(r)}{f'(r)}e_ne_{n-1}=Ce_ne_{n-1}
  \end{split}
  $$

  $$
  |e_{n+1}|\approx A|e_n|^{(1+\sqrt 5)/2}
  $$

  **order: $(1+\sqrt 5)/2$**

### 3.4 Fixed Points and Functional Iteration

In fact, we need to find a function $F(x)$ and generate a sequence of points computed by 
$$
x_{n+1} = F(x_n)\ \ \ (n\geq0)
$$
In Newton's method, $F(x)=x-\frac{f(x)}{f'(x)}$, and in Setffensen's method, $F(x)=x-\frac{[f(x)]^2}{f(x+f(x))-f(x)}$. If $F$ is continuous, then
$$
F(s) = F(\lim_{n\to\infty}x_n)=\lim_{n\to\infty}F(x_n)=\lim_{n\to\infty}s
$$
 The theorem to be proved concerns **contractive mappings**. A mapping (or dunction) $F$ is said to be **contractive** if there exists a number $\lambda$ less than 1 such that
$$
|F(x)-F(y)|\leq \lambda|x-y|
$$

-  Contractive Mapping Theorem
  *Let $C$ be a closed subset of the real line. If $F$ is a contractive mapping of $C$ into $C$, then $F$ has a unique fixed point. Moreover, this fixed point is the limit of every sequence obtained from $x_{n+1} = F(x_n)$ with a starting point $x_0\in C$.*

- error analysis
  suppose that $F$ has a fixed point, $s$, and the sequence $[x_n]$ has been defined by $x_{n+1}=F(x_n).$ Let $e_n=x_n-s$, if $F'$ exists and is continuous, then by the Mean-Value Theorem, $x_{n+1}-s=F(x_n)-F(s)=F'(\zeta_n)(x_n-s)$, or, $e_{n+1}=F'(\zeta_n)e_n$ where $\zeta$ is a point between $x_n$ and $s$. Assume $q$ is an integer satisfying $F^{(k)}(s)=0,1\leq k<q, \text{but}\ F^{(k)}\neq 0.$ We have
  $$
  \begin{split}
  e_{n+1}&=x_{n+1}-s=F(x_n)-F(s)\\
  &=e_nF'(s)+\dots+\frac1{(q-1)!}e^{q-1}_nF^{(q-1)}(s)+\frac1{q!}e_n^{(q)}(\zeta_n)\\
  &=\frac1{q!}e_n^{(q)}(\zeta_n)
  \end{split}
  $$
  We define the **order of convergence ** to be the largest reqal number $q$ such that the limit
  $$
  \lim_{n\to\infty}\frac{|x_{n+1}-s|}{|x_n-s|^q}.
  $$
  

## Chap4. Solving Systems of Linear Equations

### 4.1 Matrix Algebra

- *If one system of equations is obtained from another by a finite sequence of elementary operations, then the two systems are equivalent.*
- *A square matrix can possess at most one right inverse.*
- *If $A$ and $B$ are square matrices such that $AB=I$, then $BA=I$*.

- If each product $A_{is}B_{sj}$ can be formed, and if $C_{ij}=\sum_{s=1}^n A_{is}B_{sj}$, then $C=AB$.

### 4.2 $LU$ and Cholesky Factorizations

#### Easy-to-Solve Systems

Assume we have a system of linear equations $Ax=b$ to be solved.

If $A$ has a **lower triangular stracture**, the system can be solved with **forward subsitition**:

**input** $n, (a_{ij}), (b_i)$
**for** $i=1$ **to** $n$ **do**
	$x_i\longleftarrow (b_i-\sum_{j=1}^{i-1}a_{ij}x_j)/a_{ii}$
**end do**
**ouput** $(x_i)$

If $A$ has a **upper triangular stracture**, the system can be solved with **backward subsitition**:

**input** $n, (a_{ij}), (b_i)$
**for** $i=n$ **to** $1$ **step** $-1$ **do**
	$x_i\longleftarrow (b_i-\sum_{j=i+1}^{n}a_{ij}x_j)/a_{ii}$
**end do**
**ouput** $(x_i)$

#### $LU$-Factorizations

Suppose that $A$ can be factored into the product fo a lower triangular matrix $L$ adn an upper triangular matrix $U$: $A=LU$. Then, to solve the system of equations $Ax=b$, it's enough to solve this probelm in two stages: 
1. $Lz=b\ \ \text{Solve for} z$; 2. $Ux=z\ \ \text{Solve for }x$

We assume
$$
L=\left[\begin{array}{ccccc}
l_{11} & 0 & 0 & \cdots & 0 \\
l_{21} & l_{22} & 0 & \cdots & 0 \\
l_{31} & l_{32} & l_{33} & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \cdots \\
l_{n1} & l_{n2} & l_{n3} & \cdots & l_{nn}
\end{array}\right], \ \ U=\left[\begin{array}{ccccc}
u_{11} & u_{12} & u_{13} & \cdots & u_{1n} \\
0 & u_{22} & u_{23} & \cdots & u_{2n} \\
0 & 0 & u_{33} & \cdots & u_{3n} \\
\vdots & \vdots & \vdots & \ddots & \cdots \\
0 & 0 & 0 & \cdots & u_{nn}
\end{array}\right]
$$
We start with the formula for matrix multiplication
$$
a_{ij} = \sum_{s=1}^n l_{is}u_{sj}=\sum_{s=1}^{\min(i,j)}l_{is}u_{sj}\\
a_{kk} = \sum_{s=1}^{k-1} l_{ks}u_{sk} + l_{kk}u_{kk}\\
a_{kj} = \sum_{s=1}^{k-1} l_{ks}u_{sj} + l_{kk}u_{kj}\ \ \ (k+1\leq j\leq n)\\
a_{ik} = \sum_{s=1}^{k-1} l_{is}u_{sk} + l_{ik}u_{kk}\ \ \ (k+1\leq i\leq n)\\
$$


**Doolittle's fractorization**
$L$ is unit lower triangular ($l_{ii}=1\text{ for }1\leq i\leq n$).

**input** $n, (a_{ij})$
**for** $k=1$ **to** $n$ **do**
	$l_{kk}\longleftarrow 1$
	**for** $j=k$ **to** $n$ **do**
		$u_{kj}\longleftarrow a_{kj}-\sum_{s=1}^{k-1}l_{ks}u_{sj}$
	**end do**
	**for** $i=k+1$ **to** $n$ **do**
		$l_{ik}\longleftarrow (a_{ij}-\sum_{s=1}^{k-1}l_{is}u_{sk})/u_{kk}$
	**end do**
**end do**
**output** $(l_{ij}), (u_{ij})$



**Crout's fractorization**
$U$ is unit upper triangular ($u_{ii}=1\text{ for }1\leq i\leq n$).

**input** $n, (a_{ij})$
**for** $k=1$ **to** $n$ **do**
	$u_{kk}\longleftarrow 1$
	**for** $i=k$ **to** $n$ **do**
		$l_{ik}\longleftarrow a_{ik}-\sum_{s=1}^{k-1}l_{is}u_{sk}$
	**end do**
	**for** $i=k+1$ **to** $n$ **do**
		$u_{kj}\longleftarrow (a_{kj}-\sum_{s=1}^{k-1}l_{ks}u_{sj})/l_{kk}$
	**end do**
**end do**
**output** $(l_{ij}), (u_{ij})$

- *If all $n$ leading principal minors of  the $n\times n$ matrix $A$ are nonsingular, then $A$ has an $LU$-decomposition.*

#### Cholesky Factorization

- *If $A$ is a real, symmetric, and positive definite matrix, then it has a unique factorization, $A=LL^T,$ in which $L$ is lower triangular with a positive diagonal.*
  Pf. $LU=A= A^T=U^TL^T, U(L^T)^{-1} = L^{-1}U^T$, consequently, there is a diagonal matrix $D$ such that $U(L^T)^{-1}=D$. Hence, $U=DL^T$ and $A=LDL^T$, let $\tilde{L}=LD^{1/2}, A=\tilde{L}\tilde{L}^T$.

- Algorithm
- **input** $n, (a_{ij})$
  **for** $k=1$ **to** $n$ **do**
  	$l_{kk}\longleftarrow (a_{kk}-\sum_{s=1}^{{k-1}}l_{ks}^2)^{{1/2}}$
  	**for** $i=k$ **to** $n$ **do**
  		$l_{ik}\longleftarrow (a_{ik}-\sum_{s=1}^{k-1}l_{is}l_{ks})/l_{{kk}}$
  	**end do**
  **end do**
  **output** $(l_{ij})$

### 4.3  Pivoting and Constructing an Algorithm

#### Basic Guassian Elimination：

- The k-th step: row $i$ - row $k$ $\times $ $\frac{a_{ik}^{(k)}}{a_{kk}^{(k)}}$ . 
- Actually, left multiply the matrix

$$
P_k=\left[\begin{array}{cccccc}
1 &  &  &  &  &  \\
 & \ddots &  &  &  &  \\
 &  & 1 &  &  &  \\
 &  & -l_{k+1,k}^{(k)} & 1 &  &  \\
 &  & \cdots &  & \ddots &  \\
 &  & -l_{n,k}^{(k)} &  &  & 1\end{array}\right], \text{where }l_{ik}^{(k)} = \frac{a_{i,k}^{(k)}}{a_{k,k}^{(k)}}, i=k+1,\cdots,n.
$$

 Use these lower triangular matrix $P_k$ to left-multiply $A$ and generate a upper triangular matrix $P_{n-1}\dots P_2P_1A$.

Essentially, the Guassian Elimination is a process to do $LU$-decomposition.

- If all the pivot elements $a_{kk}^{(k)}$ are nonzero in the process just described, then $A=LU$.

#### Pivoting

Failure will emerg if $a_{kk}$ is too small compared to other elements $a_{kj}$ in the same row $k$

Thus, we can choose the pivot of every step with the following algorithm

**input** $n, (a_{ij}), (p_i)$
**for** $k=1$ **to** $n-1$ **do**
	**for** $i=k+1$ **to** $n$ **do**
		$z\longleftarrow a_{p_i,k}/a_{p_k, k}$
		$a_{p_i, k}\longleftarrow 0$
		**for** $j=k+1$ **to** $n$ **do**
			$a_{p_{i},j}\longleftarrow a_{p_i,j}-za_{p_k,j}$
		**end do**
	**end do**
**end do**
**output** $(a_{ij})$

#### Gaussian Elimination with Scaled Row Pivoting

The algorithm consists of two parts: a **factorization phase** (also called **forward elimination**) and a **solution phase** (**involving updating** and **back substitution**).

##### Factorization phase

produce an $LU$-decomposition of $PA$ (i.e., $PA = LU$) where $P$ is a permutation matrix derived from the permutation array $p$ ($PA$ is obtained from $A$ by permuting its rows).

##### Solution phase: 

solve the permuted system $PAx = Pb$ $(PA = LU$; $|P| \neq 0$), $Lz = Pb$; $Ux = z$.

<img src="/Users/guyanwu/Library/Application Support/typora-user-images/image-20211222201250947.png" alt="image-20211222201250947" style="zoom: 33%;" /><img src="/Users/guyanwu/Library/Application Support/typora-user-images/image-20211222201335948.png" alt="image-20211222201335948" style="zoom: 33%;" /><img src="/Users/guyanwu/Library/Application Support/typora-user-images/image-20211222201440447.png" alt="image-20211222201440447" style="zoom:33%;" />

#### Gaussian Elimination with Complete Pivoting

选用最大绝对值的元素，但是由于需要计算很多额外的计算，一般不选用

#### Factorization $PA=LU$

- Let $p_1,p_2,\dots,p_n$ be the indives of the rows in the order in which they become pivot rows. Let $A^{(1)}=A$, and define $A^{(2)},A^{(3)},\dots,A^{(n)}$ recursively by the formula
  $$
  a_{p_{i},j}^{(k+1)}=\begin{cases}
  a_{p_i,j}^{(k)}\ \ \ \ &\text{ if } i\leq k \text{ or } i>k>j\\
  a_{p_i,j}^{(k)}-(a_{p_i,k}^{(k)}/a_{p_k,k}^{(k)})p_{p_k,j}^{(k)}\ \ \ \ &\text{ if } i> k \text{ and } j>k\\
  a_{p_i,k}^{(k)}/a_{p_k,k}^{(k)}\ \ \ \ &\text{ if } i> k \text{ and } j=k
  \end{cases}
  $$
  
- *Define a permutation matrix $P$ whose elements are $P_{ij}=\delta_{p_ij}$. Define an upper triangular matrix $U$ whose elements are $u_{ij} = a_{p_i,j}^{(n)}$ if $j\geq i$. Define a unit lower triangular matrix $L$ whose elements are $l_{ij}=a_{p_i,j}^{(n)}$. Then $PA = LU$.*

- *If the factorization $PA = LU$ is produced from the Gaussian algorithm with scaled row pivoting, then the solution of $Ax = b$ is obtained by first solving $Lz = Pb$ and then solving $Ux = z$. Similarly, the solution of $y^TA = c^T$ is obtained by solving $U^Tz = c$ and then solving $L^T Py = z$.*

- If Gaussian elimination is used with scaled row pivoting, then the solution of the system $Ax = b$, with fixed $A$, and $m$ different vectors $b$, involves approximately
  $$
  \frac13n^3+(\frac12+m)n^2
  $$
  long operations (multiplications and divisions).

#### Diagonally Dominant Matrices

The **diagonally dominant matrices** have the property that
$$
|a_{ii}|>\sum_{j=1,\ j\neq i}^n |a_{ij}|\ \ \ \ (1\leq i\leq n)
$$

- *Every diagonally dominant matrix $M\in \mathbb{R}^{n\times n}\ \ (n \geq 2)$ is nonsingular.*
- *Gaussian elimination without pivoting preserves the diagonal dominance of a matrix.*==Pf==
  - Corollary 1
    Every diagonally dominant matrix is nonsingular and has an $LU$-factorization.
  - Corollary 2
    If the scaled row pivoting version of Gaussian elimination recomputes the scale array after each major step is applied to a diagonally dominant matrix, then the pivots will be the natural ones: $1, 2, \dots ,  n$. Hence, the work of choosing the pivots can be omitted in this case.

### 4.4 Norms and Analysis of Errors

On a vector space$ $V, a **norm** is a funtion $\|\cdot\|$ from $V$ to the set of nonnegative reals that obeys these three postulates

1. $\|x\|>0 \text{ if } x\neq 0, x\in V$
2. $\|\lambda x\|=|\lambda|\|x\|\text{ if }\lambda\in\mathbb R, x\in V$
3. $\|x+y\|\leq\|x\|+\|y\|\text{ if }x,y\in V$

And we consider these norms:

**Euclidean $l_2$-norm**
$$
\|x\|_2=(\sum_{i=1}^nx_i^2)^{1/2}, \text{   where }x=(x_1,x_2,\dots,x_n)^T
$$
**$l_{\infty}$-norm**
$$
\|x\|_{\infty} = \max_{1\leq i\leq n}|x_i|
$$
**$l_{\infty}$-norm**
$$
\|x\|_{1} = \sum_{i=1}^n|x_i|
$$
If a vector norm $\|\cdot\|$ has been specified, the matrix norm **subordinate** to it is defined by
$$
\|A\|=\sup\{\frac{\|Au\|}{\|u\|}:u\in R\}
$$


- **If $\|\cdot\|$ is any norm on $\mathbb R^n$, then the equation**
  $$
  \|A\|=\sup_{\|u\|=1}\{\|Au\|:u\in\mathbb R^n\}
  $$
  **defines a norm on the linear space of all $n\times n$ matrices.**

Then, we illustrate some properties of the matrix norm

1. $\|\lambda A\|=|\lambda|\|A\|$
2. $\|A+B\|\leq\|A\|+\|B\|$
3. $\|Ax\|\leq\|A\|\|x\|,\ \ (x\in\mathbb R^n)$

We introduce some special matrix norms

1. 1范数（列和范数）：$\|A\|_{1}=\max_{1\leq j\leq n}\sum_{k=1}^n|a_{kj}|$
2. 无穷范数（行和范数）：$\|A\|_{\infty}=\max_{1\leq i\leq n}\sum_{k=1}^n|a_{ik}|$
3. 2范数：$\|A\|_2=\max_{1\leq i\leq n}|\sigma_i|=\sqrt{\rho (A^TA)}$
4. F范数：$\|A\|_F=(\sum_{i=1}^n\sum_{j=1}^na_{ij}^2)^{1/2}$

#### Condition Number

For the equation $Ax=b$, if $A^{-1}$ is perturbed to obtain a new matrix $B$, then the solution $x=A^{-1}b$ is perturbed to become a new vector $\tilde{x}=Bb$. The perturbation in relative term is 
$$
\|x-\tilde x\|=\|x-Bb\|=\|x-BAx\|=\|(I-BA)x\|\leq\|I-BA\|\|x\|\\
\frac{\|x-\tilde x\|}{\|x\|}\leq\|I-BA\|
$$
Suppose that the vector $b$ is perturbed to obtain a vector $\tilde b$. If $x$ and $\tilde x$ satisfy $Ax=b$ and $A\tilde x=\tilde b$, the relative terms is
$$
\|x-\tilde x\|=\|A^{-1}b-A^{-1}\tilde b\|=\|A^{-1}(b-\tilde b)\|\leq\|A^{-1}\|\|b-\tilde b\|\\
=\|A^{-1}\|\|Ax\|\frac{\|b-\tilde b\|}{\|b\|}\leq \|A^{-1}\|\|A\|\|x\|\frac{\|b-\tilde b\|}{\|b\|}
$$
Hence, 
$$
\frac{\|x-\tilde x\|}{\|x\|}\leq \kappa(A)\frac{\|b-\tilde b\|}{\|b\|}, \ \ \text{ where     }\kappa(A)\equiv\|A\|\cdot\|A^{-1}\|
$$
The number $\kappa(A)$ is called a **condition number** of the matrix $A$.

- *In solving systems of equations $Ax=b$, the condition number $\kappa(A)$, the residual vector $r$, and the eooro vector $e$ satisfy the following inequality:*
  $$
  \frac1{\kappa(A)}\frac{\|r\|}{\|b\|}\leq\frac{\|e\|}{\|x\|}\leq\kappa(A)\frac{\|r\|}{\|b\|}
  $$
  

### 4.5 Neumann Series and Iterative Refinement

#### Neemann Series

- Theorem on Neumann Series
  *If $A$ is an $n\times n$ matrix such that $\|A\|<1$, then $I-A$ is invertible, and*
  $$
  (I-A)^{-1}=\sum_{k=0}^\infty A^k
  $$

- Theorem on Invertible Matrices
  *If $A$ and $B$ are $n\times n$ matrices such that $\|I-AB\|<1$, then $A$ and $B$ are invertible. Furthermore, we have*
  $$
  A^{-1}=B\sum_{k=0}^\infty(I-AB)^k,\ \text{ and }\ B^{-1}=\sum_{k=0}^\infty(I-AB)^k A
  $$
  

#### Iterative Refinement

For an approximate solution $x_0$ of the equation $Ax=b$, the precise solution $x$ is given by
$$
x=x_0+A^{-1}(b-Ax)=x_0+e_0
$$
We do not want to compute $A^{-1}$, but the vector $e_0$ can be obtianed by solving the equation
$$
Ae_0=r_0
$$
The algorithm can be disputed as followed
$$
\begin{cases}
r^{(k)} &=&b-Ax^{(k)}\\
Ae^{(k)}&=&r^{(k)}\\
x^{(k+1)}&=&x^{(k)}+e^{(k)}
\end{cases}
$$


- *If $\|I-BA\|<1$, then the method of iterative improvement given by the above alogrithm produces the sequence of vectors*
  $$
  x^{(m)}=B\sum_{k=0}^m(I-AB)^kb
  $$
  *where $B$ satisfying $x^{(0)}=Bb$ is an approximation of $A^{{-1}}$*

  ##### 

### 4.6 Solution of Equaitons by Iterative Methods

**direct** methods to solve the matrix problem $Ax=b$: Gaussian algorirthm, a finite numbers of steps and produce a completely accurate solution

**indirect** methods: produces a sequence of vectors that ideally converges to the solution

We suppose $A=D+L+U$ where $D$ is a diagonal matrix, $L$ is a lower triangular matrix and $U$ is an upper triangular matrix.

#### Basic idea

A general type of iterative process for solving the system $Ax=b$ can be described as follows: A certain matrix $Q$, called the **splitting matrix**, is prescribed, and the original problem is rewritten in the equivalent form
$$
Qx=(Q-A)x+b,  \ \ \text{ or }\ \ Qx^{(k)}=(Q-A)x^{(k-1)}+b\ \ (k\geq1)
$$
If we assume that $A$ and $Q$ are nonsingular, then we get a iterative formula that
$$
x^{(k)}=(I-Q^{-1}A)x^{(k-1)}+Q^{-1}b
$$
To analysis whether it will converge to the real solution
$$
x^{(k)}-x=(I-Q^{-1}A)(x^{(k-1)}-x)\\
\|x^{(k)}-x\|\leq\|I-Q^{-1}A\|\|x^{(k-1)}-x\|\leq \|I-Q^{-1}A\|^k\|x^{(0)}-x\|\\
\lim_{k\to \infty}\|x^{(k)}-x\|=0
$$

- *If $\|I-Q^{-1}A\|<1$ for some subordinate matrix norm, then the sequence produced by the above algorithm converges to the solution of $Ax=b$ for any initial vector $x^{0}$.*

#### Richardson Method

Let $Q=I, G=I-A$, and the iterative formula is
$$
x^{(k)}=(I-A)x^{(k-1)}+b=x^{(k-1)}+r^{(k-1)}
$$

#### Jacobi Method

Let $Q=D, G=-D^{-1}(L+U)$.

Thus, the generic element of $Q^{-1}A$ is $a_{ij}/a_{ii}$. the diagnal elements of this matrix are all 1, and hence, 
$$
\|I-Q^{-1}A\|_\infty = \max_{1\leq i\leq n}\sum_{j=1,\ j\neq i}^n|a_{ij}/a_{ii}|
$$

- *If $A$ is diagonally dominant, then the sequence produced by the Jacobi iteration coneerges to the solution of $Ax=b$ for any starting vector.*

#### Analysis

For the equation
$$
x^{(k)}=Gx^{(k-1)}+c
$$
we want to find a necessary and sufficient condition on $G$ so that the iteration will converge for any starting vector.

- *Every square matrix is similar to an (possibly complex) upper triangular matrix whose off-diagonal elements are arbitratily small.*

- *The spectral radius function satisfires the equation* 
  $$
  \rho(A)=\inf_{\|\cdot\|}\|A\|
  $$
  *in which the infimum is taken over all subordinate matrix norms.*

- Thus, $\rho(A)\leq \|A\|$ for any subordinate matrix norm $\|\cdot\|$.

- *In order that the iteration formula*
  $$
  x^{(k)}=Gx^{(k-1)}+c
  $$
  *produce a sequence converging to $(I-G)^{-1}c$, for any starting vector $x^{(0)}$, it is necessary and sufficient that the spectral radius of $G$ be less than 1.*

- *The iteration formula $Qx^{(k)}=(Q-A)x^{(k-1)}+b$  will produce a sequence converging to the solution of $Ax=b$, for any $x^{(0)}$, if $\rho(I-Q^{-1}A)<1$.*



#### Gauss-Seidel Method

Let $Q$ be the lower triangular part of $A$, including th diagonal, that is, $Q=D+L, G = -(D+L)^{-1}U$.

-  *If $A$ is diagonally dominant, then the Gauss-Seidel method converges for any starting vector.*

#### SOR Method

Successive over-relaxation

- *In SOR method, suppose that the splitting matrix $Q$ is chosen to be $\alpha D-C$, where $\alpha$ is a real parameter, $D$ is any positive definite Hermitian matrix, and $C$ is any matrix satisfying* $C+C^*=D-A$. *If $A$ is positive definite Hermitian, if$Q$ is nonsingular, and if $\alpha>\frac12$, then the SOR iteration converges for any starting vector.*
- Let $Q=\omega^{-1}D+ L, G=(D+\omega L)^{-1}((1-\omega)D-\omega U)$
- Note that this method converges when $0<\omega<2$.



#### Extrapolation

Consider the iteration formula
$$
x^{(k)} = Gx^{(k-1)}+c
$$
We introduce a parameter $\gamma\neq 0$ and then we gain
$$
x^{(k)}=\gamma(Gx^{(k-1)}+c)+(1-\gamma)x^{(k-1)}= G_\gamma x^{(k-1)}
+\gamma c
$$
where $G_\gamma=\gamma G+(1-\gamma )I$

- *If $\lambda$ is an eigenvalue of a matrix $A$ and if $p$ is a polynomial, then $p(\lambda)$ is an eigenvalue of $p(A)$.*
- *If the only information available about the eigenvalues of $G$ is that they lie in the interval $[a,b]$, and if $1\notin [a,b]$, then the best choice for $\gamma$  is $2/(2-a-b)$. With this value of $\gamma$, $\rho(G_\gamma)\leq 1-|\gamma|d$, where $d$ is the distance from $1$ to $[a,b]$.*
- The extrapolation process or technique just discussed can be applied to methods that are not convergent themselves. All that is required is that the eigenvalues of $G$ be real and lie in an interval that does not contain $1$.



#### Chebyshew Acceleration

Suppose we have calculated the vectors $x^{(1)},\dots,x^{(k)}$ using the iteration $x^{(k)}=Gx^{(k-1)}+c$, and we want a linear combination of them to be a better approximation to the solution than $x^{(k)}$. We assume $a_0^{(k)}+a_1^{(k)}+\dots+a_k^{(k)}=1$, and set
$$
u^{(k)}=\sum_{i=0}^ka_i^{(k)}x^{(i)}
$$
By familiar techniques, we obtain
$$
u^{(k)}-x=\sum_{i=0}^ka_i^{(k)}(x^{(i)}-x)=\sum_{i=0}^ka_i^{(k)}G^i(x^{(0)}-x)=P(G)(x^{(0)}-x)\\
\|u^{(k)}-x\|\leq\|P(G)\|\|x^{(0)}-x\|
$$
If the eigenvalues $\mu_i$ of $G$ lie within some bounded set $S$ in the complex plane, then by the previous analysis,
$$
\rho(P(G))=\max_{1\leq i\leq n}|P(\mu_i)|\leq \max_{z\in S}|P(z)|
$$
Then, this reduces to
$$
\min_{p\in \mathbb P_k,\ p(1)=1}\rho(p(G)) \leq \min_{p\in \mathbb P_k,\ p(1)=1} \max_{z\in S}|p(z)|
$$
where $\mathbb P_k$ denotes to the set of all the real polynomials with degree less than $k$. It's a standard problem in **approximation problem.**



### 4.7 Steepest Descent and Conjugate Gradient Methods

- If $A$ is symmetric and positive definite, then the problem of solving $Ax=b$ is equivalent to the probelm of minimizing the quadratic form
  $$
  q(x)=<x,Ax>-2<x,b>
  $$
  To prove this, we discuss how the function $q$ behaves along a one-dimensional ray
  $$
  q(x+tv)=q(x)+2t<v,Ax-b>+b^2<v,Av>\\
  \frac{d}{dt}q(x+tv)=2<v,Ax-b>+2t<v,Av>
  $$
  The minimum of $q$ along the ray occurs when
  $$
  \hat t=<v,b--Ax>/<v,Av>
  $$
  Using this value $\hat v$, we compute the minimum of $q$ on the ray :
  $$
  q(x+\bar t v)=q(x)-<v,b-Ax>^2/<v,Av>
  $$
  The value will reduce unless $v$ is **orthogonal** to the residual, that is, $<v,b-Ax>$.

- The iteration of these method can be written as
  $$
  x^{(k+1)}=x^{(k)}+t_kv^{(k)},\ \ \text{ where }\ t_k=\frac{<v^{(k)}, b-Ax^{(k)}>}{<v^{(k)},Av^{(k)}>}
  $$
  ​	

#### Steepest Descent

Use the negative gradient of $q$ as the direction of $v^{(k)}$, that is, the residual, $r^{(k)}=b-Ax^{(k)}.$

But the speed of steepest descent is not enough.



#### Conjugate Directions

Assuming that $A$ is an $n\times n$ symmetric and positive definite matrix, suppose that a set of vectors $\{u^{(1)},u^{(2)},\dots,u^{(n)}\}$ is provided and has the property $<u^{(i)},Au^{(j)}>=\delta_{ij}.$ We observe that the $A-$orthonormality condition can be expressed as a matrix equation
$$
U^TAU=I
$$
where $U$ is the $n\times n$ matrix whose columns are $u^{(1)},u^{(2)},\dots,u^{(n)}.$ From this it is clear that $A$ and $U$ are nonsingular, and the columns $u^{(1)},u^{(2)},\dots,u^{(n)}$ form a basis for $\mathbb R^n$. 

- *Let $\{u^{(1)},u^{(2)},\dots,u^{(n)}\}$ be an $A$-orthonormal system. Define*
  $$
  x^{(i)}=x^{(i-1)}+<b-Ax^{(i-1)},u^{(i)}>u^{(i)}
  $$
  *in which  $x^{(0)}$ is an arbitrary point of $\mathbb R^n$. Then $Ax^{(n)}=b$.*

- Let $\{v^{(1)},v^{(2)},\dots,v^{(n)}\}$ be an $A$-orthogonal system of nonzero vectors for a symmetric and positive definite $n\times n$ matrix $A$. Define
  $$
  x^{(i)}=x^{(i-1)}+\frac{<b-Ax^{(i-1)},v^{(i)}>}{<v^{(i)}, Av^{(i)}>}v^{(i)}
  $$
  in which $x^{(0)}$ is arbitrary. Then $Ax^{(n)}=b.$

- orthogonal 不需要基向量模长为1， orthonormal需要



#### Conjugate gradient Method

Suppose $A$ is a symmetric and positive definite matrix. The search directions $v^{(i)}$ are chosen one by one during the iterative process and form an $A$-orthogonal system. In fact, the residuals $r^{(i)}=b-Ax^{(i)}$ form an orthogonal system in the ordinary sense, that is, $<r^{(i)},r^{(j)}>=0$ if $i\neq j$.

**Algorithm**

**input $x^{(0)}, M, A, b, \varepsilon$**
$r\leftarrow b-Ax^{(x)}$
$v^{(0)}\leftarrow r^{(0)}$
**output $0, x^{(0)}, r^{(0)}$**
**for $k=0$ to $M-1$ do**
	**if $v^{(k)}=0$ then stop**
	$t_k\leftarrow <r^{(k)},r^{(k)}>/<v^{(k)},Av^{(k)}>$
	$x^{(k+1)}\leftarrow x^{(k)}+t_kv^{(k)}$
	$r^{(k+1)}\leftarrow r^{(k)}-t_kAv^{(k)}$
	**if $\|r^{(k+1)}\|^2_2<\varepsilon$ then stop**
	$s_k \leftarrow <r^{(k+1)},r^{(k+1)}>/<r^{(k)},r^{(k)}>$
	$v^{(k+1)}\leftarrow r^{(k+1)}+s_kv^{(k)}$
	**output $k+1, x^{(k+1)},r^{(k+1)}$**
**end do**

#### Theorem on Conjugate Gradient Algorithm

*In the conjugate gradient algorithm, for any integer $m<n$, if $v^{(0)},v^{(1)},\dots,v^{(m)}$ are all nonzero vectors, then $r^{(i)}=b-Ax^{(i)}$ for  $0\leq i\leq m$, and $\{r^{(0)},r^{(1)},\dots,r^{(m)}\}$ is an orthogonal set of nonzero vectors.*

What's more, we have properties as following:

1. $<r^{(m)},v^{(i)}>=0\ \ \ (0\leq i<m)$
2. $<r^{(i)}, r^{(i)}>=<r^{(i)},v^{(i)}>\ \ \ (0\leq i\leq m)$
3. $<v^{(m)}, Av^{(i)}>=0\ \ \ (0\leq i<m)$
4. $r^{(i)}=b-Ax^{(i)}\ \ \ (0\leq i\leq m)$
5. $<r^{(m)}, r^{(i)}>=0 \ \ \ (0\leq i<m)$
6. $r^{(i)}\neq 0\ \ \ (0\leq i\leq m)$



#### Preconditioned Conjugate Gradient

It would be advantageous to precondition the equation system $Ax=b$ and obtain a new system that is better conditioned than the original system. By this we mean that for some nonsingular matrix $S$, the preconditioned system
$$
\hat A\hat x=\hat b
$$
where
$$
\begin{cases}
&\hat A=S^TAS\\
&\hat x=D^{-1}x\\
&\hat b=S^Tb
\end{cases}
$$
is such that $\kappa(\hat A)<\kappa(A)$. We suppose that the symmetric and positive definite splitting matrix $Q$ can be factored so that $Q^{-1}=SS^T$.	

We write
$$
\begin{cases}
\hat x^{(k)}=S^{-1}x^{(k)}\\
\hat v^{(k)}=S^{-1}v^{(k)}\\
\hat r^{(k)}=\hat b-\hat A\hat x^{(k)}=S^Tb-(S^TAS)(S^{-1}x^{(k)})=S^Tr^{(k)}\\
\tilde r^{(k)}=Q^{-1}r^{(k)}
\end{cases}
$$
, and we have the new iteration that
$$
\hat t_k=<\tilde r^{(k)},r^{(k)}>/<v^{(k)},Av^{(k)}>\\
x^{(k+1)}=x^{(k)}+\hat t_k v^{(k)}\\
r^{(k+1)}=r^{(k)}-\hat t_k Av^{(k)}\\
\hat s_k = <\tilde r^{(k+1)}, r^{(k+1)}>/<\tilde r^{(k)},r^{(k)}>\\
v^{(k+1)} = \tilde r^{(k+1)}+\hat s_k v^{(k)}
$$


 When $Q=I$, it is the conjuagate gradient algorithm.

When $Q=A, S=A^{-1/2},$ unfortunately, it's hard to compute.

We must solve a system of the form $Qx=y$ on each iteration of the preconditioned conjugate gradient algorithm, $Q$ must be selected so that this system is easy to solve. As $Q^{-1}$ becomes a better approximation to $A$, th e preconditioned system becomes better conditioned, and the convergence of the iterative procedure occurs in fewer steps.

On the other hand, fewer steps means more difficult computition.



## Chap5. Selected Topics in Numerical Linear Algebra

### 5.1 Matrix Eigenvalue Problem: Power Method

We assume that the matrix has the following two prosperties:

1. There is a single eigencalue of maximum modulus;
2. There is a linearlly independent set of $n$ eigenvectors.

The eigenvalues $\lambda_1,\dots, \lambda_n$ can be labeled that
$$
|\lambda_1|>|\lambda_2|\geq|\lambda_3|\geq\cdots\geq|\lambda_n|
$$
and a basis $\{u_1,u_2,\dots u_n\}$ for $\mathbb C^n$ such that
$$
Au_j=\lambda_ju_j, (1\leq j\leq n)
$$
Let $x^{(0)}\in\mathbb C^n$, it can be expressed as
$$
x^{(0)}=\alpha_1u_1+\alpha_2u_2+\dots+\alpha_nu_n, (\alpha_1\neq 0)
$$
Then, the iteration formula for the power method is 
$$
\begin{split}
x^{(k)}&=Ax^{(k-1)}\\
&=A^kx^{(0)}\\
&=A^k(\alpha_1u_1+\alpha_2u_2+\dots+\alpha_nu_n)\\
&=\alpha_1A^ku_1+\alpha_2A^ku_2+\dots+\alpha_nA^ku_n\\
&=\alpha_1\lambda_1^ku_1+\alpha_2\lambda_2^ku_2+\dots+\alpha_n\lambda_n^ku_n\\
&=\lambda_1^k[\alpha_1u_1+\alpha_2(\frac{\lambda_2}{\lambda_1})^ku_2+\dots+\alpha_n(\frac{\lambda_n}{\lambda_1})^ku_n]
\end{split}
$$
Since $|\lambda_1|>|\lambda_j|$ for $2\leq j\leq n$, then
$$
(\frac{\lambda_j}{\lambda_1})^k\to 0 \text{ as } k\to \infty
$$
we can write $x^{(k)}$ as $x^{(k)}=\lambda_1^k[\alpha_1u_1+\epsilon^{(k)}]$. To get $\lambda_1$, we can use some linear functional $\varphi$ on $\mathbb C^n$ s.t. $\varphi(u_1)\neq 0$
$$
\varphi(\alpha x+\beta y)=\alpha\varphi(x)+\beta\varphi(y)
$$
Then, $\varphi(x^{(k)})=\lambda_1^k[\varphi(\alpha_1u_1)+\varphi(\epsilon^{(k)})]$, so the ratio
$$
r_k\equiv\frac{\varphi(x^{(k+1)})}{\varphi(x^{(k)})}=\lambda_1\frac{\varphi(\alpha_1u_1)+\varphi(\epsilon^{(k+1)})}{\varphi(\alpha_1u_1)+\varphi(\epsilon^{(k)})}\to\lambda_1
$$
<img src="/Users/guyanwu/Library/Application Support/typora-user-images/image-20211229213522803.png" alt="image-20211229213522803" style="zoom:30%;" />

#### Aitken Acceleration

if $|\lambda_2|>|\lambda_3|$, rate of descent comes linear, that is $r_{k+1}-\lambda_1=(c+\delta_k)(r_k-\lambda_1)$, where $|c<1|$ and $\delta_k$ converges to 0.

Construct a new sequence $[s_k]$ by
$$
s_k=\frac{r_kr_{k+2}-r_{k+1}^2}{r_{k+2}-2r_{k+1}+r_k}
$$
Converges faster than $r_k$



REMARK: It’s better to stop the Aitken acceleration process soon after it produces apparently stationary values because subtractive cancellation in the formula will eventually spoil the results



#### Inverse Power Method

- If $λ$ is an eigenvalue of $A$ and if $A$ is nonsingular, then $λ^{−1}$ is an eigenvalue of $A^{−1}$:
- The largest modulus eigenvalue of $A^{-1}$ is the countdown of the smallest modulus eigenvalue of $A$
- Don't take inverse directly, but use LU-decomposition to solve $Ax^{(k+1)}=x^{(k)}$

<img src="/Users/guyanwu/Library/Application Support/typora-user-images/image-20211229214445704.png" alt="image-20211229214445704" style="zoom:50%;" />

### 5.2 Schur's and Gershgorin's Theorems

- 　Similar matrices have the same eigenvalues. For $B=PAP^{{-1}}$
  $$
  det(B-\lambda I)=det(PAP^{-1}-\lambda PP^{-1})=\det(P)det(A-\lambda I)\det(P^{-1})=det(A-\lambda I)
  $$

- A matrix $U\in\mathbb C^{n×n}$ is unitary if $UU^∗ = I$, where $U^∗$ is the conjugate transpose of $U$, i.e., $U^∗ = \bar U^T$ : Matrices $A, B$ are unitarily similar if $B = UAU^∗$ for some unitary matrix U.

- Lemma (1st Lemma on Unitary Matrix)
  For any vector $v \in \mathbb C^n$, the matrix $I − vv^∗$ is unitary if and only if $\|v\|_2 = \sqrt 2$ or  $ v = 0$.

- Lemma (2nd Lemma on Unitary Matrix)
  Let $x, y \in\mathbb C^n$ s.t. $\| x\|_2 = \|y\|_2$ and $\langle x, y\rangle = y^∗x$ is real. Then there exists a unitary matrix $U$ of the form $I − vv^∗$ s.t. $Ux = y$.

- Theorem (Schur’s Theorem )
  Every square matrix is unitarily similar to a triangular matrix.

- Corollary (on Similar Matrix)
  Every square matrix is similar to a triangular matrix

- Corollary (on Unitarily Similar Matrix)
  Every Hermitian matrix is unitarily similar to a diagonal matrix



Theorem (Gershgorin’s Theorem )
The spectrum of an $n × n$ matrix $A = (a_{ij})$  (i.e., the set of of its eigenvalues) is contained in the union of the following $n$ disks, $D_i$, in the complex plane:
$$
D_i = \{z \in \mathbb C : |z −a_{ii}| ≤\sum_{j=1,j\neq i}^n|a_{ij}|\} (1 \le i \le n)
$$
Theorem (on Eigenvalue Disks )
If the matrix $A$ is diagonalized by the similarity transformation $P^{−1}AP$, and if $B$ is any matrix, then the eigenvalues of $A + B$ lie
in the union of the disks
$$
\{λ \in\mathbb C : |λ − λ_i| ≤ \kappa_\infty(P)\|B\|_\infty\}
$$
where $λ_1, λ_2,\dots; λ_n$ are the eigenvalues of $A$, and $κ_\infty(P)$ is the condition number of $P$.

Particularly, if $A$ is Hermitian, then for any matrix $B$, the eigenvalues of $A+B$ lie in the union of the disks
$$
\{\lambda\in\mathbb C:|\lambda-\lambda_i|\leq n\|B\|_\infty\}
$$
where $\lambda_1,\dots,\lambda$ are the eigenvalues of $A$. In this case, $P$ can be chosen to be unitary and hence $\|P\|_\infty\leq \sqrt n$



### 5.3 Orthogonal Factttorizations and Least-Squares Problems

The inner-product notation for complex vectors $x,y\in\mathbb C^n$ is defined as
$$
\langle x,y\rangle=y^*x=\sum_{i=1}^nx_i\bar y_i
$$
and satisfies 

1. $\langle x,x\rangle>0$ if $x\neq 0$
2. $\langle\alpha x+\beta t,z\rangle=\alpha\langle x,z\rangle+\beta\langle y,z\rangle, \forall\alpha,\beta\in\mathbb C$
3. $\langle x,y\rangle=\overline{\langle y,x\rangle}$
4. $\|x+y\|_2^2=\|x\|_2^2+\|y\|_2^2$ if and only if $\langle x,y\rangle=0$



Suppose that $\{v_1, v_2, \dots ,v_n\}$ is an orthonormal basis for $\mathbb C^n$. Then, each element $x \in\mathbb C^n$ has a
unique representation in the form
$$
x=\sum_{i=1}^n c_iv_i\in\mathbb C^n
$$
where $c_i=\langle x, v_i\rangle (1 ≤ i ≤ n)$. Thus,
$$
x=\sum_{i=1}^n \langle x, v_i\rangle v_i\in\mathbb C^n
$$

#### Gram-Schmidt Process

It can be used to obtain orthonormal systems in an inner-product space. Suppose that $[x1, x2,\dots  ]$ is a linearly independent sequence of vectors
in an inner-product space (The sequence can be finite or infinite). We can generate an orthonormal sequence $\{u_1, u_2, \dots,\}$  by the formula
$$
u_k=\frac{x_k-\sum_{i=1}^{k-1}\langle x_k,u_i\rangle u_i}{\|x_k-\sum_{i=1}^{k-1}\langle x_k,u_i\rangle u_i\|_2},\ \ (k\geq 1)
$$

- Theorem (on Gram-Schmidt Sequence)

  The Gram-Schmidt Sequence $[u_1, u_2, \dots, ]$ has the property that $\{u_1, u_2, \dots, u_k\}$ is an orthonormal base for a space spanned by the linearly independent vectors $\{x_1, x_2, \dots , x_k\} (k ≥ 1)$.
  $$
  span\{x_1, x_2, \dots, x_k\} = span\{u_1, u_2,\dots u_k\}
  $$

- Theorem (on Gram-Schmidt Factorization)
  The Gram-Schmidt process, when applied to the columns of an $m × n$$ (m ≥ n)$ matrix $A$ of rank $n$, produces a factorization
  $$
  A_{m×n} = B_{m×n}T_{n×n}
  $$
  in which $B$ is an $m × n$ matrix with orthonormal columns and $T$ is an $n × n$ upper triangular matrix with positive diagonal.

- Theorem (on Modified Gram-Schmidt Factorization)
  If the modified Gram-Schmidt process is applied to the columns of an $m × n$ $(m ≥ n)$ matrix $A$ of rank $n$, the transformed $m × n$ matrix $B$ has an orthogonal set of columns and satisfies
  $$
  A_{m×n} = B_{m×n}T_{n×n}
  $$

  where $T$ is a unit $n × n$ upper triangular matrix whose elements $t_{kj} (j > k)$ are generated in the algorithm. 

#### Least-Squares Problems

An important application of the orthogonal factorizations being discussed is the Least-Squares Problem for a linear system of equations. Consider
$$
Ax=b
$$
where $A$ is $m × n$, $x$ is $n × 1$, $b$ is $m × 1$. Assume the rank of $A$ is $n$; hence, $m ≥ n$. Usually, system will have no solution if
$$
b \notin span\{A_1,\dots,A_n\} \subset \mathbb C^m
$$
In such cases, it is often required to find an $x$ that minimizes the norm of the residual vector, $b − Ax$: The least-squares “solution" of (1) is the vector x that makes $\|b − Ax\|^2$ a minimum. (If $rank(A) = n$; then this $x$ will be unique.)

- Lemma (on the Least-Square Problems)
  If $x$ is a point s.t. $A^∗(Ax − b) = 0$; then $x$ solves the least-square problems.
  Imply that $Ax-b$ is orthogonal to the column space of $A$

- When $A_{m\times n}$ is factored in the form $A=BT$, the exact solution is
  $$
  Tx=(B^*B)^{-1}B^*b
  $$
  This can be verified that
  $$
  A^*Ax=(BT)^*BTx=T^*B^*B(B^*B)^{-1}B^*b=T^*B^*b=A^*b
  $$
  The matrix $(B^*B)^{-1}=diag\{d_1^{-1},\dots,d_n^{-1}\}$, the numbers $d_i$ being those computed in the modified Gram-Schimdt algorithm.

- Also can be solved through $A^*Ax=A^*b$, since $A^*A$ is Hermitian and positive definite, Cholesky factorization may be used.

The direct use of the normal equations for solving a least-squares problem seems very appealing because of its conceptual simplicity. However, it is
regarded as one of the least satisfactory methods to use on this problem. One reason is that the condition number of A∗A may be considerably worse than that of A. For example,
$$
A=
\left (\begin{array}{ccc}
1 & 1& 1\\
\epsilon & 0 & 0\\
0&\epsilon  & 0\\
0&0&\epsilon\\
\end{array}\right)
\ \ \ A^*A=
\left (\begin{array}{ccc}
1+\epsilon^2 & 1& 1\\
1&1+\epsilon^2 &  1\\
1&1&1+\epsilon^2\\
\end{array}\right)
$$
For small , in a computer one may have $rank(A) = 3$ but $rank(A^∗A) = 1$

#### Householder’s QR-Factorization

One of the most useful orthogonal factorizations is called QR-Factorization. The objective is to factor an $m × n$ matrix $A$ into a product
$$
A_{m×n} = Q_{m×m}R_{m×n}
$$
where $Q $ is a unitary matrix and $R$ is an $m × n$ upper triangular matrix. The factorization algorithm actually produces
$$
Q^*A_{m×n} = R_{m×n}
$$
$Q^∗$ is build up step-by-step as
$$
Q^∗ = U^{(n−1)}U^{(n−2)}\dots U^{(1)}
$$
where
$$
U_{m\times m}^{(k)}=\left(\begin{array}{cc}
l_{(k-1)\times(k-1)} & 0\\
0 & I_{(m-k+1)\times (m-k+1)}-vv^*
\end{array}\right)
$$
with $v \in \mathbb C^{m−k+1}$; $\|v\|_2 = \sqrt 2$. 

So $U^{(1)} = I_{m×m} − vv^∗$ and we want $U^{(1)}A_1 = β_1e^{(1)}$ with $e^{(1)} = (1, 0, \dots, 0)^T$; $|β_1| = \|A_1\|_2$. 

Next,$U^{(2)}U^{(1)}A_1 = β_1e^{(1)}, U^{(2)}U^{(1)}A_2 = (*,\beta_2,0,\dots,0)^T$

Finally, $U^{(n−1)}U^{(n−2)}\dots U^{(1)}A = R_{m×n}$.



### 5.4 Singular-Value Decomposition and Pseudo-inverses

#### Singular-Value Decomposition 

- Theorem (on Singular-Value Factorization)

  An arbitrary complex $m × n$ matrix $A$ can be factored as
  $$
  A_{m×n} = P_{m×m} D_{m×n} Q_{n×n}
  $$
  where $P$ is an $m × m$ unitary matrix, $D$ is an $m × n$ diagonal matrix, and $Q$ is an n × n unitary matrix.

- （半）正定阵特征值大于（等于）0

- How to choose $P,D,Q$?
  Let $\{u_1,u_2,\dots,u_n\}$ be an orthonormal set of eigenvectors for $A^*A$, $Q_{n\times n}=(u_1,u_2,\dots,u_n)^*$

  Then, define $v_i=\sigma_i^{-1}Au_i, (\sigma_i>0,1\leq i\leq r)$​. Here, $r$​ is the numebr of singular value of $A$​ which is not negative, add some additional vectors $v_i (r + 1 ≤ i ≤ n)$ so that $\{v_1, v_2,\dots v_m\}$ is an orthonormal base for $\mathbb C^m$. The unitary matrix $P_{n\times m}=(v_1,v_2,\dots,v_m)$,

   $D_{m\times n}=(d_{ij})_{m\times n}$ so $d_{ii}=\sigma_i(1\leq i\leq r)$ and $d_{ij}=0$​ elsewhere.

- $u_i^∗A^∗Au_i = u_i^∗σ_i^2u_i = σ_i^2$

- Finally, singular-value decomposition is not unique.



#### Pseudo-inverses

For an $m × n$ matrix of the form
$$
D_{m\times n}=\left[
\begin{array}{}
\sigma_1&&&&&&\\
&\sigma_2&&&&&\\
&&\ddots&&&&\\
&&&\sigma_r&&&\\
&&&&0&&\\
&&&&&\ddots&\\
&&&&&&0
\end{array}
\right]
$$
in which each $\sigma_i>0$

define its pseudoinverse to be the $n\times m$ matrix 
$$
D^+=\left[
\begin{array}{}
\sigma_1^{-1}&&&&&&\\
&\sigma_2^{-1}&&&&&\\
&&\ddots&&&&\\
&&&\sigma_r^{-1}&&&\\
&&&&0&&\\
&&&&&\ddots&\\
&&&&&&0
\end{array}
\right]
$$
For a general matrix $A$, let
$$
A=PDQ
$$
be its singular-value decomposition; then the pseudoinverse of $A$ is defined as
$$
A^+=Q^*D^+P^*
$$
The pseudoinverse of a matrix is unique determined, although the singular-value decomposition is not unique.

#### Inconsistent & Underdetermined(欠定) Systems

The principal applications the pseudoinverse is to systems of equations
that are inconsistent or have non-unique solutions. Consider a system
of equations
$$
Ax=b
$$
where $A$ is $m × n$, $x$ is $n × 1$, $b$ is $m × 1$. The minimal solution of this problem
is defined as:

1. If the system is consistent and has a unique solution, x, then the minimal
   solution is defined to be x.
2. If the system is consistent and has a set of solutions, then the minimal
   solution is the element of this set having  the least Euclidean norm.
3. If the system is inconsistent and has a unique least-squares solution, then
   the minimal solution is that x. 
4. If the system is inconsistent and has a set of least-squares solutions, then
   the minimal solution is the element of this set having the least Euclidean
   norm.

- Theorem (Pseudoinverse Minimal Solution)

- The minimal solution of $Ax = b$ is given by
  $$
  x=A^+b
  $$



#### Penrose Properties

Theorem (on Penrose Properties)

Corresponding to any matrix $A$, there exists at most one matrix $X$ having these four properties:

1. $AXA = A$
2. $XAX = X$
3. $(AX)^∗ = AX$
4. $(XA)^∗ = XA $

Theorem (on Unique Pseudoinverse)
The pseudoinverse of a matrix has the four Penrose properties. Hence, each matrix has a unique pseudoinverse

Theorem (on Singular-Value Decomposition Properties)
Let $A$ have the singular-value decomposition $A_{m×n} = P_{m×m} D_{m×n} Q_{n×n}$ as described in the proof of the Singular-Value Factorization (Decomposition) Theorem. Then

1. The rank of $A$ is $r$:
2. $\{u_{r+1}, u_{r+2},\dots, u_n\}$ is orthonormal base for the null space of $A$.
3. $\{v_1, v_2,\dots, v_r\}$ is orthonormal base for the range of $A$.
4. $\|A\|^2 = \max_{1≤i≤n} σ_i$. 



Theorem (Singular-Value Decomposition: Economical Version)
If $A$ is an $m × n$ matrix of rank $r (m ≥ n ≥ r)$; then $A$ can be factored as
$$
A_{m×n} = V_{m×r} S_{r×r} U_{r×n}
$$
in which $V$ is an $m × r$ matrix with orthonormal columns, $S$ is a nonsingular
$r × r$ diagonal matrix, and $U$ is an $r × n$ matrix with orthonormal rows



Theorem (on Orthonormal Bases, P295)
Let $L$ be a linear transformation from $\mathbb C^m$ to $\mathbb C^n$. Then there are orthonormal bases $\{u_1,u_2\dots,u_m,\}$ for $\mathbb C^m$ and $\{v_1,v_2\dots,v_n,\}$ for $\mathbb C^n$ s.t.
$$
Lu_i = \begin{cases}
σ_iv_i & \text{ if }1≤ \min(m,n)\\
0 & \text{ if } \min(m,n)<i\leq m
\end{cases}
$$







