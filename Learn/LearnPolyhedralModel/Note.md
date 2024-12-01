
[cs6120-Advanced Compilers](https://www.cs.cornell.edu/courses/cs6120/2023fa/)
[text](https://www.cs.cornell.edu/courses/cs6120/2023fa/blog/polyhedral/#1)
# Polyhedral Model

[text](https://en.wikipedia.org/wiki/Polytope_model)



The polyhedral model (also called the polytope method) is a mathematical framework for programs that perform large numbers of operations -- too large to be explicitly enumerated -- thereby requiring a compact representation. Nested loop programs are the typical, but not the only example, and the most common use of the model is for loop nest optimization in program optimization. The polyhedral method treats each loop iteration within nested loops as lattice points inside mathematical objects called polyhedra, performs affine transformations or more general non-affine transformations such as tiling on the polytopes, and then converts the transformed polytopes into equivalent, but optimized (depending on targeted optimization goal), loop nests through polyhedra scanning.

多面体模型（也称为多面体方法）是一个数学框架，用于执行大量运算的程序 -- 太大而无法显式列举 -- 因此需要紧凑的表示。嵌套循环程序是典型的但不是唯一的例子，该模型最常见的用途是程序优化中的循环嵌套优化。多面体方法将嵌套循环中的每个循环迭代视为称为多面体的数学对象内的格子点，执行仿射变换或更通用的非仿射变换，例如在多面体上平铺，然后通过多面体扫描将转换后的多面体转换为等效但优化（取决于目标优化目标）的循环嵌套。

![alt text](image.png)

# Polyhedral model in programming

[Frameworks supporting the polyhedral model](https://en.wikipedia.org/wiki/Frameworks_supporting_the_polyhedral_model)


# Affine transformation

[text](https://en.wikipedia.org/wiki/Affine_transformation)

[text](https://www.mathworks.com/discovery/affine-transformation.html)

[text](https://blog.csdn.net/u011681952/article/details/98942207)

In Euclidean geometry, an affine transformation or affinity (from the Latin, affinis, "connected with") is a geometric transformation that preserves lines and parallelism, but not necessarily Euclidean distances and angles.

在欧几里得几何中，仿射变换或亲和力（来自拉丁语 affinis，“连接”）是一种几何变换，它保留线条和平行度，但不一定是欧几里得距离和角度。

More generally, an affine transformation is an automorphism of an affine space (Euclidean spaces are specific affine spaces), that is, a function which maps an affine space onto itself while preserving both the dimension of any affine subspaces (meaning that it sends points to points, lines to lines, planes to planes, and so on) and the ratios of the lengths of parallel line segments. Consequently, sets of parallel affine subspaces remain parallel after an affine transformation. An affine transformation does not necessarily preserve angles between lines or distances between points, though it does preserve ratios of distances between points lying on a straight line.

更一般地说，仿射变换是仿射空间的自同态（欧几里得空间是特定的仿射空间），即一个函数，它将仿射空间映射到自身，同时保留任何仿射子空间的维度（意味着它将点发送到点、线到线、平面到平面等）和平行长度的比率线段。因此，并行仿射子空间集在仿射变换后保持平行。仿射变换不一定保留线条之间的角度或点之间的距离，尽管它确实保留了位于直线上的点之间的距离比率。

If X is the point set of an affine space, then every affine transformation on X can be represented as the composition of a linear transformation on X and a translation of X. Unlike a purely linear transformation, an affine transformation need not preserve the origin of the affine space. Thus, every linear transformation is affine, but not every affine transformation is linear.

如果 X 是仿射空间的点集，则 X 上的每个仿射变换都可以表示为 X 上的线性变换和 X 的平移的组合。与纯线性变换不同，仿射变换不需要保留仿射空间的原点。因此，每个线性变换都是仿射变换，但并非每个仿射变换都是线性变换。

Examples of affine transformations include translation, scaling, homothety, similarity, reflection, rotation, hyperbolic rotation, shear mapping, and compositions of them in any combination and sequence.

仿射变换的示例包括平移、缩放、同质性、相似性、反射、旋转、双曲旋转、剪切映射以及它们以任意组合和顺序的组合。

Viewing an affine space as the complement of a hyperplane at infinity of a projective space, the affine transformations are the projective transformations of that projective space that leave the hyperplane at infinity invariant, restricted to the complement of that hyperplane.

将仿射空间视为投影空间无穷远处的超平面的补码，仿射变换是该射影空间的投影变换，它使无穷远处的超平面保持不变，仅限于该超平面的补码。

A generalization of an affine transformation is an affine map[1] (or affine homomorphism or affine mapping) between two (potentially different) affine spaces over the same field k. Let (X, V, k) and (Z, W, k) be two affine spaces with X and Z the point sets and V and W the respective associated vector spaces over the field k. A map f: X → Z is an affine map if there exists a linear map mf : V → W such that mf (x − y) = f (x) − f (y) for all x, y in X.[2]

仿射变换的泛化是同一域k 上两个（可能不同的）仿射空间之间的仿射映射1（或仿射同态或仿射映射）。设 （X， V， k） 和 （Z， W， k） 是两个仿射空间，其中 X 和 Z 是点集，V 和 W 是场 k 上各自的关联向量空间。如果存在线性映射mf ： V → W，使得 xf （x − y） = f （x） − f （y） 对于 X.2 中的所有 x、y → Z 是仿射映射