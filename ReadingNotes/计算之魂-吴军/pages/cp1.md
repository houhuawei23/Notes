# 第一章 毫厘千里之差———大O概念

# 1.1 算法的规范化和量化度量

计算机算法基础分析鼻祖：高德纳

编程的自我要求：

>力争一次全对，没有错误，算法在设计时就达到最佳。  
遇到问题时解决问题的积极态度。

### 思考题1.1：
*什么产品和计算机类似，是软硬件分离的？*

解答：乐器，汽车...

# 1.2 大数和数量级的概念

>我们对大数没有概念——>对计算机资源没有概念：

    在自己使用计算机时，绝大部分时候，都感觉速度超级快，内存用不完，根本不会在意计算资源的耗费。

    要逐步的培养起对运算量、运算速度、运算空间的“感觉”来。这要求对算法做空间和时间的复杂度分析，根据限制条件，计算量进行预估。

>高德纳算法分析思想：  
1.只考虑数据量大到接近无穷的情况。  
2.将决定算法快慢的因素划分为 不随数据量变化的因素 与 随数据量变化的因素。

### 思考题1.2：

*如果一个程序只运行一次，在编写它的时候，你是采用最直观但是效率较低的算法，还是依然寻找复杂度最优的算法？*

    寻找最优的算法，因为此算法可能在其他地方复用。

# 1.3 怎样寻找最好的算法

### 例题1.3 总和最大区间问题

给定一个实数序列，设计一个最有效的算法，找到一个总和最大的区间

输入格式：

第一行：一个正整数N，表示序列长度  
第二行：N个实数，以空格为分隔

#### 1. 三重循环 O(N^3)

* 确定一个子区间需要起点p和终点q，再由p到q遍历求和，共三重循环。

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<double> arr;
int N;
double tmp;
int p, q, mp, mq; //记录左右下标
double maxSum, sum; //记录当前最大子序和与子序和

int main(){
	cin>>N;
	for(int i=0; i<N; i++){
		cin>>tmp;
		arr.push_back(tmp);
	}
	for(p=0; p<N; p++){
		for(q=p; q<N; q++){
			sum = 0;
			for(int i=p; i<=q; i++){
				sum+=arr[i];
				if(sum>maxSum){
					maxSum=sum;
					mp=p;
					mq=q;
				}
			}
		}
	}
	printf("maxSum:%.2f\nmp:%d\nmq:%d\n", maxSum, mp, mq);
	for(int i=mp; i<=mq; i++){
		printf("%.2f ", arr[i]);
	}
	
	return 0;
}
```

#### 2. 两重循环 O(N^2)

* 子序列和没必要每次从头开始遍历累加，只需要在前一个子序列和之上加减即可。
* 在三重循环的代码上稍作修改

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<double> arr;
int N;
double tmp;
int p, q, mp, mq;
double maxSum, sum;

int main(){
	cin>>N;
	for(int i=0; i<N; i++){
		cin>>tmp;
		arr.push_back(tmp);
	}
	maxSum=arr[0];
	
	for(p=0; p<N; p++){
		sum=0;
		for(q=p; q<N; q++){
			sum+=arr[q];
			if(sum>maxSum){
				maxSum=sum;
				mp=p;
				mq=q;
			}
		}
	}
	
	printf("maxSum:%.2f\nmp:%d\nmq:%d\n", maxSum, mp, mq);
	for(int i=mp; i<=mq; i++){
		printf("%.2f ", arr[i]);
	}
	
	return 0;
}
```
#### 3. 分治法 O(NlogN)

* 序列S的最大和子序列分如下三种情况：
* 1. 完全位于左半区间 [p1, q1]
* 2. 完全位于右半区间 [p2, q2]
* 3. 横跨中间 [p1, q2]

1和2可递归调用，3经简单分析可知，其区间为 [p1, q2]


<font color="#dd0000">第三种情况有问题，书上称：</font><br /> 

>2．前后两个子序列的总和最大区间中间有间隔，我们假定这两个子序列的总和最大区间分别是[p1,q1]和[p2,q2]。这时，整个序列的总和最大区间是下面三者中最大的那一个：  
（1）[p1,q1]  
（2）[p2,q2]；  
（3）[p1,q2]。  
至于为什么，这是本节的思考题。  
><p align="right">Page: 037</p>

但是，横跨中间的可能的最大子区间，不一定为[p1,q2]。举例来说明：
序列：
>[-2, 1, -3, 4, -1, 2, 1, -5, 4]  

mid = (0+8)/2 = 4  
划分为:

>左半子序列：[-2, 1, -3, 4, -1]  
其最大和子序列为：[4]，对应下标区间为：[3,3]， 即p1=q1=3，lmsum = 4

>右半子序列：[2, 1, -5, 4]，  
其最大和子序列为：[4]，其对应下标区间为：[8,8]， 即p2=q2=8， rmsum = 4

>进而，[p1,q2]对应的区间为[3,8]，对应子序列和为5

然而，可知实际最大和子序列为[4, -1, 2, 1]，对应下标区间为[3, 6]，对应子序和为6  
并不是[p1,q2]

**按书中描述的代码：**
```cpp
#include <bits/stdc++.h>
using namespace std;

void maxSubSum(vector<double> arr, int p, int q, int &mp, int &mq, double &maxSum);

vector<double> arr;
int N=0;
int mp=0, mq=0;
double tmp=0, maxSum=0;

int main(){
	cin>>N;
	for(int i=1; i<=N; i++){
		cin>>tmp;
		arr.push_back(tmp);
	}

	maxSubSum(arr, 0, N-1, mp, mq, maxSum);
	
	printf("maxSum:%.2f\nmp:%d\nmq:%d\n",maxSum, mp, mq);
	for(int i=mp; i<= mq; i++){
		printf("%.2f ", arr[i]);
	}
	return 0;
}

void maxSubSum(vector<double> arr, int p, int q, int &mp, int &mq, double &maxSum){
	//边界情况
	if(p>=q){
		mp=mq=p;
		maxSum=arr[p];
		return ;
	}
	
	int mid = p+(q-p)/2;
	//mid = (p+q)/2;
	int lmp, lmq, rmp, rmq;
	double lmsum, rmsum, midsum=0;
	double tmpsum=0;
	//递归情况
	maxSubSum(arr, p, mid, lmp, lmq, lmsum);
	maxSubSum(arr, mid+1, q, rmp, rmq, rmsum);
	
	/*
	midsum错误，原本想midsum应该起止于mp， mq，想当然以为要包含lmsum，rmsum了；
	应该按照从中间向两边扩展
	
	*/
	for(int i=lmp; i<= rmq; i++){
		midsum+=arr[i];
	}
	
	if(lmsum>rmsum && lmsum>midsum){
		mp=lmp;
		mq=lmq;
		maxSum=lmsum;
	}
	else if(rmsum >lmsum && rmsum>midsum){
		mp=rmp;
		mq=rmq;
		maxSum=rmsum;
	}	
	else{
		mp=lmp;
		mq=rmq;
		maxSum=midsum;
	}
	return ;
	
}
```
>输入：  
9  
-2 1 -3 4 -1 2 1 -5 4

>输出：  
maxSum:5.00  
mp:3  
mq:8  
4.00 -1.00 2.00 1.00 -5.00 4.00

>期望输出：  
maxSum:6.00  
mp:3  
mq:6  
4.00 -1.00 2.00 1.00 

<font color="#dd0000">修正：</font><br /> 
重新考虑跨越中间(mid)的的情况：从mid开始，向序列左右扩展，保留最大值及对应下标  

**对应代码为：**

```cpp
#include <bits/stdc++.h>
using namespace std;

void maxSubSum(vector<double> arr, int p, int q, int &mp, int &mq, double &maxSum);

vector<double> arr;
int N=0;
int mp=0, mq=0;
double tmp=0, maxSum=0;

int main(){
	cin>>N;
	for(int i=1; i<=N; i++){
		cin>>tmp;
		arr.push_back(tmp);
	}
	
	maxSubSum(arr, 0, N-1, mp, mq, maxSum);
	
	printf("maxSum:%.2f\nmp:%d\nmq:%d\n",maxSum, mp, mq);
	for(int i=mp; i<= mq; i++){
		printf("%.2f ", arr[i]);
	}
	return 0;
}

void maxSubSum(vector<double> arr, int p, int q, int &mp, int &mq, double &maxSum){
	//边界情况
	if(p>=q){
		mp=mq=p;
		maxSum=arr[p];
		return ;
	}
	
	int mid = p+(q-p)/2;
	//mid = (p+q)/2;
	int lmp, lmq, rmp, rmq;
	double lmsum, rmsum, midrsum=0, midlsum=0;
	double tmprsum=0, tmplsum=0;
	//递归情况
	maxSubSum(arr, p, mid, lmp, lmq, lmsum);
	maxSubSum(arr, mid+1, q, rmp, rmq, rmsum);
	
	/*
	midsum错误，原本想midsum应该起止于mp， mq，想当然以为要包含lmsum，rmsum了；
	应该按照从中间向两边扩展
	
	*/
	int midp=mid, midq=mid+1;
	
	for(int j=mid; j>=p; j--){
		tmplsum+=arr[j];
		if(tmplsum>midlsum){
			midlsum=tmplsum;
			midp=j;
		}
		
	}
	for(int i=mid+1; i<=q; i++){
		tmprsum+=arr[i];
		if(tmprsum>midrsum){
			midrsum=tmprsum;
			midq=i;
		}
	}
	double midsum=midlsum+midrsum;
	if(lmsum>rmsum && lmsum>midsum){
		mp=lmp;
		mq=lmq;
		maxSum=lmsum;
	}
	else if(rmsum >lmsum && rmsum>midsum){
		mp=rmp;
		mq=rmq;
		maxSum=rmsum;
	}	
	else{
		mp=midp;
		mq=midq;
		maxSum=midsum;
	}
	return ;
	
}
```
#### 4. 正反两遍扫描法 O(N)

类似于前缀和？

```cpp
#include <bits/stdc++.h>
using namespace std;

double maxSubSum(vector<double> arr, int l, int r);

vector<double> arr;
int N;
double tmp;
int main(){
	cin>>N;
	for(int i=1; i<=N; i++){
		cin>>tmp;
		arr.push_back(tmp);
	}
	maxSubSum(arr, 0, N-1);
	return 0;
}

double maxSubSum(vector<double> arr, int l, int r){
	int mp,mq, p, q;
	mp=mq=p=q=l;
	int i;
	
	double maxsum=0, foresum=0, maxforesum=0;
	for(p=l; p<=r; p++){
		if(arr[p]>0){//找到大于0的项
			//printf("p: %.2f\n",arr[p]);
			//sum=arr[p];
			foresum=0;
			maxforesum=0;
			for(i=p; i<=r; i++){
				foresum+=arr[i];
				
				if(foresum>maxforesum){
					maxforesum=foresum;
					q=i;
					//printf("maxforesum: %.2f\n",maxforesum);
				}
				
				if(foresum<0 || i==r){//foresum<0或着已经到序列尾部了
					
					if(maxforesum>maxsum){
						maxsum=maxforesum;
						mp=p;
						mq=q;
					}
					

					break;
				}
			}

			p=i;
		}
		
	}
	printf("maxSum:%.2f\nmp:%d\nmq:%d\n",maxsum, mp, mq);
	for(int i=mp; i<= mq; i++){
		printf("%.2f ", arr[i]);
	}
	return maxsum;
}
```


#### 5. 动态规划dp O(N)

将问题分解考察，首先，最大子序和对应的区间为[p, q]，直观上感觉到，下标可以作为某种“状态”的标识。  
状态定义：dp[i]时以下标i为结尾的最大子序列  
状态转移关系/方程：  

>当dp[i-1] > 0时，dp[i] = dp[i-1]+arr[i]  
当dp[i-1]<=0时，dp[i]= arr[i]

初始化：dp[0]=arr[0];

代码实现如下：  

```cpp

#include <bits/stdc++.h>
using namespace std;

template<class elemType>
elemType maxSubSum(vector<elemType> arr, int r, int l);

int N;

int main(){
	cin>>N;
	vector<double> arr;
	double tmp;
	for(int i=0; i<N; i++){
		cin>>tmp;
		arr.push_back(tmp);
	}
	double maxsum=maxSubSum(arr, 0, N-1);
	return 0;
}

template<class elemType>//使用模板来提供更好的可移植性
elemType maxSubSum(vector<elemType> arr, int r, int l){
	int size = arr.size();
	//下标为i处的dp，代表：以i为最后一个元素的最大子序列和
	//当dp[i-1] > 0时，dp[i] = dp[i-1]+arr[i]
	//当dp[i-1]<=0时，dp[i]= arr[i]
	//用maxdp维护所有“以i结尾的最大子序列和”中最大的那个，即整个序列的最大子序列和
	elemType maxdp=0, dp=0;
	
	for(int i=0; i<size; i++){
		if(dp>0){
			dp+=arr[i];
		}
		else{
			dp=arr[i];
		}
		if(dp>maxdp){
			maxdp=dp;
		}
	}
	
	//printf("maxdp: %.")
	cout<<"maxdp: "<<maxdp;
	return maxdp;
}

```

#### 6.  简单扫描 O(N)  

首先扫描到一个大于零的数，以它为起点，向后累加得到thisSum，  
每次累加得到的hisSum与maxSum比较并去留，  
因为maxSum对应的子序列不能以负数开头，所以，当thisSum<0时，  
重置累加起点和thisSum，重复上面的流程。

```cpp

#include <bits/stdc++.h>
using namespace std;

double maxSubsequenceSum(vector<double> a, int &start, int &end);

int N;
double maxsum;
int p, q;

int main(){
	cin>>N;
	vector<double> arr;
	double tmp;
	for(int i=0; i<N; i++){
		cin>>tmp;
		arr.push_back(tmp);
	}
	double maxsum=maxSubsequenceSum(arr, p, q);
	printf("maxSum:%.2f\nmp:%d\nmq:%d\n",maxsum, p, q);
	
	return 0;
}

//枚举分析改进
/*
首先扫描到一个大于零的数，以它为起点，向后累加得到thisSum，
每次累加得到的hisSum与maxSum比较并去留，
因为maxSum对应的子序列不能以负数开头，所以，当thisSum<0时，
重置累加起点和thisSum，重复上面的流程
*/
//有特殊情况，需要改进，遇到<0的序列不能全盘否定，要保留最优
double maxSubsequenceSum(vector<double> a, int &start, int &end){
	int maxSum=0, thisSum=0, starttmp=0;
	start=end=0;
	int size= a.size();
	for(int i=0; i<size; i++){
		thisSum+=a[i];
		if(thisSum<0){
			thisSum=maxSum=0;
			starttmp=i+1;//遇到thisSum<0时，只修改starttmp
		}
		else if(thisSum>maxSum){
			maxSum=thisSum;
			start=starttmp;//只有thisSum>maxSum时才更新start
			end=i;
		}
	}
	if(start==size){start=end=0;}
	return maxSum;
}

```

#### 思考题1.3

* Q1．将例题1.3的线性复杂度算法写成伪代码。

* Q2．在一个数组中寻找一个区间，使得区间内的数字之和等于某个事先给定的数字。

* Q3．在一个二维矩阵中，寻找一个矩形的区域，使其中的数字之和达到最大值。

**解答：**

Q2：  
参考“两数之和”问题求解的思路：  
```伪代码
procedure twoSum(list, target)
	map: list[i]->i  
	
	for e, i in list, listindex:
		if target - e not in map :
			map[e] = i;
		
		else if target - e in map :
			print(map[e], map[target-e])
			//找到了
```
解答：
step1: 建立字典/哈希表map：数组list的前缀和 -> 前缀和对应的下标q  
因为，若S(p,q)==target， 就有S(1, p-1) = S(1, q) - S(p,q) = S(1, q) - target  


step2: 遍历到q，得到S(1,q)，查询map中有无key=S(1, q) - target
		如果有，则返回p=map[key]+1  

其中，map的建立可与遍历同时进行


# 1.4 关于排序的讨论

#### 思考题1.4
* Q1．赛跑问题（GS）。  

* 假定有25名短跑选手比赛争夺前三名，赛场上有五条赛道，一次可以有五名选手同时比赛。比赛并不计时，只看相应的名次。假设选手的发挥是稳定的，也就是说如果约翰比张三跑得快，张三比凯利跑得快，约翰一定比凯利跑得快。最少需要几次比赛才能决出前三名？  

* Q2．区间排序。

* 如果有N个区间[l1,r1],[l2,r2],…,[lN,rN]，只要满足下面的条件我们就说这些区间是有序的：存在xi∈[li,ri]，其中i=1,2,…,N。

* 比如，[1, 4]、[2, 3]和[1.5, 2.5]是有序的，因为我们可以从这三个区间中选择1.1、2.1和2.2三个数。同时[2, 3]、[1, 4]和[1.5, 2.5]也是有序的，因为我们可以选择2.1、2.2和2.4。但是[1, 2]、[2.7, 3.5]和[1.5, 2.5]不是有序的。

* 对于任意一组区间，如何将它们进行排序？