# Getting Started With R.

## Introduction
 R is a programming language and software environment for statical computing and graphics supported by the R foundation. R is not like a general-purpose programming language like Java, C, because it was created by statisticians as an active environment. Interactivity is the critical characteristic that allows R to explore our data. It is also a programming language and development environment for statistical testing and graphical testing. Each statistical testing is either linear, non-linear modeling, classification or many more. Different types of the plot are required while doing data analysis. In order to run R, we will use IDE(according to Wikipedia an integrated development environment(IDE) is a software application that provides comprehensive facilities to the programmer for software development). The core component which is required for every R program is BaseR. These core components contain only the code importing bits that run our code successfully. 


## History About R 
Bell labs develops s language in 1976. In 1993 Ross Lhaka and Robert Gentleman created R in New-Zealand. R became a free source in 1995. R version 1.0.0 is released in 2000 to the public. IDE Rstdio is release in 2011.

## Drawback
* R is build by using `S`. If we want to build apps R probabily one be our choice.
* The object that we work must be strored in memory and working with fetch data set can queckly

## Installing and Setting up R in your Windows
### Step 1: Downloading installation file 
* Download R tools from [Official Website](https://cran.r-project.org/bin/windows/Rtools/)
* Next, we need to have an IDE, most popular one is Rstudio. We can download it from [this link](https://www.rstudio.com/products/rstudio/download/).

After downloading installation file, install them on desired places and then open the console.

After installation completed open R then we get window just like below

![img](https://github.com/iamdurga/iamdurga.github.io/raw/main/assets/Math_blog/windows%203.PNG)

Now we can write our R codes within console or we can do it via Rstudio. 

I prefer to use Jupyter Notebook for runing R because it is more friendly for me. A good tutorial is available at [Anaconda's Documentation](https://docs.anaconda.com/anaconda/navigator/tutorials/r-lang/).

## My First R program
I am assigning variable in R as my first R programs.

### Assigning Variable and operator in R
 A Variable is a container that stores values. An assignment statement set or reset the value store in the storage location(s) denoted by variable name(by Wikipedia). The assignment operator is a command that is it telling the computer to assign the text apple to the variable product. we can also assign by `assign('products', ' apple)`. We can assign the variable in R in many ways like below.

#### Way 1
```R
('apple'-> product)
```

#### Way 2
```R
(product = 'apple')
```

#### Way 3
```R
assign('products', ' apple)
```

## Logical Operators in R 
Logical operator means those which gives `True` and `False` value. For example
### Example 1
```R
apple <- 2
banana <- 3
most_expensive <- banana> apple
most_expensive
```

Output of above code is,

```R
TRUE
```

### Example 2
```R
apple <- 2
banana <- 3
most_expensive <- banana< apple
most_expensive
```

Output of above code is,

```R
FALSE
```

### Example 3
```R
apple <- 2
banana <- 2
most_expensive <- banana == apple
most_expensive
```

Output is,

```R
TRUE
```

### Example 4
```R
apple <- 2
banana <- 2
most_expensive <- banana != apple
most_expensive
```

Output is,

```R
FALSE
```

## Some Commonly Used Data Types in R
Data is centre for analysis if there is no data there is no analysis. Every piece of data are working with some characteristics thses characteristics can be summarize with data type.
* `Character` :
Anything inside quotation is a character.
* `Number`: 
Number in R is double. Working with whole and fraction is a unique feature of double. Another is integer. 
* `Integer` 
Integer is actually simplified version of double. It store data as a string we must use capital letter L. In our use we need to use double rather than integer.
* `Logical(Boolean)`:
`Yes` or `No`. Also `T` or `F`.
* `Complex Number`:
$$ 2 + 6i $$
* `Raw`:
It is not so popular data type. It is not easy to create variable of raw type. If we really need to create raw function as a result of calling this function we get raw type data.

All the fundamental data types are called atomic data type.

### Example of numbers
An integer:
```R
a <- 2L
class(a)
```
Output is,
```R
'integer'
```
A numeric:
```R
a <- 2
class(a)
```
Output is,
```R
'numeric'
```

```R
quantity <- 2
typeof(quantity)
```
Output is,
```R
'double'
```

```R
quantity_integer <- 2L
typeof(quantity_integer)
```
Output is,
```R
'integer'
```

## Comments
Comments are used to give important information about the code. Comments are not run by the program but a programmer writes it for better explanation of the code.
```R
# This is a comment in R
```

## Exploring vectors and factors
Data structure as name suggest represent way to organize data to facilate different operations to perform faster calculations.

* `Vectors`:
Collection of data of same structure.
* `Factors`:
Which are used to store categorical data.
* `Array`:
Is a matrix which are generalization of vectors.
* `List\DataFrame`:
Elements of different list are dataframe. List are more complex data structure because they allow us to store other list too. We can think data frame as spreadsheets where  data are organize as columns and rows where each column has specific data type. Within a data frame we have all kinds of datatype but within one column we have only one data type. Other criteria to categorize our data is by dimensional. 

Vector and list are one dimensional objects. Matrices and dataframe are two dimensional data structure. Array are the object that have more than two dimensions.

> Vector have two properties they are one dimensional and containing element of same type.

### Assigning a column vector
Lets assign a column vector,
```R
assign('b',c(1,2,3,4))
print(b)
```

Output is,
```R
1 2 3 4
```

### Vectors attributes:
* `length`: 
It is denoted by length(a) and its meaning is number of elements.
* `Name`: 
 names(a), it allows us to add element in the list.
* `Type`: 
typeof(a), It gives type of data.

There are six vectors types
* Double
* logical
* character
* complex
* Raw
* Integer

```R
vector <- c("Durga","Puja","Ram","Hari")
vector
length(vector) # length 
names(vector)= "Sita" #names
typeof(vector) # type
vector
```
Output is,
```R
'Durga''Puja''Ram''Hari'
4
'character'
Sita'Durga'2'Puja'3'Ram'4'Hari'
```
### Manipulating vectors.
Manipulating of vectors consists of sorting, ordering, indexing.
* `sorting`: Sort the data in some order.
* `Ordering`:
The order function return the index needed to get the vector sort.
* `Indexing`:
Selecting specifics iteam by position.

```R
quantity <- c(1,3,2,5,6,7)
sort(quantity)
order(quantity)
```
Output is,
```R
1 2 3 5 6 7
1 3 2 4 5 6
```
```R
a <- c(1,7,36,0,7,5)
a[2]
a[3:5]
a[c(2,4)]
a[c(4,7)]# it return particular element from vector
a[-2]
a[-(2:4)] # it skip the element in the vector.
a[a==1]
a[a>3]
a[a %in%c(2,4)] # it gives matching element.
```
Output is,
```R
7
36 0 7
7 0
0 <NA>
1 36 0 7 5
1 7 5
1
7 36 7 5
```
### Operating vector
Adding or multipling vector of different size is called recycling rule. For recycling largest vector must be multiple of small one.

```R
c <- 1:6
d <- 1:3
c * d
```
Output is,
```R
1 4 9 4 10 18
```


## Sequence generation
It is used to create sequence of elements in a vector. `seq()` function takes length and difference between values as optional argument. In a code below, I take elements in the range 1 to 5 in the interval of 1.5.

Example:
```R
seq(1,5,by = 1.5)
```
Output is,
```R
1 2.5 4
```

## Replicating elements
It is used to return the replicating element in the list in a specified times. In the following code I replicate the numbers from 1 to 6 two times. A builtin function `rep()` is used.

Example:
```R
e<- rep(1:6,times = 2)
e
```
Output is,
```R
1 2 3 4 5 6 1 2 3 4 5 6
```
We can replicate the same number at desirable times.
```R
x <- rep(c(1),each = 10)
x
```
Out put is,
```R
1 1 1 1 1 1 1 1 1 1
```

## Scan Function
Scan function read any file into vector. It is very powerful function. In the code given below, it scan function read `covid_data.csv`.

```R
f <- scan("covid data.csv", what = "Character")
f
```
Out put of the above code is,
```R
'date,totalCases,newCases,totalRecoveries,newRecoveries,totalDeaths,newDeaths' '1/23/2020,1,1,0,0,0,0' '1/24/2020,0,0,0,0,0,0' '1/25/2020,0,0,0,0,0,0' '1/26/2020,0,0,0,0,0,0' '1/27/2020,0,0,0,0,0,0' '1/28/2020,0,0,0,0,0,0' '1/29/2020,0,0,0,0,0,0' '1/30/2020,0,0,0,0,0,0' '1/31/2020,0,0,1,1,0,0' '2/1/2020,0,0,1,0,0,0' '2/2/2020,0,0,1,0,0,0' '2/3/2020,0,0,1,0,0,0' '2/4/2020,0,0,1,0,0,0' '2/5/2020,0,0,1,0,0,0' '2/6/2020,0,0,1,0,0,0' '2/7/2020,0,0,1,0,0,0' '2/8/2020,0,0,1,0,0,0' '2/9/2020,0,0,1,0,0,0' '2/10/2020,0,0,1,0,0,0' '2/11/2020,0,0,1,0,0,0' '2/12/2020,0,0,1,0,0,0' '2/13/2020,0,0,1,0,0,0' '2/14/2020,0,0,1,0,0,0' '2/15/2020,0,0,1,0,0,0' '2/16/2020,0,0,1,0,0,0' '2/17/2020,0,0,1,0,0,0' '2/18/2020,0,0,1,0,0,0' '2/19/2020,0,0,1,0,0,0' '2/20/2020,0,0,2,1,0,0' '2/21/2020,0,0,2,0,0,0' '2/22/2020,0,0,2,0,0,0' '2/23/2020,0,0,2,0,0,0' '2/24/2020,0,0,2,0,0,0' '2/25/2020,0,0,2,0,0,0' '2/26/2020,0,0,2,0,0,0' '2/27/2020,0,0,2,0,0,0' '2/28/2020,0,0,2,0,0,0' '2/29/2020,0,0,2,0,0,0' '3/1/2020,0,0,2,0,0,0' '3/2/2020,0,0,2,0,0,0' '3/3/2020,0,0,2,0,0,0' '3/4/2020,0,0,2,0,0,0' '3/5/2020,0,0,2,0,0,0' '3/6/2020,0,0,2,0,0,0' '3/7/2020,0,0,2,0,0,0' '3/8/2020,0,0,2,0,0,0' '3/9/2020,0,0,2,0,0,0' '3/10/2020,0,0,2,0,0,0' '3/11/2020,0,0,2,0,0,0' '3/12/2020,0,0,2,0,0,0' '3/13/2020,0,0,2,0,0,0' '3/14/2020,0,0,2,0,0,0' '3/15/2020,0,0,2,0,0,0' '3/16/2020,0,0,2,0,0,0' '3/17/2020,0,0,2,0,0,0' '3/18/2020,0,0,2,0,0,0' '3/19/2020,0,0,2,0,0,0' '3/20/2020,0,0,2,0,0,0' '3/21/2020,0,0,2,0,0,0' '3/22/2020,0,0,2,0,0,0' '3/23/2020,1,1,2,0,0,0' '3/24/2020,1,0,2,0,0,0' '3/25/2020,2,1,2,0,0,0' '3/26/2020,2,0,2,0,0,0' '3/27/2020,3,1,2,0,0,0' '3/28/2020,4,1,2,0,0,0' '3/29/2020,4,0,2,0,0,0' '3/30/2020,4,0,2,0,0,0' '3/31/2020,4,0,2,0,0,0' '4/1/2020,4,0,2,0,0,0' '4/2/2020,5,1,2,0,0,0' '4/3/2020,5,0,2,0,0,0' '4/4/2020,8,3,2,0,0,0' '4/5/2020,8,0,2,0,0,0' '4/6/2020,8,0,2,0,0,0' '4/7/2020,8,0,2,0,0,0' '4/8/2020,8,0,2,0,0,0' '4/9/2020,8,0,2,0,0,0' '4/10/2020,8,0,2,0,0,0' '4/11/2020,8,0,2,0,0,0' '4/12/2020,11,3,2,0,0,0' '4/13/2020,13,2,2,0,0,0' '4/14/2020,15,2,2,0,0,0' '4/15/2020,15,0,2,0,0,0' '4/16/2020,15,0,2,0,0,0' '4/17/2020,29,14,2,0,0,0' '4/18/2020,30,1,4,2,0,0' '4/19/2020,30,0,5,1,0,0' '4/20/2020,30,0,5,0,0,0' '4/21/2020,41,11,6,1,0,0' '4/22/2020,44,3,8,2,0,0' '4/23/2020,47,3,9,1,0,0' '4/24/2020,48,1,11,2,0,0' '4/25/2020,48,0,12,1,0,0' '4/26/2020,51,3,14,2,0,0' '4/27/2020,51,0,14,0,0,0' '4/28/2020,53,2,14,0,0,0' '4/29/2020,56,3,14,0,0,0' '4/30/2020,56,0,14,0,0,0' '5/1/2020,58,2,14,0,0,0' '5/2/2020,58,0,14,0,0,0' '5/3/2020,74,16,14,0,0,0' '5/4/2020,74,0,14,0,0,0' '5/5/2020,81,7,14,0,0,0' '5/6/2020,98,17,20,6,0,0' '5/7/2020,100,2,20,0,0,0' '5/8/2020,101,1,28,8,0,0' '5/9/2020,108,7,29,1,0,0' '5/10/2020,109,1,29,0,0,0' '5/11/2020,133,24,31,2,0,0' '5/12/2020,216,83,31,0,0,0' '5/13/2020,242,26,33,2,0,0' '5/14/2020,248,6,33,0,1,1' '5/15/2020,266,18,34,1,1,0' '5/16/2020,280,14,34,0,1,0' '5/17/2020,294,14,34,0,3,2' '5/18/2020,374,80,34,0,3,0' '5/19/2020,401,27,35,1,3,0' '5/20/2020,426,25,43,8,3,0' '5/21/2020,456,30,47,4,4,1' '5/22/2020,515,59,68,21,4,0' '5/23/2020,583,68,68,0,5,1' '5/24/2020,602,19,85,17,5,0' '5/25/2020,681,79,110,25,5,0' '5/26/2020,771,90,152,42,5,0' '5/27/2020,885,114,180,28,6,1' '5/28/2020,1041,156,184,4,6,0' '5/29/2020,1211,170,184,0,6,0' '5/30/2020,1400,189,188,4,7,1' '5/31/2020,1571,171,189,1,8,1' '6/1/2020,1810,239,190,1,8,0' '6/2/2020,2098,288,235,45,9,1' '6/3/2020,2299,201,238,3,11,2' '6/4/2020,2633,334,256,18,12,1' '6/5/2020,2911,278,289,33,12,0' '6/6/2020,3234,323,295,6,13,1' '6/7/2020,3447,213,340,45,13,0' '6/8/2020,3760,313,363,23,14,1' '6/9/2020,4083,323,394,31,15,1' '6/10/2020,4362,279,394,0,17,2' '6/11/2020,4612,250,394,0,17,0' '6/12/2020,5059,447,394,0,18,1' '6/13/2020,5334,275,394,0,19,1' '6/14/2020,5759,425,394,0,19,0' '6/15/2020,6210,451,1044,650,20,1' '6/16/2020,6590,380,1161,117,20,0' '6/17/2020,7176,586,1170,9,22,2' '6/18/2020,7847,671,1189,19,22,0' '6/19/2020,8273,426,1405,216,23,1' '6/20/2020,8604,331,1581,176,23,0' '6/21/2020,9025,421,1775,194,23,0' '6/22/2020,9558,533,2151,376,24,1' '6/23/2020,10098,540,2225,74,24,0' '6/24/2020,10727,629,2339,114,25,1' '6/25/2020,11161,434,2651,312,27,2' '6/26/2020,11754,593,2699,48,27,0' '6/27/2020,12308,554,2835,136,29,2' '6/28/2020,12771,463,3014,179,30,1' '6/29/2020,13247,476,3135,121,30,0' '6/30/2020,13563,316,3195,60,30,0' '7/1/2020,14045,482,4657,1462,33,3' '7/2/2020,14518,473,5321,664,33,0' '7/3/2020,15258,740,6144,823,33,0' '7/4/2020,15490,232,6416,272,34,1' '7/5/2020,15783,293,6548,132,35,1' '7/6/2020,15963,180,6812,264,35,0' '7/7/2020,16167,204,7500,688,36,1' '7/8/2020,16422,255,7753,253,36,0' '7/9/2020,16530,108,7892,139,38,2' '7/10/2020,16648,118,8012,120,39,1' '7/11/2020,16718,70,8443,431,39,0' '7/12/2020,16800,82,8590,147,39,0' '7/13/2020,16944,144,10295,1705,39,0' '7/14/2020,17060,116,10329,34,39,0' '7/15/2020,17176,116,11026,697,40,1' '7/16/2020,17343,167,11250,224,41,1' '7/17/2020,17444,101,11388,138,41,0' '7/18/2020,17501,57,11491,103,41,0' '7/19/2020,17657,156,11549,58,41,0' '7/20/2020,17843,186,11722,173,41,0' '7/21/2020,17993,150,12331,609,42,1' '7/22/2020,18093,100,12538,207,44,2' '7/23/2020,18240,147,12694,156,44,0' '7/24/2020,18373,133,12801,107,47,3' '7/25/2020,18482,109,12907,106,47,0' '7/26/2020,18612,130,12982,75,49,2' '7/27/2020,18751,139,13608,626,50,1' '7/28/2020,19062,311,13729,121,50,0' '7/29/2020,19272,210,13875,146,53,3' '7/30/2020,19546,274,14102,227,56,3' '7/31/2020,19770,224,14253,151,57,1' '8/1/2020,20085,315,14346,93,59,2' '8/2/2020,20331,246,14457,111,59,0' '8/3/2020,20749,418,14815,358,61,2' '8/4/2020,21008,259,14880,65,62,1' '8/5/2020,21389,381,15010,130,67,5' '8/6/2020,21749,360,15243,233,71,4' '8/7/2020,22213,464,15668,425,74,3' '8/8/2020,22591,378,16167,499,76,2' '8/9/2020,22971,380,16207,40,80,4' '8/10/2020,23309,338,16347,140,83,3' '8/11/2020,23947,638,16518,171,86,3' '8/12/2020,24431,484,16582,64,95,9' '8/13/2020,24956,525,16691,109,96,1' '8/14/2020,25550,594,16931,240,101,5' '8/15/2020,26018,468,17055,124,102,1' '8/16/2020,26659,641,17189,134,104,2' '8/17/2020,27240,581,17349,160,107,3' '8/18/2020,28256,1016,17434,85,114,7' '8/19/2020,28937,681,17554,120,120,6' '8/20/2020,29644,707,17818,264,126,6' '8/21/2020,30482,838,18068,250,137,11' '8/22/2020,31116,634,18204,136,146,9' '8/23/2020,31934,818,18485,281,149,3' '8/24/2020,32677,743,18660,175,157,8' '8/25/2020,33532,855,18973,313,164,7' '8/26/2020,34417,885,19358,385,175,11' '8/27/2020,35528,1111,19927,569,183,8' '8/28/2020,36455,927,20096,169,195,12' '8/29/2020,37339,884,20409,313,207,12' '8/30/2020,38560,1221,20676,267,221,14' '8/31/2020,39459,899,21264,588,228,7' '9/1/2020,40528,1069,22032,768,239,11' '9/2/2020,41648,1120,23144,1112,250,11' '9/3/2020,42876,1228,24061,917,257,7' '9/4/2020,44235,1359,25415,1354,271,14' '9/5/2020,45276,1041,26981,1566,280,9' '9/6/2020,46256,980,28795,1814,289,9' '9/7/2020,47235,979,30531,1736,300,11' '9/8/2020,48137,902,32818,2287,306,6' '9/9/2020,49218,1081,33736,918,312,6' '9/10/2020,50464,1246,35554,1818,317,5' '9/11/2020,51918,1454,36526,972,322,5' '9/12/2020,53119,1201,37378,852,336,14' '9/13/2020,54158,1039,38551,1173,345,9' '9/14/2020,55328,1170,39430,879,360,15' '9/15/2020,56787,1459,40492,1062,371,11' '9/16/2020,58326,1539,41560,1068,379,8' '9/17/2020,59572,1246,42803,1243,383,4' '9/18/2020,61592,2020,43674,871,390,7' '9/19/2020,62796,1204,45121,1447,401,11' '9/20/2020,64121,1325,46087,966,411,10' '9/21/2020,65275,1154,47092,1005,427,16' '9/22/2020,66631,1356,47915,823,429,2' '9/23/2020,67803,1172,49808,1893,436,7' '9/24/2020,69300,1497,50265,457,452,16' '9/25/2020,70613,1313,51720,1455,458,6' '9/26/2020,71820,1207,52867,1147,466,8' '9/27/2020,73393,1573,53752,885,476,10' '9/28/2020,74744,1351,54494,742,481,5' '9/29/2020,76257,1513,55225,731,491,10' '9/30/2020,77816,1559,56282,1057,498,7'
```

## Conversion of different type of data into character type is called implicit coercian
R convert coerced data type into character.

```R
x <- c(1,'two',4,"durga")
x
typeof(x)
```
Output is,
```R
'1' 'two' '4' 'durga'
'character'
```
## Explicit type coercian
* We do this by typing `as.desire data type`. Explicit type coercian helps us to deal with incorectly catagorized data.
* We can not transfer numeric into character 
* Character into numberic.
```R
num <- 1:5
num_char <- as.character(num)
num_char
```
Output is,
```R
'1' '2' '3' '4' '5'
```
```R
product <- c("apple",1,"banana")
as.numeric(product)
```
Output is,

```R
Warning message in eval(expr, envir, enclos):
"NAs introduced by coercion"
<NA> 1 <NA>

```

## Installing Packages in R
There are numerous useful packages to do various tasks in R and with those packages, we could do things better and faster way. Once simpler way to install packages is via console;

```R
install.packages("haven")
```

```R
library("haven") # allows to read sav file
saq8 <- read_sav("F:/Statisticts with R/CSV file for covid data/SAQ8.sav")
```

In above example, I first installed package named as `haven` and then I used it to read `sav` file.

This all for this blog and I hope you enjoyed it. Please leave the feedbacks and stay tuned for my next blog.