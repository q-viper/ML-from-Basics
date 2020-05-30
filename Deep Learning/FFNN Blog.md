# 1. Writing a Multi Layer Feedforward Neural Network from Scratch
All hail to the pandemic Corona virus, without it, we would not have lockdown and i won't have been stuck on village without internet to write these codes. I am not using <i>gist</i> for codes, so don't panic if you find unfriendly text formats. Also i have written this blog on `Markdown` of `Jupyter Notebook` so the formats are bit different. But truth is, the class we will be building will be just like `keras`. Yes Keras! 

## 1.1 What i am covering on this blog?
* Honestly, a scary and another blog about writing a Neural Network from scratch but i am leaving all the complex mathematics(also giving links to them on last). 
* This blog will also act as <b>prerequisites concept for Convolutional Neural Network from scratch</b> which i will write on next blog.
* Doing MNIST classification using `softmax crossentropy` and `GD` optimizer.
* Saving and loading model.

For this code, i will be using:
* `numpy`
* `matplotlib` for plotting
* `pandas` for just summary
* `time` for viewing time

# 2. Warning
<b>I am not going to make next Keras or Tensorflow here.</b>

Most of these days, we have many ML frameworks with many choices. We have high level to low level frameworks. Recently PyTorch has earned huge popularity but for beginners, Keras is a best choice. But writing a ML code from scratch is always a challenging and complex for even intermediate programmers. The mathematics behind the cute ML frameworks are scary. But once we understood a prerequistiqes of ML, then starting to code from scratch is a good idea. 

# 3. Steps
* Create a FF layer class.
* Create a NN class which will do bind FF layer and also does training.

OOP is very awesome feature of python and using the object of FF layer class, we can access its attributes and methods anywhere at anytime. 

## 3.1 Creating a FF layer class.

### 3.1.1 As usual, importing necessary requirements.
`import numpy as np`
    
### 3.1.2 Next, create a class and initialize it with possible parameters.
<code>def __init__(self, input_shape=None, neurons=1, bias=None, weights=None, activation=None, is_bias = True):
        np.random.seed(100)
        self.input_shape = input_shape
        self.neurons = neurons
        self.isbias = is_bias
        self.name = ""
        self.w = weights
        self.b = bias
        if input_shape != None:
            self.output_shape = neurons
        if self.input_shape != None:
            self.weights = weights if weights != None else np.random.randn(self.input_shape, neurons)
            self.parameters = self.input_shape *  self.neurons + self.neurons if self.isbias else 0  
        if(is_bias):
            self.biases = bias if bias != None else np.random.randn(neurons)
        else:
            self.biases = 0  
        self.out = None
        self.input = None
        self.error = None
        self.delta = None
        activations = ["relu", "sigmoid", "tanh", "softmax"]
        self.delta_weights = 0
        self.delta_biases = 0
        self.pdelta_weights = 0
        self.pdelta_biases = 0        
        if activation not in activations and activation != None:
             raise ValueError(f"Activation function not recognised. Use one of {activations} instead.")
        else:
            self.activation = activation    </code>
    
* `input_shape` : It is for the number of input from previous layer's neurons.
* `neurons` : How many neurons on this layer?
* `activation`: What activation function to use?
* `bias`: A bias value if `is_bias` is `true`.
* `isbias` : Will we use bias?

#### 3.1.2.1 Inside `__init__`
* `self.name`: To store name of this layer.
* `self.weights`: A connection strength or weights from previous to this layer. Use from `np.random.randn(n_input, neurons)` if not given.
* `self.biases` : A bias value. On this layer.
* `self.out` : Output of this layer.
* `self.input` : Input to this layer. Is the input data for input layer, and is output of previous layer for all other.
* `self.error` : Error term of this layer.
* `self.delta_weights` : \begin{equation}\delta{w}\end{equation}
* `self.delta_biases` : \begin{equation}\delta{b}\end{equation}
* `self.pdelta_weights` : Previous self.delta_weights
* `self.pdelta_biases` : Previous self.delta_biases
* `activations` : A list of possible activation functions. If given activation function not recognised, raise an error.
* `self.activation` : A variable to store activation function of this layer.



### 3.1.3 Now prepare the activation functions. For begining, we will use only few. 
<code>def activation_fn(self, r):
    """
    A method of FFL which contains the operation and defination of given activation function.
    """
    if self.activation == None or self.activation == "linear":
        return r   
    if self.activation == 'tanh': #tanh
        return np.tanh(r)
    if self.activation == 'sigmoid':  # sigmoid
        return 1 / (1 + np.exp(-r))
    if self.activation == "softmax":# stable softmax   
        r = r - np.max(r)
        s = np.exp(r)
        return s / np.sum(s)</code>
 

#### Recall the mathematics, 

\begin{equation}
i. tanh(soma) = \frac{1-soma}{1+soma}
\end{equation}

\begin{equation}
ii. linear(soma) = soma
\end{equation}

\begin{equation}
iii. sigmoid(soma) = \frac{1}{1 + exp^{(-soma)}}
\end{equation}

\begin{equation}
iv. relu(soma) = \max(0, soma)
\end{equation}

\begin{equation}
v. softmax(x_j) = \frac{exp^{(x_j)}}{\sum_{i=1}^n{exp^{(x_i)}}}
\end{equation}

\begin{equation}
Where, soma = XW + \theta
\end{equation}

And `W` is weight vector of shape `(n, w)`. `X` is input vector of shape `(m, n)` and `𝜃` is bias term of shape `w, 1`. 



<code>
def activation_dfn(self, r):
        """
            A method of FFL to find derivative of given activation function.
        """
        if self.activation is None:
            return np.ones(r.shape)
        if self.activation == 'tanh':
            return 1 - r ** 2
        if self.activation == 'sigmoid':
            return r * (1 - r)
        if self.activtion == 'softmax':
            soft = self.activation_fn(r)
            return soft * (1 - soft)
</code>

Lets revise bit of calculus. 

#### 3.1.3.2 Why do we need derivative? 
Well, if you are here then you already know that gradient descent is based upon the derivatives(gradients) of activation functions and errors. So we need to perform this derivative. But you are on your own to perform calculation. I will also explain the gradient descent later. 

\begin{equation}
i. \frac{d(linear(x))}{d(x)} = 1
\end{equation}

\begin{equation}
ii. \frac{d(sigmoid(x))}{d(x)} = sigmoid(x)(1- sigmoid(x))
\end{equation}

\begin{equation}
iii. \frac{d(tanh(x))}{d(x)} = \frac{2x}{(1+x)^2} 
\end{equation}

\begin{equation}
iv. \frac{d(relu(x))}{d(x)} = 1
\end{equation}

\begin{equation}
v. \frac{d(softmax(x_j))}{d(x_k)} = softmax(x_j)(1- softmax(x_j)) \space when \space j = k \space else\
\space -softmax({x_j}).softmax({x_k})
\end{equation}

For the sake of simplicity, we use the case of `j = k` for softmax.

### 3.1.4 Next create a method to perfom activation.
<code>
    def apply_activation(self, x):
        soma = np.dot(x, self.weights) + self.biases
        self.out = self.activation_fn(soma)
</code>

This method takes the input vector x and performs the linear combination and then applies activation function to this value. The soma term is the total input to this node.

### 3.1.5 Next create a method to set the new weight vector. 
This method is called when this layer is as hidden. If a layer is hidden, we won't give input shape but only the neurons on this layer. So we must set the `n_input`manually and same as weights. This method is used when we will be stacking the layers to make a <b>sequential</b> model.
<code>
    def set_n_input(self):
        self.weights = self.w if self.w != None else np.random.normal(size=(self.n_input, self.neurons))
</code>

I think we have made a simple Feedforward layer. Now is the time for us to create a class which can stack these layers together and also perform operations like train.

### 3.1.6 Next create a method to get total parameters of this layer:
<code>
    def get_parameters(self):
        self.parameters = self.input_shape *  self.neurons + self.neurons if self.isbias else 0  
        return self.parameters
</code>
Total parameters of a layer is total number of weights plus total biases.

### 3.1.7 Now create a method which will call above `get_parameters` and `set_n_input` also do additional task and 
<code>
    def set_output_shape(self):
        self.set_n_input()
        self.output_shape = self.neurons
        self.get_parameters()
</code>

This method will be called from the stackking class. And i have made this method to be identical to the `CNN` layers.

### 3.1.8 Finally, last but not least, a backpropagation method of this layer. 
<b> Note that every layer have different way of passing error backwards. I have done CNN from scratch hence i am making this article to support that one also.</b>
<code>
    def backpropagate(self, nx_layer):
        self.error = np.dot(nx_layer.weights, nx_layer.delta)
        self.delta = self.error * self.activation_dfn(self.out)
        self.delta_weights += self.delta * np.atleast_2d(self.input).T
        self.delta_biases += self.delta
 </code>
 Here, nx_layer is next layer. Let me share a little equation from <b>Tom M Mitchell's ML book(page 80+)</b>.
 
 If the layer is output layer then its error is final error:
\begin{equation}
 \delta_j = \frac{d(E_j)}{d(o_j)} f^1(o_j)
\end{equation}
And for all hidden and input layers:
\begin{equation}
\delta_j = - \frac{d(E_j)}{d(net_j)} = f^1(o_j) \sum_{k=downstream(j)} \delta_k w_{kj}
\end{equation}

Note that: If this layer is output layer, then the error will be the final error and we will not call this method. The term `𝑑(𝐸𝑗)/𝑑(𝑜𝑗)` is the derivative of error function wrt. output. I will share some explanation later on Gradient Descent.

Again going back to our method `backpropagate` here, this method is called only when this layer is not final layer. Otherwise next layer wont exist. Lets take a look into `self.error`, it is brought to this layer from its immediate layer or called `downstream(j)` here. Then we find the delta term. We need first derivative of `activation` funtction of this layer and we do it wrt output. When the term `delta` for this layer is found, we can get `delta_weights` for this layer by multiplying `delta` with this layer's most recent `input`. Similarly `delta_biases` is just the term delta. Note that, the len of delta will be equal to total number of neurons. It stores the delta term for this layer.

## 3.2 Writing a stackking class
AHHHH long journey Aye!!

We will name it `NN`. And we will perform all training operations under this class.
### 3.2.1 Initializing a class. 
(Note that:- the assumption of how many attributes we need will always fail, you might use lesser than initialized or you will create later on). Please follow the written comments below, for explanation.

<code>
    def __init__(self):
        self.layers = [] # a list to stack all the layers
        self.info_df = {} # this dictionary will store the information of our model
        self.column = ["LName", "Input", "Output", "Activation", "Bias"] # this list will be used the header of our summary
        self.parameters = 0 # how many parameters do we have?
        self.optimizer = "" # what optimizer are we using?
        self.loss = "" # what loss function are we using?
        self.all_loss = {} # loss through very epochs, needed for visualizing
        self.lr = 1 # learning rate
        self.metrics = []
        self.av_optimizers = ["sgd", "iterative", "momentum", "rmsprop", "adagrad", "adam", "adamax", "adadelta"] # available optimizers
        self.av_metrics = ["mse", "accuracy", "cse"] # available metrics
        self.av_loss = ["mse", "cse"] # available loss functions
        self.iscompiled = False # if model is compiled
        self.batch_size = 8 # batch size of input
        self.mr = 0.0001 # momentum rate, often called velocity
        self.all_acc = {} # all accuracy
        self.eps = 1e-8 # epsilion, often used to avoid divide by 0.
</code>

And hold on, we will write all optimizers from scratch too. 

### 3.2.2 Writing a method for stackking layers.
<code>
  def add(self, layer):
        if(len(self.layers) > 0):
            prev_layer = self.layers[-1]
            if prev_layer.name != "Input Layer":
                prev_layer.name = f"Hidden Layer{len(self.layers) - 1}" 
            if layer.input_shape == None:
                layer.input_shape = prev_layer.output_shape
                layer.set_output_shape()
            layer.name = "Output Layer"
            if prev_layer.neurons != layer.input_shape and layer.input_shape != None:
                raise ValueError(f"This layer '{layer.name}' must have neurons={prev_layer.neurons} because '{prev_layer.name}' has output of {prev_layer.neurons}.")
        else:
            layer.name = "Input Layer"
        self.layers.append(layer)    
</code>

Lots of dumb things happening under this method. It takes the object of the layer and stacks it to the previous layer. 
First we check if we have more than 0 layers from `self.layers`. If we do, then we set prev_layer to last layer of all layers. And if the name of prev_layer is not "Input Layer" we will name all hidden layer as "Hidden Layer". And if this layer's number of input is none, we set it to the number of neurons of prev_layer. Because any hidden layer will have input as the output of previous layer. And then we call the `set_output_shape` method for weight initialization, and other tasks. Note that number of bias term is equal to number of neurons or nodes, hence we won't have to set them like this. But if this layer's input is given and it doesn't matches the number of neuron of previous layer is not equal then this is invalid assumption and we will throw an error. 

Second, if we have 0 layers, then it is obviously a Input layer. We name it so.

Finally we make a stack of layers(not the data structure stack but a list) by appending them to a list of layers.

### 3.2.3 Lets write a method for a summary. And yes we will test it right now.
<code>
    def summary(self):
        lname = []
        linput = []
        lneurons = []
        lactivation = []
        lisbias = []
        for layer in self.layers:
            lname.append(layer.name)
            linput.append(layer.input_shape)
            lneurons.append(layer.neurons)
            lactivation.append(layer.activation)
            lisbias.append(layer.isbias)
            self.parameters += layer.parameters
        model_dict = {"Layer Name": lname, "Input": linput, "Neurons": lneurons, "Activation": lactivation, "Bias": lisbias}    
        model_df = pd.DataFrame(model_dict).set_index("Layer Name")
        print(model_df)
        print("Total Parameters: ", self.parameters)
</code>

I am taking help of `pandas` library here and instead of write tables like output, why not use table? Nothing huge is hapenning here, but we create a different list for layer name, input shape, neurons, activation, is bias and appended every layer's that value on this. Then after we collected every value of attribute from every layer, we create a dictionary with right keys. Then BAAAAM! we created a dataframe and set index to `Layer Name`. 

Lets write a example:-
<code>
model = NN()
model.add(FFL(input_shape=28*28, 10, activation="softmax"))
model.summary()
</code>
If no errors, then lets proceed.

### 3.2.4 Train Method
Afterall, what use of all those fancy methods if you still not get train method?\
But before that, lets create a method to check if our dataset meets the requirements of model.
<code>
    def check_trainnable(self, X, Y):
        if self.iscompiled == False:
            raise ValueError("Model is not compiled.")
        if len(X) != len(Y):
            raise ValueError("Length of training input and label is not equal.")
        if X[0].shape[0] != self.layers[0].input_shape:
            layer = self.layers[0]
            raise ValueError(f"'{layer.name}' expects input of {layer.input_shape} while {X[0].shape[0]} is given.")
        if Y.shape[-1] != self.layers[-1].neurons:
            op_layer = self.layers[-1]
            raise ValueError(f"'{op_layer.name}' expects input of {op_layer.neurons} while {Y.shape[-1]} is given.")  
</code>

This method takes training input and label, and if is all good then we can walk proudly to train method. We are checking if model is compiled. Well model compilation is done by another method and will present here. Then there are other cases of error. Please see the statement inside `ValueError` for explanation.

Lets write a compiling method, shall we?\
What this method should do is, prepare a optimizer, prepare a loss fxn, learning rate and so on.

<code>
    def compile_model(self, lr=0.01, mr = 0.001, opt = "sgd", loss = "mse", metrics=['mse']):
        if opt not in self.av_optimizers:
            raise ValueError(f"Optimizer is not understood, use one of {self.av_optimizers}.")       
        for m in metrics:
            if m not in self.av_metrics:
                raise ValueError(f"Metrics is not understood, use one of {self.av_metrics}.")       
        if loss not in self.av_loss:
            raise ValueError(f"Loss function is not understood, use one of {self.av_loss}.")       
        self.loss = loss
        self.lr = lr
        self.mr = mr
        self.metrics = metrics
        self.iscompiled = True
        self.optimizer = Optimizer(layers=self.layers, name=opt, learning_rate=lr, mr=mr)
        self.optimizer = self.optimizer.opt_dict[opt]
</code>

This method is under development but, the important part here is last two lines. `Optimizer(layers=self.layers, name=opt, learning_rate=lr, mr=mr)` is a class which encapsulated all our optimizers. When calling a class, it will initialise all our optimizer's necessary terms also. I will provide a that code also but lets take a look at some glimpse. 

<code>
  class Optimizer:
    def __init__(self, layers, name=None, learning_rate = 0.01, mr=0.001):
        self.name = name
        self.learning_rate = learning_rate
        self.mr = mr
        keys = ["sgd"]
        values = [self.sgd]
        self.opt_dict = {keys[i]:values[i] for i in range(len(keys))}
        if name != None and name in keys:
            self.opt_dict[name](layers=layers, training=False)
    def sgd(self, layers, learning_rate=0.01, beta=0.001, training=True):
        learning_rate = self.learning_rate
        for l in layers:
            if l.parameters !=0:
                if training:
                    l.weights += l.pdelta_weights*self.mr + l.delta_weights * learning_rate
                    l.biases += l.pdelta_biases*self.mr + l.delta_biases * learning_rate
                    l.pdelta_weights = l.delta_weights
                    l.pdelta_biases = l.delta_biases
</code>

We will be using only `Gradient Descent` here. I will also provide code to other optimizers on another blog. Things to note are, we create a key as normal string of corresponding optimizer and value as a method.

#### 3.2.4.1 Gradient Descent
(Refer from the chapter 4 (page 80) of Machine Learning by <b> Tom M. Mitchell</b>.)

For weight update, we use this concept along with Back Propagation.
Lets first prepare notations.
\begin{equation}
E_j\ is\ error\ function.
\end{equation}
\begin{equation}
net_j\ is\ soma\ i.e. XW + \theta\ or\ \sum_i{w_{ji}x_{ji}}
\end{equation}
\begin{equation}
o_j\ is\ the\ output\ of\ unit\ j\ due\ to\ the\ activation\ function\ i.e.\ o_j = f(net_j)
\end{equation}
\begin{equation}
t_j\ is\ target\ for\ j
\end{equation}
\begin{equation}
w_{ji}\ is\ weight\ value\ from\ j^{th}\ unit\ to\ i^{th}\ unit. 
\end{equation}
\begin{equation}
x_{ji} is\ the\ input\ value\ from\ j^{th}\ unit\ to\ i^{th}\ unit.
\end{equation}

Note that `𝑑(𝐸𝑗)/𝑑(𝑤𝑗𝑖)` varies with the case, if jth unit is output unit or internal. 

* Case 1: <b>j is output unit</b>.
\begin{equation}
\frac{d(E_j)}{d({w_{ji})}} = \frac{d(E_j)}{d({net_j})} \frac{d(net_j)}{d({w_{ji}})}   
\end{equation}
\begin{equation}
\space = \frac{d(E_j)}{d_{net_j}} x_{ji}
\end{equation}
\begin{equation}
\ = \frac{d(E_j)}{d(o_j)} \frac{d(o_j)}{d(net_j)}x_{ji}
\end{equation}
\begin{equation}
\ = \frac{d(E_j)}{d(o_j)} f^1(o_j) x_{ji}
\end{equation}


* Case 2: <b>j is hidden unit,</b>\
We have to refer to the set of all units immediately downstream of unit j.(i.e all units whose direct i/p include o/p of unit j) and denoted by `downstream(j)`. And `net_j` can influence network o/p by only `downstream(j)`.

\begin{equation}
\frac{d(E_j)}{d({net_{j})}}  = \sum_{k=downstream(j)} \frac{d(E)}{d({net_k})} \frac{d(net_k)}{d({net_{j}})}
\end{equation}
\begin{equation}
\ = \sum_{k=downstream(j)} -\delta_k \frac{d(net_k)}{d({o_{j}})} \frac{d(o_j)}{d({net_{j}})}
\end{equation}
\begin{equation}
\ = \sum_{k=downstream(j)} -\delta_k w_{kj} f^1(oj)
\end{equation}
\begin{equation}
\ reordering\ terms,
\end{equation}
\begin{equation}
\ \delta_j = - \frac{d(E_j)}{d(net_j)} = f^1(o_j) \sum_{k=downstream(j)} \delta_k w_{kj}
\end{equation}

And the weight update term for all units is:-
\begin{equation}
\triangle w_{ji} = \alpha \delta_j x_{ji}
\end{equation}
\begin{equation}
\ when\ momentum\ term\ is\ applied\,
\end{equation}
\begin{equation}
\triangle w_{ji}(n) = \beta \delta_j x_{ji} + \triangle w_{ji}(n-1) 
\end{equation}
\begin{equation}
\ \beta\ is\ momentum\ rate
\end{equation}
\begin{equation}
\delta_j\ formula\ varies\ with\ the\ unit\ being\ output\ or\ internal. 
\end{equation}
\begin{equation}
w_{ji} = w_{ji} -  \triangle w_{ji}\\
\end{equation}

<b> The Gradient Descent algorithm will be easier to understand after we specify the activation function and loss function. Which i will be covering on below parts.</b>

### 3.2.5 Training Method
<code>
def train(self, X, Y, epochs, show_every=1, batch_size = 32, shuffle=True):
        self.check_trainnable(X, Y)
        self.batch_size = batch_size
        t1 = time.time()
        len_batch = int(len(X)/batch_size)
        batches = []
        curr_ind = np.arange(0, len(X), dtype=np.int32)
        if shuffle: 
            np.random.shuffle(curr_ind)
        if len(curr_ind) % batch_size != 0:
            len_batch+=1
        batches = np.array_split(curr_ind, len_batch)
        for e in range(epochs):            
           err = []
            for batch in batches: 
                curr_x, curr_y = X[batch], Y[batch]
                b = 0
                batch_loss = 0
                for x, y in zip(curr_x, curr_y):
                    out = self.feedforward(x)
                    loss, error = self.apply_loss(y, out)
                    batch_loss += loss
                    err.append(error)
                    update = False
                    if b == batch_size-1:
                        update = True
                        loss = batch_loss/batch_size
                    self.backpropagate(loss, update)
                    b+=1
            if e % show_every == 0:      
                out = self.feedforward(X)
                loss, error = self.apply_loss(Y, out)
                out_activation = self.layers[-1].activation
                print(out_activation)
                if out_activation == "softmax":
                    pred = out.argmax(axis=1) == Y.argmax(axis=1)
                elif out_activation == "sigmoid":
                    pred = out > 0.7                    
                elif out_activation == None:
                    pred = abs(Y-out) < 0.000001                    
                self.all_loss[e] = round(error.mean(), 4)
                self.all_acc[e] = round(pred.mean() * 100, 4)                
                print(f"Time: {round(time.time() - t1, 3)}sec")
                t1 = time.time()
                print('Epoch: #%s, Loss: %f' % (e, round(error.mean(), 4)))
                print(f"Accuracy: {round(pred.mean() * 100, 4)}%")    
</code>
    
Alright folks, this is the train method. I hope you are not scared with the size.
Some major steps:
* Check if the dataset is trainnable or not
* Start a timer(or should we start timer after making batches)
* Create a indices of dataset
* If shuffle, then we do shuffle
* Then we create a indices for each batch, we also make each batch of mostly same size but on odd case `np.array_split` does work.
* On every epoch:
    * For each batch:
        * For each x, y on batch:
            * Feed Forward the example set, (method is given below)
            * Find the loss for last layer and error, (method is given below)
            * Add loss to batch loss
            * If current example is last of batch, then we will update parameters
            * We backpropagate the error of current example, (the backpropagate method is given below)
    * If we want to show on this epoch,
        * Feedforward all trainset and take training output.
        * Find train error
        * Find accuracy
        * Take average of error and accuracy and show them.
        * Store loss and accuracy of this epoch(we will visualise later)       
  

### 3.2.6 Write a feedforward method.
<code>
    def feedforward(self, x):
        for l in self.layers:
            l.input = x
            x = l.apply_activation(x)  
            l.out = x
        return x
</code>

Nothing strange is happening here. We take a input vector of single example and pass it to the first layer. Then we set the input of that layer to `x` and get the output of this layer. And also set `out` of this layer to output given by apply_activation method of that layer. Note that we need the ouput of this every layer for backpropation and also the output of one layer acts as input to another. When there is no layers left, we pass the output of last layer(o/p layer) as the output of this input.

### 3.2.7 Next we need a method to find error. We have few error functions on our assumption.
<code>
def apply_loss(self, y, out):
    if self.loss == "mse":
        loss = y - out
        mse = np.mean(np.square(loss))       
        return loss, mse
    if self.loss == 'cse':
        """ Requires out to be probability values. """     
        if len(out) == len(y) == 1: #print("Using Binary CSE.")            
            cse = -(y * np.log(out) + (1 - y) * np.log(1 - out))
            loss = -(y / out - (1 - y) / (1 - out))
        else: #print("Using Categorical CSE.")            
            if self.layers[-1].activation == "softmax":
                """if o/p layer's fxn is softmax then loss is y - out
                check the derivation of softmax and crossentropy with derivative"""
                loss = y - out
                loss = loss / self.layers[-1].activation_dfn(out)
            else:
                y = np.float64(y)
                out += self.eps
                loss = -(np.nan_to_num(y / out) - np.nan_to_num((1 - y) / (1 - out)))
            cse = -np.sum((y * np.nan_to_num(np.log(out)) + (1 - y) * np.nan_to_num(np.log(1 - out))))
        return loss, cse
</code>


The code is pretty weird but math is cute. 

* MSE(Mean Squared Error):- Mean of Squared Error.
\begin{equation}
E = \frac{1}{m} \sum_{i=1}^m(t_i - o_i)^2
\end{equation}
where `o` is output of model and `t` is target or true label.

* CSE(Cross Entropy):- Good for penalizing bad prediction more. 
\begin{equation}
E = \frac{1}{m}\sum_{i=1}^{m} -y*log(h_{(\theta)}(x^i) - (1-y)*log(1-h_{(\theta)}(x^i)
\end{equation}
The loss value returned from above equation is the term required for gradient descent. It will be clear by viewing Gradient Descent.



Recall the delta term from <b>Gradient Descent</b>, as the delta term depends upon the derivative of error function w.r.t weight, we need to find it. In fact our target is to find the term `𝑑(𝐸𝑗)/𝑑(𝑜𝑗)`. It is not that hard by the way.\
i. MSE
\begin{equation}
\frac{d(E_j)}{d(o_j)} =  \frac{d\frac{1}{m} \sum_{i=1}^m(t_i - o_i)^2}{d(o_j)}\\
\end{equation}
\begin{equation}
above\ term\ is\ 0\ for\ all\ except\ i=j\\
\end{equation}
\begin{equation}
\therefore\ \frac{d(E_j)}{d(o_j)} = \frac{d\frac{1}{m} (t_j - o_j)^2}{d(o_j)}\\
\end{equation}
\begin{equation}
\ = -(t_j - o_j)\\
\end{equation}
\begin{equation}
\ and\ term\ \frac{d(E_j)}{d(net_j)} = -(t_j - o_j) f^1(o_j) 
\end{equation}

ii. CSE
I am skipping long derivatives but note that `d(log(x))/d(x)` = `1/x`.
\begin{equation}
E = \frac{1}{m}\sum_{i=1}^{m} -t_i*log(o_i) - (1-t_i)*log(1-o_i)\\
\end{equation}
\begin{equation}
\ now\ term\ \frac{d(E_j)}{d(o_j)} = - \frac{t_i}{o_i} + \frac{1-t_i}{1-o_i} will\ be\ calculated.
\end{equation}

Now going back to our code, what if we have activation function `softmax` for output layer? Well, since we will be using its derivative as `softmax(1-softmax)`. Here `softmax` is `o`. So if we rearrange terms, `𝑑(𝐸𝑗)/𝑑(𝑜𝑗)` = 
`(o-t)/(o(1-o))`. Hence the term `𝑑(𝐸𝑗)/𝑑(𝑤𝑗𝑖)` will be `(o-t)` when using softmax and crossentropy.\
`np.nan_to_num` will turn `nan` value to 0 that we got from `log` or `1/0`. 

### 3.2.8 Backpropagate method:
<code>
    def backpropagate(self, loss, update = True):  
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer == self.layers[-1]:
                layer.error = loss
                layer.delta = layer.error * layer.activation_dfn(layer.out) 
                layer.delta_weights += layer.delta * np.atleast_2d(layer.input).T
                layer.delta_biases += layer.delta
            else:
                nx_layer = self.layers[i+1]
                layer.backpropagate(nx_layer)
            if update:
                layer.delta_weights /= self.batch_size
                layer.delta_biases /= self.batch_size
        if update:      
            self.optimizer(self.layers)
            self.zerograd()
</code>

This method is called per example on every batch on every epoch. What happens is, when we pass the loss of model and update term, it run over every layers and checks updates the delta term for all parameters. 
More simply:
* For every layer from output to input:
    * If this layer is output layer, find delta term now 
    * If this layer isnt output layer, call the `backpropagate` method of that layer and send next layer also.
    (I have already provided a individual `backpropagate` method for Feedforward layer.)
    * If we want to update the parameters now, then average the delta terms
* If we are updating, then call the optimizer method, if we looked back to `compile` method, then we can see that `self.optimizer` is holding a reference to the method of `Optimizer` class. We pass the entire layers again here.
* Now we have updated our parameters, we need to zero all the gradients terms. So we have another method, `zerograd`.

<code>
def zerograd(self):
    for l in self.layers:
        l.delta_weights=0
        l.delta_biases = 0
</code>

It is pretty simple here. But once we are working with more than one type of layers, it will get messy.


<code>
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = x_train.reshape(-1, 28 * 28) 
    x = (x-x.mean(axis=1).reshape(-1, 1))/x.std(axis=1).reshape(-1, 1)
    y = pd.get_dummies(y_train).to_numpy()
    m = NN()
    m.add(FFL(784, 10, activation="softmax")) 
    m.compile_model(lr=0.01, opt="sgd", loss="cse", mr= 0.001)
    m.summary()
    m.train(x[:], y[:], epochs=100, batch_size=32)
</code>

This is a classification problem.

I am using `keras` just for getting mnist dataset. We can get mnist data from official website also. Then we normalize our data by substracting mean and dividing with its corresponding standard deviation. Thanks to NumPy. Then we converted our data to one hot encoding using pandas method, get_dummies and convert it to numpy array. We created a object of NN class and then added a Feed Forward layer of input shape 784 and neurons 10, we gave activation function as softmax. Since mnist dataset is 28X28 on each example, we made single image of shape 28*28. Softmax function is very useful for classification problems and usually used on last layers. Something like below happens but accuracy increases very slowly.\
`
Time: 19.278sec
Epoch: #20, Loss: 3208783.038700
Accuracy: 88.55%
`

We can make our dataset to `one hot encoded` vector using below method also:\
`
def one_hot_encoding(lbl, classes):
    encoded = np.zeros((len(lbl), classes))
    c = list(set(lbl))
    if len(c) != classes:
        raise ValueError("Number of classes is not equal to unique labels.")
    for i in range(len(yy)):
        for j in range(len(c)):
            if c[j] == lbl[i]:
                encoded[i, j] = 1
    return encoded
`

With model like below accuracy was great
<code>
m = NN()
m.add(FFL(784, 100, activation="sigmoid")) 
m.add(FFL(100, 10, activation="softmax")) 
m.compile_model(lr=0.01, opt="adam", loss="cse", mr= 0.001)
m.summary()
m.train(x[:], y[:], epochs=100, batch_size=32)
</code>

# 4. Lets do something interesting.
## 4.1 Preparing Train/Validate data
Up to now, we have done some training only. But it is <b>not a good idea to boast the train accuracy.</b> We need to take validaion data also. For that lets modify our few methods. First, we will edit `__init__` method of our `NN`.\
`
self.train_loss = {} # to store train loss per view_every
self.val_loss = {} # to store val loss per view_every
self.train_acc = {} # to store train acc per view_every
self.val_acc = {} # to store val acc per view_every
`

Next, change `train` method as below:\

<code>
def train(self, X, Y, epochs, show_every=1, batch_size = 32, shuffle=True, val_split=0.1, val_x=None, val_y=None):     
        self.check_trainnable(X, Y)
        self.batch_size = batch_size
        t1 = time.time()
        curr_ind = np.arange(0, len(X), dtype=np.int32)
        if shuffle: 
            np.random.shuffle(curr_ind)            
        if val_x != None and val_y != None:
            self.check_trainnable(val_x, val_y)
            print("\nValidation data found.\n")
        else:
            val_ex = int(len(X) * val_split)
            val_exs = []
            while len(val_exs) != val_ex:
                rand_ind = np.random.randint(0, len(X))
                if rand_ind not in val_exs:
                    val_exs.append(rand_ind)
            val_ex = np.array(val_exs)
            val_x, val_y = X[val_ex], Y[val_ex]
            curr_ind = np.array([v for v in curr_ind if v not in val_ex])                
        print(f"\nTotal {len(X)} samples.\nTraining samples: {len(curr_ind)} Validation samples: {len(val_x)}.")           
        batches = []
        len_batch = int(len(curr_ind)/batch_size) 
        if len(curr_ind)%batch_size != 0:
            len_batch+=1
        batches = np.array_split(curr_ind, len_batch)     
        print(f"Total {len_batch} batches, most batch has {batch_size} samples.\n")       
        batches = []
        if(len(curr_ind) % batch_size) != 0 :
            nx = batch_size-len(curr_ind) % batch_size
            nx = curr_ind[:nx]
            curr_ind = np.hstack([curr_ind, nx])  
        batches = np.split(curr_ind, batch_size)   
        for e in range(epochs):            
            err = []
            for batch in batches:
                a = [] 
                curr_x, curr_y = X[batch], Y[batch]
                b = 0
                batch_loss = 0
                for x, y in zip(curr_x, curr_y):
                    out = self.feedforward(x)
                    loss, error = self.apply_loss(y, out)
                    batch_loss += loss
                    err.append(error)
                    update = False
                    if b == batch_size-1:
                        update = True
                        loss = batch_loss/batch_size
                    self.backpropagate(loss, update)
                    b+=1
            if e % show_every == 0:      
                train_out = self.feedforward(X[curr_ind])
                train_loss, train_error = self.apply_loss(Y[curr_ind], train_out)
                out_activation = self.layers[-1].activation
                val_out = self.feedforward(val_x)
                val_loss, val_error = self.apply_loss(val_y, val_out)
                if out_activation == "softmax":
                    train_acc = train_out.argmax(axis=1) == Y[curr_ind].argmax(axis=1)
                    val_acc = val_out.argmax(axis=1) == val_y.argmax(axis=1)
                elif out_activation == "sigmoid":
                    train_acc = train_out > 0.7
                    val_acc = val_out > 0.7
                elif out_activation == None:
                    train_acc = abs(Y[curr_ind]-train_out) < 0.000001
                    val_acc = abs(Y[val_ex]-val_out) < 0.000001                    
                self.train_loss[e] = round(train_error.mean(), 4)
                self.train_acc[e] = round(train_acc.mean() * 100, 4)                
                self.val_loss[e] = round(val_error.mean(), 4)
                self.val_acc[e] = round(val_acc.mean()*100, 4)
                print(f"Epoch: {e}, Time: {round(time.time() - t1, 3)}sec")               
                print(f"Train Loss: {round(train_error.mean(), 4)} Train Accuracy: {round(train_acc.mean() * 100, 4)}%")
                print(f'Val Loss: {(round(val_error.mean(), 4))} Val Accuracy: {round(val_acc.mean() * 100, 4)}% \n')
                t1 = time.time()
</code>

The pseudo code or explation of above code is:
* Check trainnable training data.
* Prepare indices from 0 to no. examples.
* If validation data is given on `val_x, val_y` then check their trainnable also.
* Else, we will split the prepared indices of data for train and validation.
* First we get number of indices for validation, then get indices for them and data too.
* We will also edit `curr_ind` Instead of using actual data, i am using only indices because of memory.
* Then train just as above processes.
* For `show_every`, We do pass entire train data and get accuracy, loss. And do similar for vlaidation set.

## 4.2 Lets add some visualizing methods
<code>
    def visualize(self):
        plt.figure(figsize=(10,10))
        k = list(self.train_loss.keys())
        v = list(self.train_loss.values())
        plt.plot(k, v, "g-") 
        k = list(self.val_loss.keys())
        v = list(self.val_loss.values())
        plt.plot(k, v, "r-")
        plt.xlabel("Epochs")
        plt.ylabel(self.loss)
        plt.legend(["Train Loss", "Val Loss"])
        plt.title("Loss vs Epoch")
        plt.show()
        plt.figure(figsize=(10,10))
        k = list(self.train_acc.keys())
        v = list(self.train_acc.values())
        plt.plot(k, v, "g-")
        k = list(self.val_acc.keys())
        v = list(self.val_acc.values())
        plt.plot(k, v, "r-")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Acc vs epoch")
        plt.legend(["Train Acc", "Val Acc"])
        plt.grid(True)
        plt.show()
</code>

Nothing strange happening here. We are only using the keys and values of previously stored train/val acc/loss. If we set `show_every=1` then, graph will be shown great.

# 5 Finally
My version of final Feedforward Deep Neural Network will be given on the link and at the meantime, i am gonna share my results.

<code>
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = x_train.reshape(-1, 28 * 28) 
    x = (x-x.mean(axis=1).reshape(-1, 1))/x.std(axis=1).reshape(-1, 1)
    y = pd.get_dummies(y_train).to_numpy()
    xt = x_test.reshape(-1, 28 * 28) 
    xt = (xt-xt.mean(axis=1).reshape(-1, 1))/xt.std(axis=1).reshape(-1, 1)
    yt = pd.get_dummies(y_test).to_numpy()
    m = NN()
    m.add(FFL(784, 10, activation='sigmoid')) 
    m.add(FFL(10, 10, activation="softmax")) 
    m.compile_model(lr=0.01, opt="adam", loss="cse", mr= 0.001)
    m.summary()
    m.train(x[:], y[:], epochs=10, batch_size=32, val_x=xt, val_y = yt)
    m.visualize()
</code>

# 6 Bonus Topics

## 6.1 Saving Model on JSON File

<code>
import os
import json
def save_model(self, path="model.json"):
    """
        path:- where to save a model including filename
        saves Json file on given path.
    """
    dict_model = {"model":str(type(self).__name__)}
    to_save = ["name", "isbias", "neurons", "input_shape", "output_shape", "weights", "biases", "activation", "parameters"]
    for l in self.layers:
        current_layer = vars(l)
        values = {"type":str(type(l).__name__)}
        for key, value in current_layer.items():
            if key in to_save:
                if key in ["weights", "biases"]:
                    value = value.tolist()
                values[key] = value
        dict_model[l.name] = values
    json_dict = json.dumps(dict_model)    
    with open(path, mode="w") as f:
        f.write(json_dict)
save_model(m)
</code>

> Note that, we are not saving parameters on encrypted form and neither we are saving it on different files.
* We want to save everything on JSON format so, we are creating dictionary first.
* `vars(obj)` allows us to create a dictionary from `attrib:value` structure of class object.
* We are about to save only few things necessary to use a model. `to_save` is a list of all the attributes that we need to predict a model. 
* Still we haven't implemented a way to check if saved model is compiled or not. But we do need a `predict` method.


## 6.2 Loading a JSON Model
<code>
def load_model(path="model.json"):
    """
        path:- path of model file including filename
        returns:- a model
    """    
    models = {"NN": NN}
    layers = {"FFL": FFL}
    """layers = {"FFL": FFL, "Conv2d":Conv2d, "Dropout":Dropout, "Flatten": Flatten, "Pool2d":Pool2d}"""
    with open(path, "r") as f:
        dict_model = json.load(f)
        model = dict_model["model"]
        model = models[model]()
        for layer, params in dict_model.items():
            if layer != "model":                
                lyr = layers[params["type"]](neurons=params["neurons"])# create a layer obj
                if params.get("weights"):
                    lyr.weights = params["weights"]
                if params.get("biases"):
                    lyr.biases = params["biases"]
                lyr.name = layer
                lyr.activation = params["activation"]
                lyr.isbias = params["isbias"]
                lyr.input_shape = params["input_shape"]
                lyr.output_shape = params["output_shape"]
                lyr.neurons = params["neurons"]
                lyr.parameters = params["parameters"]
                model.layers.append(lyr)
        return model
m = load_model()
</code>

* Nothing is strange here. But few things to note is, `FFL` is a method's address. And `NN` is a class which we will call later.
* The model is created on line `model = models[model]()`.
* First test of if our model works or not can be seen from `m.summary()`.
* Next try to use the `predict(x)` method.

## 6.3 Predict Method
<code>
    def predict(self, X):
        out = []
        for x in X:
            out.append(self.feedforward(x))
        return out
</code>


# 7 References and Credits
* [Optimizers were referenced from here](www.github.com/ShivamShrirao/dnn_from_scratch)
* [About Softmax Activation Function and Crossentropy](www.sefiks.com/2017/11/08/softmax-as-a-neural-networks-activation-function)
* Machine Learning by Tom M Mitchell 
* Tensorflow For Dummies by Matthew Scarpino
* [Artificial Intelligence Deep Learning Machine Learning Tutorials](www.github.com/TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials)(Most awesome repository.)
* Grokking Deep Learning by Andrew Trask


```python

```
