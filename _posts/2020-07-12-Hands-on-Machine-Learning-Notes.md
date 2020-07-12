---
layout: page
title: "Notes from Hands on ML with Scikit Learn and Tensorflow (2nd Ed)"
date: 2020-07-12 11:40:00 +0530
categories: Notes Deep-learning
---

## Neural Networks

- McCulloch & Pitts model was known as an *artificial neuron* and was able to solve basic logic operations - AND, OR, NOT. This could be combined to form more complex operations
- ***Perceptron*** by Rosenblatt evolved slightly differently from an artificial neuron. There'd be a threshold unit that controls whether the neurons fire or not. The weighted sum of inputs is passed through a *step function* which would then apply the threshold operation on the linear combination and decide whether to fire. A perceptron would contain multiple such *threshold logic units*, **each of which receive inputs. Also there'd be a bias parameter applied to the linear combination of inputs at each TLU. In other words, the whole architecture is that of a ***Fully Connected Layer*** and the step function is actually the ***Activation function***. Works well for linearly separable data only, which is a limitation
- A Perceptron fails to solve XOR (in fact, other linear classifiers like Logistic Regression fails too!). But this is overcome by stacking multiple perceptrons - ***Multi Layer Perceptron (MLP)***

### MLP and Backprop

- MLP replaced *step function* with *logistic function*. This is continuous and differentiable and therefore it's possible to work with gradients (a *step function* consists of flat surfaces which can't have gradients, mathematically). Alternate activations are - *tanh*, *relu*, etc.
- Activations primarily are included in the neural net architecture to introduce non-linearity

### Keras API

1. Sequential API
2. Functional API
3. Model Subclassing

- Model saving and loading - saves model architecture + weights + optimizer
- For Sequential and Functional API - `model.save(..)` and `tf.keras.models.load_model(...)` are used
- Model save/load has to be done differently with sub-classing ; `save_weights()` and `load_weights()`
- Callbacks - Custom callbacks, Checkpointing, Tensorboard
- Keras scikit wrappers - can be used to wrap around Keras models to make them behave like scikit-learn models (with `fit()`, `score()` and `predict()`
- Hyperparameter fine-tuning can be done using `GridSeachCV` or `RandomSearchCV` (if Keras scikit wrapped). Alternately, there are packages like `hyperopt`, `hyperas`, etc.
- Fine-tuning hyperparameters:
    - Number of hidden layers
    - Number of hidden units in a layer
    - Learning rate, Batch size, etc.

*Stretch Pants* approach - for fine-tuning hyperparameters, it's a good idea to pick up a larger model and stop the training (early stopping) once it converges to the optimal results. This helps avoid bottleneck layers. Rather than wasting time finding the right fit, its better to start with a size larger than required and let it shrink to the optimal fit.

On the flip side, fewer neurons in a layer may not have the required representational power and hence information is lost and can never be recovered in the subsequent layers.

> **Tip**: In general you will get more bang for your buck by increasing the number of layers instead of the number of neurons per layer.

- **Learning Rate**: Ideally, the rate should be initialized to a very low value and then gradually increased to a higher value. More on this below.
- **Batch size**: Larger batch sizes may/may not generalize well. A good strategy would be to begin with a large batch size along with *learning rate warmup*. If the training is unstable, then switch to a smaller batch size.

> Learning Rate and Batch size go together. Better to tweak them together.

### Training Deep Neural Nets

- Potential problems faced - Vanishing & Exploding gradients

    As the input flows from layer to layer, the input/output variances would vary depending on the activation function. This is down to the way the layer weights are initialized and how the activation function behaves in these situations. The behaviour was strongly evident with sigmoid activation. This leads to an explosion of values when the gradients are calculated in some cases, or sometimes the gradients are extremely low. 

    With either of these outcomes, the convergences of the network is impacted. With exploding gradients problem, the weights would be iteratively growing into abnormally high values, while in the case of vanishing gradients, the gradient values are so low that they hardly affect the weights during backprop - therefore, the weights barely change. This is especially evident in the lower layers of the network because the gradients vanish (lower values multiplied by lower values result in extremely low values) as they propagate towards the initial layers of the network. 

    To stabilize this, research pointed towards layer weight initialization and alternate activation functions. 

    **Side note**: Deviating from previously popular sigmoid activations and weight initialization from standard normal distribution. The mean of gradient of activation for a sigmoid is roughly near 0.5, which is a deviation from the underlying distribution of weight initialization. This was empirically shown to be leading to excess variance seen in the network, thereby diluting the backpropagation.

    Alternatives suggested were - Glorot / Glorot uniform initialization,  ReLU activation and few more depending on the layers used.

    Using Glorot (Xavier) and He initialization, the weights were initialized based on mean 0 and variance $fan_{avg}$. Where, $fan_{avg} = (fan_{in} + fan_{out}) / 2$

    [Initialization parameters](https://www.notion.so/f990170ba4db4320965cb58d71682563)

    For example, 

    `keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")`

    ReLU - Rectified Linear Unit works well for deep neural nets more often. However, there's a problem. Although this never saturates for positive values, the negative values would always transform to zeros. This leads to *dying* neurons, where the negative values lead to zero activation across the neuron and they are largely unaffected during backprop too (gradient of zero being zero).

    Alternatives for ReLU are:

    Leaky ReLU - negative values induce a non-zero activation (defined by a *leak* parameter)

    Randomized Leaky ReLU - The *leak* parameter is randomly set during training ; average value is set for testing

    Parametric Leaky ReLU (PReLU) - leakage is a hyperparameter ; better with large image datasets

    Exponential Linear Unit or ELU - negative values induce a non-zero activation defined by an exponential function ; this would ensure there's a non-zero gradient during backprop, thereby averting vanishing gradients (may outperform all the above)

    $α(exp(z) − 1), if z < 0$ or $z, if z ≥ 0$

    However due to the involvement of exponent, ELU is *slower* than its counterparts.

    SELU or Scaled-ELU was found to be very effective when the deep neural net has a stack of dense layers exclusively. Scaling means the activation output would self-normalize (mean=0, sd=1). However, there are certain other constraints to be also followed to achieve this self-normalization.

### Batch Normalization

Batch Normalization helps in fixing the issue of vanishing gradients (**have a tremendous impact in deeper neural nets**). Also, it enables faster training (with fewer training steps in SOTA architectures).

The intuition is to have the input standardized (for a mini-batch). It works on the below parameters:

1. The input mean estimated on the mini-batch
2. The input standard deviation on the mini-batch
3. The **scaling vector** (one per each unit) by which the batch normalized input would be scaled. This is parameterized by $gamma$ 
4. The **offset vector** (one per each unit) by which the input is offset/shifted. This is parameterized by $beta$

The scaling and offset vectors are applied to the input through element-wise multiplication as shown:

$z(i) = γ ⊗ x(i) + β$

where, the normalized input $x(i)$ is also *smoothened* by $ε$ (a small, non-zero value) to prevent division by zero errors.

Now, for the validation/testing phase there are two approaches on how to calculate the mean/sd for normalizing the inputs. One approach would be to pass the entire training inputs as a single batch and let the BN layer calculate the mean/sd. Another approach (by default, used in Keras BN layers) is to calculate the moving average of layer means and sd during training.

In short, the BN layers use ***batch statistics*** during training phase and ***final statistics*** (based on moving averages) during the testing phase.

Although, it speeds up training, BN layer comes with a penalty as well - these added computation slows down predictions. Therefore, as a workaround the BN layer could be fused with the preceding layer (and getting rid of the separate BN layer as such). This can be represented by the modified weights and biases as shown:

$XW + b$ becomes $XW' + b'$

where

$W' = γ⊗W/σ$ and $b' = γ⊗(b – μ)/σ + β$

While the parameters are considered, each unit for which BN is carried out will have 4 associated parameters, of which the mean/sd are moving averages calculated from the data and therefore are not trainable. Only, the other 2 parameters are trainable.

The authors of Batch Normalization paper argue that including BN layers is better off if they come before the activation function rather than after it (debatable claim). However to do that in Keras, one just has to plug out the activation argument from the respective hidden layer and add the activation separately (ofcourse, after the BN layer).

Batch Normalization hyperparams:

- Momentum (closer to 1, i.e 0.9, 0.99, etc.) is used for calculating the final statistics
- Axis (default -1, meaning final axis) is used to define across which axis is the batch statistics calculated.

    For 2D, $(batch_size, features)$ each feature would be normalized across the batch. More details can be found in the book.

### Gradient Clipping

- To avert exploding gradients, the gradient values can be clipped based on some threshold during backpropagation. This is known as *gradient clipping*. **The *threshold* is a tune-able hyperparameter*.*
- More often, used with RNNs as BN layer could be tricky here
- In Keras, this can be introduced by adding the $clipvalue$ / $clipnorm$ argument while initializing the optimizer. Trying both options would be a good idea to see which one does well with the given dataset (yes, they work differently).

### Transfer Learning

- Always clone the model before reusing weights (and re-training them) to prevent existing weights from being modified
- Layers can be flagged as trainable (or non-trainable)
- Always compile the model before and after the layers are *freezed/unfreezed*
- For very similar tasks, it's not a bad idea to freeze (reuse) all layers and replace just the output layer. Depending on how well the model performs, one may try to unfreeze one or two top layers
- For the unfreezed (reused) layers, it's a good to reduce the learning rate for them, so that the fine-tuned weights aren't wrecked
- It's also a good approach to add/remove hidden layers depending on how much data is available to train

### Optimizers

Helps speed up training in different ways.

Generally, weight updates are *learning_rate* ($η$) times the *parameter gradient* ($∇ θ J (θ)$)

- **SGD** - uses fixed, regular updates to optimize the weights

    $θ ← θ - η∇ θ J (θ)$

- **Momentum** - adds a momentum hyperparameter ($β$)

    The intuition here is that the momentum parameter update would accelerate the weight update (not just speed up) at each step, particularly when the slope is steep.

    - the update is applied to the momentum parameter rather than directly to the weights

        $m ← βm − η∇ θ J (θ)$

    - the calculated momentum ($m$) is then added as the weight update

        $θ ← θ + m$

    - If the inputs in the upper layers have varying scales, the optimizer usually takes time to converge (due to the elongated bowl in the convex curve). Especially in the absence of batch normalization, the intermediate outputs could be of varying scales. This is when momentum helps in better convergence

    `optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)`

- **Nesterov's Adaptive Gradient** - NAG is an improvement over momentum.

    NAG intuitively modifies the gradient term, it's not just a parameter gradient. The previous momentum is also considered while calculating the gradient. This serves the purpose of adapting the acceleration w.r.t the slope.

    $m ← βm − η∇ θ J (θ + βm)$

    $θ ← θ + m$

    The intuition behind including the previous momentum in gradient calculation is to provide a more accurate direction for the new gradient. This would take the information from the current step momentum and apply the gradient, which is shown to converge better and therefore faster

    `optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)`

- **AdaGrad** - An improvement over SGD, it is able to adapt the direction of descent accurately and also adapt the momentum towards the optimum

    Weight update is like in SGD, but the update quantity is normalized/scaled down by the *square root of the product of gradient term*

    $s ← s + ∇ θ J (θ) ⊗ ∇ θ J (θ)$

    $θ ← θ − η ∇ θ J (θ) ⊘ √ s + ε$

    Here, $√ s + ε$ is more like $l_2$ norm, where $ε$ is the *smoothening* factor to avoid division by zero error.

    However, shown to be working well **only with simpler problems/models** and not with complex ones.

- **RMSProp** - an improvement over AdaGrad to make it work with complex models as well. A tune-able hyperparameter ($β$) is a slight modification in the update formula:

          $s ← βs + (1 − β) ∇ θ J (θ) ⊗ ∇ θ J (θ)$

          $θ ← θ − η ∇ θ J (θ) ⊘ √ s + ε$

    Typically $**β$** is **0.9**

    `optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)`

- **Adam** is a further improvement (by far, widely popular and stable). The optimizer combines  momentum with RMSProp.

          $m ← β_1 m − (1 − β_1 ) ∇ θ J (θ)$

          $s ← β_2 s + (1 − β_2 ) ∇ θ J (θ) ⊗ ∇ θ J (θ)$

    Normalize m and s (refer book for formula)

          $θ ← θ + η m̂ ⊘ √ ŝ + ε$

    **Adamax** uses $l_{inf}$ norm in place of $l_2$ norm:

    $s ← max(β s, ∇ J (θ))$

    `optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)`

    **Nadam** combines Nesterov's Adaptive Gradient (NAG) with Adam.

**Intuition**: The choice of optimizer for a given problem is analogous to the choice of a screw-driver tool for a given screw. The tool may provide one with different threading/size and the choice is made based on what's the most suited threading for a given screw. In the context of optimizers, some datasets may be allergic to adaptive methods and this is where purely momentum based optimzers fare well.

All the optimizers mentioned above produce dense models (i.e., nonzero weight parameters). However, this could be optimizer further by applying regularization ($l_{1}$) or using **TF-MOT**.

### Learning Rates (in Keras)

**Learning Rate scheduler** - Refer book for details

Can be used as a custom (or keras built-in) function which is called as a callback - `LearningRateScheduler`.

- **Power scheduling**

    $n_{t} = n_{0}/(1 + t/s)^c$ ; the learning rate $n_{t}$ would be diminishing at every $s$ steps. In Keras, this is enabled using the `decay` parameter:

    `optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)`

- **Exponential scheduling**

    $n_{t} = n_{0} * 0.1^{t/s}$ ; the learning rate $n_{t}$ would diminish by the power of 10 for every $s$ steps. Following implements the scheduling technique:

    ```python
    def exponential_decay(lr0, s):
        
        def exponential_decay_fn(epoch):
    	return lr0 * 0.1**(epoch / s)
        
        return exponential_decay_fn

    exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

    lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
    ```

- **Piecewise constant scheduling**

    Uses a constant learning rate for the first few epochs, then reduces the learning rate and keep it constant for the next few epochs and so on.

    ```python
    def piecewise_constant_fn(epoch):
        if epoch < 5:
    	return 0.01
        elif epoch < 15:
    	return 0.005
        else:
    	return 0.001
    ```

- **Performance scheduling** - use `ReduceLROnPlateau` **callback in Keras

    Measures the validation error for N steps and then reduce the learning rate by a factor of ${lambda}$ when the error is no longer dropping.

    `lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)`

- **1cycle scheduling**

    Uses an initial learning rate $n_{0}$ and a maximum learning rate $n_{1}$. The scheduler linearly increases the rate upto $n_{1}$ over a few epochs and then dropping it down to $n_{0}$ (again, linearly). If momentum is involved, then its value is also experimented with during the course of the training.

### Regularization

> With four parameters I can fit an elephant and with five I can make him
wiggle his trunk.
        —**John von Neumann**, cited by **Enrico Fermi** in ***Nature***

Early Stopping & Batch Normalization are (sort of) regularization techniques. Further to those, there are others mentioned below:

- $l_1$ and $l_2$ - $l_{1}$ constraints the values through regularization, while $l_{2}$ results in a sparse set of values.
- Dropout
