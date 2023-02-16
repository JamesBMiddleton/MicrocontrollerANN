# MicrocontrollerANN
<p float="left" align="middle" border="0" padding="0" margin="0">
<img src="https://user-images.githubusercontent.com/73485794/219461093-48db8680-01a6-4c4e-b943-4023e8bdf0e1.gif" alt="1" width = 500px>
<img src="https://user-images.githubusercontent.com/73485794/219456615-4837c9aa-6c16-4c9a-bf55-155be06a4e50.gif" alt="2" width = 500px>
</p>


Visualising the training of an Artifical Neural Network (ANN) on a microcontroller using an LED matrix.


__Neural network:__
- Implemented from scratch.
- Multi-layer perceptron with 2 inputs, 2 hidden layers and 1 output in a 2-3-3-1 arrangement.
- Mini-batch gradient descent backpropagation.
- Sigmoid activation function.
- Learning rate scheduling.
- A sample regression problem for training the model was generated using [sklearn.datasets.make_regression()](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html).

__Visualisation details:__<br>
- _forwards pass:_
  - Each forward pulse represents the path taken by a single instance of the dataset passing through the network.
  - The brightness of each node corresponds to the output value of that node.
  - The brightness of each connection between nodes corresponds to the output from the previous node multiplied by its associated input weight.
  - Brightnesses of nodes and connections scaled with each forward pass using min-max normalization to provide a better visualisation of the differences between different nodes and connections. 
  - 32 forward passes occur (the batch size) for each backwards pass.
- _backwards pass:_
  - each backward pulse represents the updating of input weights using the average of the gradients collected over a mini-batch of instances.
  - The brightness of each node corresponds to the sum of gradients propagating backwards through that node.
  - The brightness of each connection corresponds to the value of the gradient that will be used to update the weight.
  - min-max normalization is used for both nodes and connections to provide a better visualisation of differences.
- The toggleable value in the lower right of the display represents lowest loss achieved over a mini-batch during training.
- The toggleable coloured connections represents positive weights with red and negative weights with blue.


__Hardware components:__
- [Adafruit Matrix Portal M4](https://learn.adafruit.com/adafruit-matrixportal-m4?view=all)
- [Adafruit 64x32 RGB LED Matrix - 3mm pitch](https://www.adafruit.com/product/2279)

__Key resources:__
- [Adafruit GFX library](https://learn.adafruit.com/adafruit-gfx-graphics-library/overview)
- [The essence of calculus](https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PL0-GT3co4r2wlh6UHTUeQsrf3mlS2lk6x&index=2&t=27s)
- [Deep learning maths](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Backpropagation example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
- [How LED matrices work](https://www.instructables.com/Practical-Guide-to-LEDs-4-Matrix-Multiplexing/#ALLSTEPS)


