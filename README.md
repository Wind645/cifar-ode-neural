# cifar-ode-neural
I was struggling understanding the idea of neural ODE, one of the best articles in 2018 NeuralPS. 
I have copied the code line by line from the already implemented version of neural ODE, and I tried 
to make a small image classification task with cifar 10. Unluckily, it did not work well but I don't know why.

The idea is that, since neural ODE can model the transformation of the data, curving a kind of dynamic system, 
an idea occurred to me that why not regarding the classification problem as something evolving from the raw pixels
to the classes, so I implemented a conv net to model the derivative of z, and added a small fully connected layer 
at last to compress the dimensions from 32 * 32 * 3 to 10, and this architecture simply did a little better than 
pure one layer fc net, and a lot worse than a ikunnet I designed earlier.

I think the issue might be due to the poor design of loss, as the transformation process don't have a good loss function 
to be supervised? Maybe? But I have not find the solution yet.
