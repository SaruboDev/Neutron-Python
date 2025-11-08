# Neutron

A simple wrapper for jax, equinox and optax.

Made by **SaruboDev**: <https://github.com/SaruboDev>.

Copyright: **Apache License 2.0 ©**, see github page for more.


Should support other modules that interact with jax.

### Quickstart
You can use Neutron wrapper *mostly* like you would use Tensorflow.

Make your model class in jax style, call my Model in neutron.models, compile and fit!

Adds a few quality of life stuff to make you able to speedrun tests, i'm not yet sure about big-project development with this module.
The goal with this wrapper is to make jax more accessible and train your models faster with Jit compilation.

Keep in mind that the wrapper is made in Python, so even if i'm giving you access to jit compilation and jax speed, the wrapper itself
will not be lightning-fast (but i doubt you'll notice slowness because of it).

After you've defined your model classes (for me i'll use a simple ViT transformer based on <https://docs.kidger.site/equinox/examples/vision_transformer/>,
you can check it out in the Neutron's docs where i'll explain more) you will need:
- A dataset (i'm using the Cifar100 for a quick test, you can download your datasets, or use tf/torch, whatever you prefer).
- Initialize Neutron's Model class.
It's really quick and easy:
```py
    from neutron.models import Model

    # Note that you gotta remember which arguments your model needs as this class doesn't suggest them.
    model = Model(
        Vit,
        img_size = 32,
        embedding_dim = 192,
        num_heads = 1,
        num_layers = 1,
        dropout_rate = 0.01,
        patch_size = 4,
        num_classes = 100,
        channels = 3,
        seed = 42
    )
``` 
Now we call the compile function, where we specify the optimizer we want to use, which losses, metrics, callbacks and
how much gradient accumulation you want to use:
```py
    import optax
    from neutron.metrics import accuracy

    model.compile(
        optimizer = optax.adam(learning_rate = 0.0001),
        loss = optax.softmax_cross_entropy_with_integer_labels,
        metrics = [accuracy],
        callbacks = [],
        gradAccSteps = None
    )
``` 
This compile has a few different ways you can add losses and metrics (even for multi-task), you can see more about it in the docs for it.

And now, finally, we can just call the fit function, where the wrapper will iterate over epochs, steps and will handle keys, callbacks and everything:
```py
    hist = model.fit(
        data = df_gen_train,
        data_eval = df_gen_test,
        batch_size = batch_size,
        epochs = 2,
        steps_for_epoch = steps_train, # You can leave it to None (default) for auto-calculated steps. Same for eval.
        steps_for_eval = steps_eval
    )
``` 
Note that for the fit we're using data and data_eval, because i'm using my own batchLoader generator (which is included in Neutron), but
the function also supports x/y_train and x/y_eval, along with a few extra options to suit your needs. See more about it in the docs.

The results of our model training can be seen in the history, which will be automatically returned by the wrapper, in our case, the output
from the model, considering verbose set to true to show a progress bar and me using an RTX 3060 12Gb to train the model, will be:
```py
    Starting epoch 1 / 2
    Loss: 4.6406 | accuracy : 0.0097 : 100%|████████████████████████████████████████████████████████████████████████████████| 1.56k/1.56k [01:27<00:00, 17.9it/s]
    Starting Evaluation...
    Loss: 4.6228 | accuracy : 0.0385 : 100%|████████████████████████████████████████████████████████████████████████████████████| 312/312 [00:08<00:00, 35.0it/s]
    Starting epoch 2 / 2
    Loss: 4.6281 | accuracy : 0.0100 : 100%|████████████████████████████████████████████████████████████████████████████████| 1.56k/1.56k [01:22<00:00, 18.9it/s]
    Starting Evaluation...
    Loss: 4.6198 | accuracy : 0.0000 : 100%|████████████████████████████████████████████████████████████████████████████████████| 312/312 [00:07<00:00, 39.5it/s]
    
    History:
    {'loss': [4.640584966995352, 4.628129463378201], 'accuracy': [0.009703104993597951, 0.010043213828425096], 'val_loss': [4.6227939902886765, 4.619836975699633], 'val_accuracy': [0.038461538461538464, 0.0]}
``` 
You can see (especially from the evaluation accuracy) 1 layer with 1 head is... definitely not much.
But it works as intended!
Also note that metrics are shown as per-epoch values, and not a sum of all the prior epochs + the current one, so if the first epoch finished with loss of "4.6406" (like our first epoch in the example) the second one doesn't start from "4.6406" but will start from 0.

And that's pretty much it for the quickstart of Neutron Wrapper. It will possibly grow more and more with time as i add new things or optimize the code more.
In the meanwhile you can check it out and suggest new features and bug-fixes, thanks a lot!
    

#### What it currently adds:
- A Model class that accepts custom models and makes you able to fit instantly without defining your own train cycle.
    - **Supports**:
    - Single Task Models;
    - Multi Task Models (mostly);
    - Self-supervised Models;
    - Gradient Accumulation;
    - Callbacks;
    - Metrics for train and evaluation.
    - A summary of your model, giving total params, trainable params and non-trainable params.
    - Evaluation is also done at the end of each epoch if the user specified at least x_eval in the "fit" function.
    - Predictions are also used 
- A custom batchLoader class that helps with big datasets that can't stay on memory.
- Model saving and loading with checkpoints.

#### Things to fix or planning to add:
- Fix Multi-Task History
- Multi GPU support (i know, sucks i haven't added it yet, i just don't have two gpu's)
- Grid/Random Search functions
- Pre-defined metrics
- Check if the model save-load works correctly
- Export model to Tensorflow support
- Optimize the batchLoader more


If you have some features you'd like this wrapper to support, be sure to add Pull requests or Discussions here in the github page.
