Neural Networks and pyWATTS
===========================

In the following, we will examine how we can use neural networks within pyWATTS. In the following, we will learn

* how to use the Keras and pyTorch Wrappers, and
* how to implement a Keras based network in a own module.

Usage of Keras an pyTorch Wrappers
----------------------------------

One possibility to integrate neural networks into pyWATTS is to use the provided wrappers for Keras and pyTorch.
In general, both are used in the same way. Examples of the usage of both wrappers are in the examples folder.

1. You need a Keras or torch model. Note, that for Keras you should use the functional API. Moreover, your input and
   output layers should be named.

.. code-block:: python

    def get_keras_model():
        # write the model with the Functional API, Sequential does not support multiple input tensors
        input_1 = layers.Input(shape=(24,), name='lag_features')  # layer name must match time series name
        hidden = layers.Dense(10, activation='tanh', name='hidden')(input_1)
        output = layers.Dense(24, activation='linear', name='target')(hidden)  # layer name must match time series name
        model = Model(inputs=[input_1], outputs=output)
        return model


    def get_torch_model():
        model = torch.nn.Sequential(
            torch.nn.Linear(24, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 24),
        )
    return model

2. You initialize the Keras or pyTorch Wrapper with the predefined neural networks. Additionaly, you can add the compile,
   and fit keyword-arguments of the model as Dict.

.. code-block:: python

    keras_wrapper = KerasWrapper(keras_model,
                                 fit_kwargs={"batch_size": 8, "epochs": 1},
                                 compile_kwargs={"loss": "mse", "optimizer": "Adam", "metrics": ["mse"]})

    torch_wrapper = PyTorchWrapper(get_sequential_model(),
                                     fit_kwargs={"batch_size": 8, "epochs": 1},
                                     compile_kwargs={"loss": "mse", "optimizer": "Adam", "metrics": ["mse"]})

3. You have to add the wrapper to the pipeline just as a normal module. Note, if you use keras you have to assign the
   inputs to the keywords with the same name as you have them defined in the previously defined keras neural network.

.. code-block:: python

    keras_wrapper(lag_features=lag_features, target=target)

    torch_wrapper(lag_features=lag_features, target=target)

Neural Network Module
---------------------
As an alternative to the usage of the Wrapers you can implement a new module, which contains a neural network.
Therefore, you need a model definition and a reference to a defined model in your module. Afterwards, you should
implement the *transform*, *fit*, *get_param*, and *set_param* method.

The *transform* method should call the predict function of the model. The *fit* method has to train the model. The
*get_param* and the *set_param* method should implement the functionality to storing and loading the module.

Additionally, you have also to define  your own save and load methods, which should contain all functionality for
storing and loading your neural network.

As an example you can look at the profile_neural_network module. It implements a neural network using Keras without
using the KerasWrapper.
