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
        D_in, H, D_out = 2, 10, 1  # input dimension, hidden dimension, output dimension
        input_1 = layers.Input(shape=(1,),  name='ClockShift_Lag1')  # layer name must match time series name
        input_2 = layers.Input(shape=(1,), name='ClockShift_Lag2')  # layer name must match time series name
        merged = layers.Concatenate(axis=1)([input_1, input_2])
        hidden = layers.Dense(H,input_dim=D_in, activation='tanh', name='hidden')(merged)
        output = layers.Dense(D_out, activation='linear', name='target')(hidden)  # layer name must match time series name
        model = Model(inputs=[input_1, input_2], outputs=output)
        return model

    def get_torch_model():
        # D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        D_in, H, D_out = 2, 10, 1

        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
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

    keras_wrapper((ClockShift_Lag1=shift_power_statistics,
         ClockShift_Lag2=shift_power_statistics2,
         target=scale_power_statistics))

    torch_wrapper(ClockShift_Lag1=shift_power_statistics,
         ClockShift_Lag2=shift_power_statistics2,
         target=scale_power_statistics)

Neural Network Module
---------------------
As an alternative to the usage of the Wrapers you can implement a new module, which contains a neural network.
Therefore, you need a model definition and a reference to a defined model in your module. Afterwards, you should
implement the *transform*, *fit*, *get_param*, and *set_param* method.

The *transform* method should call the predict function of the model. The *fit* method has to train the model. The
*get_param* and the *set_param* method should implement the functionality to storing and loading the module.
