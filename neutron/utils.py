import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
import pandas as pd

from typing import Callable, Any


class batchLoader:
    def __init__(
           self,
           data,
           generator: Callable,
           batch_size: int,
           outputs: tuple,
           infinite: bool = False,
           pad_x_data: tuple = (False,),
           pad_y_data: tuple = (False,),
           pad_value: int = 0
    ):
        """
        The idea is that we get the data and just pass it to the function generator.
        The data is loaded in batches and processed in the generator, giving back the batched size.
        The only thing is that i guess something is wrong cuz the memory slowly increases in usage, so i gotta find something for that.
        """
        self.data       = data
        self.gen_fn     = generator
        self.batch_size = batch_size
        self.xDtype     = outputs[0]
        self.yDtype     = outputs[1]
        self.infinite   = infinite

        if not isinstance(pad_x_data, tuple):
            pad_x_data = (pad_x_data, )
        if not isinstance(pad_y_data, tuple):
            pad_y_data = (pad_y_data, )
        self.pad_x_data = pad_x_data
        self.pad_y_data = pad_y_data
        self.pad_value  = pad_value

        if not isinstance(self.xDtype, tuple):
            self.xDtype = (self.xDtype, )
        if not isinstance(self.yDtype, tuple):
            self.yDtype = (self.yDtype, )
    
    def __iter__(self):
        self.generator = self.gen_fn()
        
        return self
    
    def __next__(self):
        n: int = 0
        # A list of lists for each task
        x_batch: list = [[] for _ in self.xDtype]
        y_batch: list = [[] for _ in self.yDtype]

        while n < self.batch_size:
            # For now it probably supports only Single-Task data. Gotta check for MTL
            try:
                # This COULD give a tuple for (x * n tasks) (y * n tasks)
                x, y = next(self.generator)

                # Just in case the x and y are single task items, we put them in a tuple.
                if not isinstance(x, tuple):
                    x = (x,)
                if not isinstance(y, tuple):
                    y = (y,)
                
                # Now we can append for each task the correct item
                for idx, item in enumerate(x):
                    x_batch[idx].append(item)
                
                for idx, item in enumerate(y):
                    y_batch[idx].append(item)
                n += 1
            except StopIteration:
                if self.infinite == True:
                    self.generator = self.gen_fn()
                    # Gives only the remaining ones even if the batch size is not complete
                    break
                elif self.infinite == False:
                    break

        x_pad = [[] for _ in self.xDtype]
        y_pad = [[] for _ in self.yDtype]

        for idx, item in enumerate(x_batch):
            # We first check if we want to pad that data
            if self.pad_x_data[idx] == True:
                # We get the padding lenght as the max of the lenght for that specific task.
                pad_len = max(len(i) for i in x_batch[idx])
                for array in item:
                    while len(array) < pad_len:
                        array = np.append(array, self.pad_value)
                    x_pad[idx].append(array)
            else:
                for array in item:
                    x_pad[idx].append(array)

        for idx, item in enumerate(y_batch):
            if self.pad_y_data[idx] == True:
                pad_len = max(len(i) for i in y_batch[idx])
                for array in item:
                    while len(array) < pad_len:
                        array = np.append(array, self.pad_value)
                    y_pad[idx].append(array)
            else:
                for array in item:
                    y_pad[idx].append(array)

        # if self.pad_data == True:
        #     # We're just padding for tasks, not across all tasks.
        #     for idx, item in enumerate(x_batch):
        #         # We get the padding lenght as the max of the lenght for that specific task.
        #         pad_len = max(len(i) for i in x_batch[idx])
        #         for array in item:
        #             while len(array) < pad_len:
        #                 array = np.append(array, self.pad_value)
        #             x_pad[idx].append(array)

        #     for idx, item in enumerate(y_batch):
        #         # We get the padding lenght as the max of the lenght for that specific task.
        #         pad_len = max(len(i) for i in y_batch[idx])
        #         for array in item:
        #             while len(array) < pad_len:
        #                 array = np.append(array, self.pad_value)
        #             y_pad[idx].append(array)

        # Now x/y pad are a tuple of tuples, one inner tuple for each task.

        # Each dtype has to specifically be written per-task.
        # So if there are like, 2 tasks "outputs" will be ((dtype1, dtype2), (dtype1, dtype2)).
        x_final = [[] for _ in self.yDtype]
        y_final = [[] for _ in self.yDtype]
        
        # Now we apply the dtypes and convert the numpy arrays to jax arrays.
        for idx, item in enumerate(x_pad):
            for array in item:
                array = array.astype(self.xDtype[idx])
                array = jnp.array(array)
                x_final[idx].append(array)
        
        for idx, item in enumerate(y_pad):
            for array in item:
                array = array.astype(self.yDtype[idx])
                array = jnp.array(array)
                y_final[idx].append(array)
        
        x_conv = []
        y_conv = []
        for task in x_final:
            task = tuple(task)
            x_conv.append(task)
        for task in y_final:
            task = tuple(task)
            y_conv.append(task)
        return tuple(x_conv), tuple(y_conv)
        # print(tuple(x_conv), "\n")    

        # return tuple(x_final), tuple(y_final)

########################
### CUSTOM CALLBACKS ###
########################

class Callbacks:
    def __init__(self, *args, **kwargs):
        pass

    def _on_train_start(self, *args, **kwargs):
        pass

    def _on_train_end(self, *args, **kwargs):
        pass

    def _on_epoch_start(self,epoch: int, *args, **kwargs):
        pass
    
    def _on_epoch_end(self, epoch: int, *args, **kwargs):
        pass

    def _on_evaluation_start(self, epoch: int, *args, **kwargs):
        pass
    
    def _on_evaluation_end(self, epoch: int, *args, **kwargs):
        pass

# Template to save and load the model
class ModelState(eqx.Module):
    model: eqx.Module
    masterKey: jax.Array
    trainKey: jax.Array
    optState: Any

##################
### CHECKPOINT ###
##################

class checkpoint(Callbacks):
    """
    A callback to give a checkpointing function to your model.
    Will save a model either every epoch, or every n epoch, always includes the first one.

    Args:
    - filepath [str]: a string that contains the filepath, concluding with just the model name, no suffixes.
    - monitor [str]: only checks for evaluation metrics, sorry not sorry. Can be "loss" or the name of whichever metric you need.\nIf you're using a multi-task model, you might want to specify which task you want to monitor, for example: 'Task_0_accuracy', tasks are always numbered from 0 onwards. It's the value you want to check for checkpointing.
    - verbose [int]: either 1 or 0, only prints errors or if the model saved successfully, recommended to turn it on with 1.
    - save_best_only [bool]: a bool that will make the checkpointing save only if the current epoch did better than the last one.
    - save_weights_only [bool]: a bool that will make the checkpointing save only the weights of the model.
    - save_freq [int|str]: the frequency to save the model. Either "epoch" to save it each epoch, or an int for how many epochs you want before saving, the first one always saves.
    - threshold [float|None]: the threshold you want the model to surpass to save. If you have the save best only you have to put a number here too.
    - mode [str]: either "min" or "max". A string that says how you want to check for the best model, some metrics need to be higher than the previous to be considered "better", others need to be lower, this is for that.
    """
    def __init__(
            self,
            filepath: str,
            monitor: str = "val_loss",
            verbose: int = 0,
            save_best_only: bool = True,
            save_weights_only: bool = True,
            save_freq: int|str = "epoch",
            threshold: float|None = None,
            mode: str = "min",
            *args,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.filepath           = filepath
        self.monitor            = monitor
        self.verbose            = verbose
        self.save_best_only     = save_best_only
        self.save_weights_only  = save_weights_only
        self.save_freq          = save_freq
        self.threshold          = threshold
        self.mode               = mode

        self.prev_save_epoch    = None
        self.prev_value: float  = None
        self.old_val            = None

        if save_best_only == True and threshold == None:
            raise Exception("Having \"threshold\" to None and \"save_best_only\" to True won't work!")

    def save_model(self, masterKey, model, opt_state, trainKey):
        if self.save_weights_only == True:
            try:
                with open(self.filepath + str(self.epoch) + ".eqx", "wb") as f:
                    eqx.tree_serialise_leaves(f, model)
                if self.verbose != 0:
                    print(f"Model Saved with {self.monitor} : {self.prev_value}")
            except Exception as e:
                print("Could not save model! Skipping...")
                print(e)
                return
        
        elif self.save_weights_only == False:
            # With this i should be saving both the master key and the model weights
            state = ModelState(model = model, masterKey = masterKey, trainKey = trainKey, optState = opt_state)
            try:
                with open(self.filepath + str(self.epoch) + ".eqx", "wb") as f:
                    eqx.tree_serialise_leaves(f, state)
                
                if self.verbose != 0:
                    print(f"Model Saved with {self.monitor} : {self.prev_value}")
            
            except Exception as e:
                print("Could not save model! Skipping...")
                print(e)
                return

    def check_monitor(self, loss, metrics):
        if self.monitor == "loss":
            return "loss", loss
        if self.monitor in metrics:
            return self.monitor, np.mean(metrics[self.monitor])
        raise ValueError(f"Target \"{self.monitor}\" not found!")

    def check_for_best(self, value, masterKey, model, opt_state, trainKey):
        if self.save_best_only == True and self.threshold != None:
            # We get the threshold to surpass
            # If the old val is not None and the current val surpasses the threshold we save the model
            if self.old_val != None:
                if self.mode == "min":
                    if value < self.old_val - self.threshold:
                        self.old_val = value
                        self.save_model(masterKey, model, opt_state, trainKey)
                elif self.mode == "max":
                    if value > self.old_val + self.threshold:
                        self.old_val = value
                        self.save_model(masterKey, model, opt_state, trainKey)
                else:
                    if self.verbose != 0:
                        print("Won't save model, threshold not surpassed.")
                    return

            # If there was no old_val we just save it
            else:
                self.old_val = value
                self.save_model(masterKey, model, opt_state, trainKey)
        
        # If the user doesn't care about saving only the best epochs, we just save it
        else:
            self.save_model(masterKey, model, opt_state, trainKey)

    def _on_epoch_end(self, epoch, loss, metrics, masterKey: jax.random.PRNGKey, trainKey: jax.random.PRNGKey, model, opt_state, *args, **kwargs):
        self.epoch = epoch
        # First epoch end, the first epoch is ALWAYS saved
        if self.prev_save_epoch == None:
            self.prev_save_epoch = epoch
            name, value = self.check_monitor(loss, metrics)
            self.prev_value = value
            if self.threshold != None:
                self.old_val = value
            self.save_model(masterKey, model, opt_state, trainKey)
        else:
            # If the save frequency is numeric
            if type(self.save_freq) == int:
                # If the epoch frequency is right
                if epoch % self.save_freq == 0:                    
                    name, value = self.check_monitor(loss, metrics)

                    if self.save_best_only == True:
                        self.check_for_best(value, masterKey, model, opt_state, trainKey)
                    else:
                        self.save_model(masterKey, model, opt_state, trainKey)
                    # We save the current epoch
                    self.prev_save_epoch = epoch
                # Simply not the correct epoch to save
                else:
                    return
            
            # Every epoch
            if self.save_freq == "epoch":
                self.prev_save_epoch = epoch
                name, value = self.check_monitor(loss, metrics)

                if self.save_best_only == True:
                    self.check_for_best(value, masterKey, model, opt_state, trainKey)
                else:
                    self.save_model(masterKey, model, opt_state, trainKey)
                # We save the current epoch
                self.prev_save_epoch = epoch

