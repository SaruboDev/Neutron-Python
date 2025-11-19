import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jt
import equinox as eqx
import optax

import numpy as np
import pandas as pd
import math

from collections import defaultdict

from tabulate import tabulate
from typing import Callable
from tqdm import tqdm

from .utils import batchLoader, Callbacks, ModelState


class Model:
    model: eqx.Module
    key: jr.PRNGKey
    eval_key: jr.PRNGKey

    def __init__(self, model: eqx.Module, *model_args, seed: int|None = None, **model_kwargs):
        """
        Initializes the custom model the user passed in the "model" argument.
        Note: For now it does not suggest the custom model parameters, idk how.
        Args:
        - model: eqx.Module = The model the user wants to use.
        - *model_args = The model's arguments.
        - seed: int|None = The seed for initializing the user's model. If not given defaults to None.
        - **model_kwargs = Same as *model_args, but you can specify the name of the argument.
        """

        # To make a model work without the need for the user to specify its keys,
        # we can just make the general key like this, this key will be the one used as master.
        if seed != None:
            self.master_key = jr.PRNGKey(seed)
        else:
            seed = np.random.randint(2**64 - 1, dtype = np.uint64)
            self.master_key = jr.PRNGKey(seed)

        # Also, an evaluation key, which does not need to change.
        self.eval_key = jr.PRNGKey(42)

        try:
            self.model = model(*model_args, **model_kwargs, key = self.master_key)
        except Exception as e:
            raise e
    
    def compile(
            self,
            optimizer: Callable,
            loss: list|Callable,
            metrics: list,
            callbacks: list,
            gradAccSteps: int|None = None
    ):
        """
        Initializes the necessary arguments to make the model fit as intended. Similarly to TensorFlow, call "compile" before running the "fit" function.
        Args:
        - optimizer: Callable = The optimizer you want to use for the model training.
        - loss: list|Callable = A list of tuples containing loss_fn:Callable, weight:float as its elements. You can also write:
                [loss_fn, loss_fn2,...] : for multiple loss functions, when not specified the default weight of 1.0 it's applied.
                loss_fn : for a single loss function.
                Automatically converts everything in a [(loss_fn, weight), ] format.
        - metrics: list = A list containing all the callables for the metrics you want to monitor.
        - callbacks: list = A list containing all the callbacks you want the model to call during training.
        - gradAccSteps: int|None = The value to automatically use Gradient Accumulation for gradAccSteps k's.
                If you don't want to use Gradient Accumulation keep it set to None.
        """

        #################
        ### Optimizer ###
        #################

        # If the user wants the gradient accumulation, they should put an intager greater than 1.
        # If the user does not want the gradient accumulation, they should leave it as None.
        # if the user gave either 1 or 0, they get a warning, and the optmizer is set without gradient accumulation.
        if gradAccSteps != None:
            if gradAccSteps <= 1:
                print(f"Argument \"gradAccSteps\" cannot be 1 or 0, compiling without Gradient Accumulation.")
                self.optimizer = optimizer
            elif gradAccSteps > 1:
                self.optimizer = optax.MultiSteps(optimizer, every_k_schedule = gradAccSteps)
        else:
            self.optimizer = optimizer

        ############
        ### Loss ###
        ############

        # This just checks if the loss is in the correct format, you'll probably find redundant code here.
        def __define_loss():
            # The loss should be a list of losses, the list containing a tuple containing the loss function and its weight.
            # this is for compatibility for Multi Task models.
            loss_list = []
            # If the loss is a list, we iter through all the items inside
            if isinstance(loss, list):
                for loss_fn in loss:

                    # If the item is not a tuple, but it's just a Callable (expected) we just append it to the loss_list
                    # with the default weight of 1.0
                    if not isinstance(loss_fn, tuple) and isinstance(loss_fn, Callable):
                        loss_list.append((loss_fn, 1.0))
                        continue

                    # This just checks if it's neither a tuple nor a callable, and raises an error.
                    elif not isinstance(loss_fn, tuple) and not isinstance(loss_fn, Callable):
                        raise ValueError(
                            f"Argument \"loss\" should either be a tuple with (loss_fn:Callable, weight:int) or a single Callable loss_fn. Found: {type(loss_fn)}"
                        )
            
                    # If it's a tuple, and the len is 2, it's correct. But let's inspect the inner items.
                    if isinstance(loss_fn, tuple) and len(loss_fn) == 2:
                        # This is correct!
                        if isinstance(loss_fn[-1], float) and isinstance(loss_fn[0], Callable):
                            loss_list.append(loss_fn)
                            continue
                        
                        # These two are not correct.
                        elif not isinstance(loss_fn[-1], float):
                            raise ValueError(
                                f"Values inside the loss tuple should be (Callable, float), last item in tuple was {type(loss_fn[-1])}."
                            )
                        elif not isinstance(loss_fn[0], Callable):
                            raise ValueError(
                                f"Values inside the loss tuple should be (Callable, float), last item in tuple was {type(loss_fn[0])}."
                            )
                    elif isinstance(loss_fn, tuple) and len(loss_fn) == 1:
                        if isinstance(loss_fn[0], Callable):
                            loss_list.append((loss_fn[0], 1.0))
                            continue
                        else:
                            raise ValueError(
                                f"Values inside the loss tuple should be (Callable, float) and of length 2, found {type(loss_fn[0])}"
                            )
                    else:
                        raise ValueError(
                            f"Values inside the loss tuple should be (Callable, float) and of lenght 2."
                        )
            else:
                # If it's just a callable outside of the list, it works anyway, we just convert it into a tuple
                # with the default weight of 1.0
                if isinstance(loss, Callable):
                    loss_list.append((loss, 1.0))
                elif isinstance(loss, tuple) and len(loss) == 1:
                    loss_list.append((loss[0], 1.0))
                else:
                    raise ValueError(
                        f"Argument \"loss\" should either be a list with inner tuples (loss_fn:Callable, weight:float), or just loss_fn:Callable. Found {type(loss)}"
                    )
            return loss_list
        
        loss_list = __define_loss()
        if len(loss_list) > 0:
            self.loss = loss_list
        else:
            self.loss = loss

        ###############
        ### Metrics ###
        ###############

        # Now we're going to check for correct metrics format.
        # We'll have each metric for each task in case of MTL models, just for ease of work to get it up and going.
        def __define_metrics():
            metrics_final = []
            # Good!
            if isinstance(metrics, list):
                if len(metrics) != 0:
                    for item in metrics:
                        if isinstance(item, Callable):
                            metrics_final.append(item)
                        else:
                            raise ValueError(f"Metrics are supposed to be Callables!")
                else:
                    return []
            else:
                raise ValueError(f"Metrics is supposed to be a list of Callables!")
            return metrics_final
        
        metrics_list = __define_metrics()
        self.metrics = metrics_list

        self.callbacks = callbacks

    def summary(self):
        """
        Gives a summary of the initialized model.
        """
        def __memory(params):
            convert = {
                jnp.int8 : 1,
                jnp.int16 : 2,
                jnp.int32 : 4,
                jnp.int64 : 8,
                jnp.bfloat16 : 2,
                jnp.float16 : 2,
                jnp.float32 : 4,
                jnp.float64 : 8,
                jnp.bool_ : 1,
            }

            total = 0

            for leaf in jt.tree_leaves(params):
                if isinstance(leaf, jnp.ndarray):
                    bSize   = convert.get(leaf.dtype, 4)
                    total   += leaf.size * bSize
            return total

        table = []

        # Calculate every params
        params              = eqx.filter(self.model, eqx.is_array)
        trainable_params    = eqx.filter(self.model, eqx.is_inexact_array)
        param_count         = sum(x.size for x in jt.tree_leaves(params))
        trainable_count     = sum(x.size for x in jt.tree_leaves(trainable_params))

        total_size          = __memory(params)
        trainable_size     = __memory(trainable_params)
        nonTrainable_size   = total_size - trainable_size

        def __convert_size(size_bytes):
            if size_bytes == 0 or size_bytes == None:
                return "0B"
            size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
            i = int(math.floor(math.log(size_bytes, 1024)))
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)
            return "%s %s" % (s, size_name[i])

        # We get the layers names, parameters and shapes
        structure = {}
        leaves, _ = jax.tree.flatten_with_path(self.model)

        for i in leaves:
            structure[i[0][0].name] = []
        for i in leaves:
            structure[i[0][0].name].append({i[0][-1].name : ""})

        for layer in leaves:
            for idx, name in enumerate(structure[layer[0][0].name]):
                if layer[0][-1].name == list(name.keys())[0]:
                    structure[layer[0][0].name][idx][list(name.keys())[0]] = layer[-1]

        # We cycle through all the keys
        for key in structure.keys():
            # We cycle through the inner dict of the key
            for idx in range(0, len(structure[key])):
                x = list(structure[key][idx].values())[0]
                if isinstance(x, jnp.ndarray):
                    if idx == 0:
                        table.append(
                            [
                                key, # Name of layer
                                list(structure[key][idx].keys())[0], # Name of Parameter
                                x.shape, # Shapes
                                "{:,}".format(x.size) # Params
                            ]
                        )
                    else:
                        table.append(
                            [
                                " ", # Name of layer
                                list(structure[key][idx].keys())[0], # Name of Parameter
                                x.shape, # Shapes
                                "{:,}".format(x.size) # Params
                            ]
                        )
                else:
                    table.append(
                        [
                            key, # Name of layer
                            list(structure[key][idx].keys())[0], # Name of Parameter
                            "", # Shapes
                            x # Params
                        ]
                    )

        # We already have the model name
        print(f"Model: {self.model.__class__.__name__}")

        # We now print the table with the layer names, shapes and params.
        print(tabulate(table, headers = ["Layers", "Parameters", "Shapes", "Params"], tablefmt = "fancy_outline"))

        # We finally print the Total Parameters, Trainable Parameters and Non-Trainable Parameters
        print(f"Total Params: {"{:,}".format(param_count)} - {__convert_size(total_size)}")
        print(f"Trainable Params: {"{:,}".format(trainable_count)} - {__convert_size(trainable_size)}")
        print(f"Non-trainable Params: {"{:,}".format(param_count - trainable_count)} - {__convert_size(nonTrainable_size)}")

    def __check_callback(self):
        for callback in self.callbacks:
            if not isinstance(callback, Callbacks):
                raise TypeError("Callbacks elements have to be instances of the Callbacks class!")
        return True

    def __call_callback(self, function, epoch = None, *args, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, function)
            method(epoch = epoch, *args, **kwargs)

    def __check_data(self, x_train, y_train = None):
        def __convert(d):
            if d is None:
                return None
            # Not using a match case cuz i kinda need it progressively and i'm honestly too lazy to check if it works.
            if isinstance(d, tuple):
                new_d = []
                for item in d:
                    if isinstance(item, pd.Series):
                        item = item.to_numpy()
                    if isinstance(item, pd.DataFrame):
                        item = item.to_numpy()
                    if isinstance(item, np.ndarray):
                        item = jnp.array(item)
                    if isinstance(item, jnp.ndarray):
                        new_d.append(item)
                return tuple(new_d)
            else:
                if isinstance(d, pd.Series):
                    d = d.to_numpy()
                if isinstance(d, pd.DataFrame):
                    d = d.to_numpy()
                if isinstance(d, np.ndarray):
                    d = jnp.array(d)
                if isinstance(d, jnp.ndarray):
                    return (d,)
                else:
                    raise ValueError(f"Data has to be pandas Dataframe, jax/numpy array or a tuple containing these types! Found: {type(d)}")
        
        # We absolutely need all of these to be tuples containing either 1 element or n elements for n tasks.
        x_train = __convert(x_train)
        y_train = __convert(y_train)
        if y_train == None:
            y_train = x_train

        return x_train, y_train

    def __check_steps(self, d, steps_for_epoch, batch_size):
        if steps_for_epoch == None:
            # We already know that x_train and y_train are tuples after the conversion, so we just pick the first item.
            steps = jnp.ceil(len(d[0]) // batch_size)
            return steps
        else:
            assert steps_for_epoch == jnp.ceil(len(d[0]) // batch_size), f"Argument \"steps_for_epoch\" has incompatible value for data len. Optimal steps should be: {jnp.ceil(len(d[0]) // batch_size)}"
        
        return steps_for_epoch

    def __calculate_metrics(self, pred, y_train_batch):
        metrics_list = {}
        # For each element in the prediction, which is the n tasks, we calculate the metrics and add them to the metrics_list
        for idx, task in enumerate(pred):
            metrics_per_task: list = []
            for metric in self.metrics:
                metrics_per_task.append((f"{metric.__name__}", metric(task, y_train_batch[idx])))

            # We're going to show the loss per task only if there's more than one task, of course.
            if len(pred) > 1:
                # If there's just one loss we're going to use that loss on all tasks.
                if len(self.loss) == 1:
                    metrics_per_task.append((f"{self.loss[0][0].__name__}", self.loss[0][0](task, y_train_batch[idx])))

                # If there's less losses than predictions but more than 1 loss it will crash.
                if len(self.loss) < len(pred) and len(self.loss) != 1:
                    raise ValueError(f"loss functions have to be as many as the tasks of the model. Or just one to have the same loss applied to every task.")
                
                # If the losses are as many as the predictions, we can just use one each.
                if len(self.loss) == len(pred):
                    metrics_per_task.append((f"{self.loss[idx][0].__name__}", self.loss[idx][0](task, y_train_batch[idx])))
            
            metrics_list[f"Task_{int(idx) + 1}"] = metrics_per_task
        return metrics_list

    # @eqx.filter_jit
    def __compute_loss(
        self,
        model,
        x,
        y,
        key: jr.PRNGKey,
        is_inference: bool = False
    ):
        # I'm splitting the key just to make dropout layers and similar work better (or at least, it should work like that. idk)
        # then updating the keyTrain variable, so this always brings a new randomness to the dropout, but we still have the
        # master key to initialize the model for the checkpoints.
        # Then we divide the key again just to get the batched keys to make dropouts and similar work between batched data.

        if is_inference == False:
            # I gotta give a different key from the batch_size it creates in the splitKey.
            # The vmap already does stuff with the batch size of the data itself, though, i guess.

            new_keyTrain, splitKey = jr.split(key, 2)
            splitKey = jr.split(splitKey, x[0].shape[0])

            axs = (0, ) * len(x) + (0,)
            # (0,) * len(x) is for the tasks and the final (0,) is for the key

            pred = jax.vmap(model, in_axes = axs)(*x, splitKey)
        
        if is_inference == True:
            # Since it's for inference we don't really need to change the key for every piece of batch, but i'm doing it anyway
            # because it won't make a difference with the layers deactivated automatically with the tree_inference on. It's just
            # to avoid crashing.
            splitKey = jr.split(key, x[0].shape[0])
            axs = (0, ) * len(x) + (0,)
            pred = jax.vmap(model, in_axes = axs)(*x, splitKey)
        
        if not isinstance(pred, tuple):
            pred = (pred,)

        # So, for the loss calculations, we have a list of tuples, with a loss fn and the weight. This means
        # we're going to sum them all to one single variable and multiply the weight for the mean.
        # If there are multiple losses it's a weighted mean for one loss * tasks repeated for all losses together.
        losses: float = 0.0
        for idx, (loss_fn, weight) in enumerate(self.loss):
            l = loss_fn(pred[idx], y[idx])
            losses += weight * jnp.mean(l)

        if is_inference == False:
            return losses, (pred, y, new_keyTrain)
        if is_inference == True:
            # Yes, it's called train even if it's on evaluation mode.
            return losses, pred, y

    @eqx.filter_jit
    def __train_step(
            self,
            model,
            x,
            y,
            opt_state,
            key: jr.PRNGKey
    ):
        # In here we basically run the forward step, backward step, calculate loss and the gradients. Then apply the changes.
        (loss, (pred, y_train_batch, new_keyTrain)), grads = eqx.filter_value_and_grad(
            self.__compute_loss,
            has_aux = True
        )(model, x, y, key, is_inference = False)

        updates, opt_state = self.optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        
        return loss, model, opt_state, new_keyTrain, pred, y_train_batch
    
    @eqx.filter_jit
    def __eval_step(
        self,
        model,
        x,
        y,
        key: jr.PRNGKey
    ):
        inf_model = eqx.tree_inference(model, value = True)

        loss, pred, y_train_batch = self.__compute_loss(inf_model, x, y, key, is_inference = True)
        metrics = self.__calculate_metrics(pred, y_train_batch)
        return loss, metrics

    @eqx.filter_jit
    def __pred_step(
        self,
        model,
        x,
        key: jr.PRNGKey
    ):
        inf_model = eqx.tree_inference(model, value = True)

        x_train_batch = []
        for task in x:
            x_train_batch.append(jnp.stack(task))
        x_train_batch = tuple(x_train_batch)
        
        splitKey = jr.split(key, x_train_batch[0].shape[0])
        axs = (0, ) * len(x) + (0,)
        pred = jax.vmap(inf_model, in_axes = axs)(*x_train_batch, splitKey)
        if not isinstance(pred, tuple):
            pred = (pred, )

        return pred

    def fit(
            self,
            data: None,
            x_train: tuple|jnp.ndarray|np.ndarray|pd.DataFrame|None = None,
            y_train: tuple|jnp.ndarray|np.ndarray|pd.DataFrame|None = None,
            data_eval: None = None,
            x_eval: tuple|jnp.ndarray|np.ndarray|pd.DataFrame|None = None,
            y_eval: tuple|jnp.ndarray|np.ndarray|pd.DataFrame|None = None,
            batch_size: int = 32,
            epochs: int = 1,
            steps_for_epoch: int|None = None,
            steps_for_eval: int|None = None,
            verbose: bool = True,
            starting_epoch: int = 1,
            starting_step_train: int = 1
    ):
        """
        This is used to start the training procedures for the model, will return the history of the metrics for each epoch.
        Usually the loss for single-task models will be calculated as a mean from all the steps up to the current step.
        For multi-task losses it will be displayed as an overall mean and as a loss for each task specifically.

        You can also toggle if the general metrics (for both single-task and multi-task models) will be displayed as a mean for
        all the steps up to the current step in the epoch, or for just the singular step.
        """
        ####################
        ### Preparations ###
        ####################

        self.__check_callback()

        x_train, y_train = self.__check_data(x_train, y_train)

        if x_train == None and data == None:
            raise ValueError("At least one of x_train and data has to not be None.")
        
        if x_train != None and data != None:
            raise ValueError("The user is using both a generator and a dataset. Only one can be used at a time.")

        # If the user is not using the batch loader, we're going to calculate the steps automatically.
        if not isinstance(data, batchLoader):
            steps_for_epoch = self.__check_steps(x_train, steps_for_epoch, batch_size)

        def __grab_batch(start, *args, **kwargs):
            x_batch: list = []
            y_batch: list = []
            for item in x_train:
                x = item[start:start + batch_size]
                x = jnp.array(x)
                x_batch.append(x)
            for item in y_train:
                y = item[start:start + batch_size]
                y = jnp.array(y)
                y_batch.append(y)
            
            # This just returns the batched for each task
            return tuple(x_batch), tuple(y_batch)

        def __grab_next(*args, **kwargs):
            # This will return a tuple for x and y each.
            x, y = next(data_iter)
            return x, y

        batch_function = __grab_batch

        if isinstance(data, batchLoader):
            data_iter = iter(data)
            batch_function = __grab_next

        ###################
        ### Train Start ###
        ###################

        # Initialize optimizer with all the model's parameters.
        # (maybe to initialize on all is_inexact_arrays, which are the trainable params)
        opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

        history = dict()
        total_loss_list = []
        total_metrics_list = defaultdict(list)
        total_val_loss_list = []
        total_val_metrics_list = defaultdict(list)

        self.keyTrain, _ = jr.split(self.master_key)

        for epoch in range(starting_epoch, epochs + 1):
            if verbose == True:
                print(f"Starting epoch {epoch} / {epochs}")
                pbar = tqdm(total = int(steps_for_epoch), unit_scale = 1, desc = "...")

            epoch_loss: list = []
            epoch_metrics = defaultdict(list)

            if isinstance(data, batchLoader):
                data_iter = iter(data)
            start: int = 0

            # Callbacks call for the train and epoch start
            if epoch == 1:
                self.__call_callback("_on_train_start")
            self.__call_callback("_on_epoch_start", epoch)

            #############
            ### Steps ###
            #############

            for step in range(starting_step_train, steps_for_epoch + 1):
                try:
                    x_batch, y_batch = batch_function(start = start)
                except StopIteration:
                    print(f"Dataset exhausted at step {step} / {steps_for_epoch}")
                    pbar.close()
                    break

                # I gotta stack the data so that they can be used with the key
                x_train_batch = []
                for task in x_batch:
                    x_train_batch.append(jnp.stack(task))
                x_train_batch = tuple(x_train_batch)
                
                y_train_batch = []
                for task in y_batch:
                    y_train_batch.append(jnp.stack(task))
                y_train_batch = tuple(y_train_batch)

                start += batch_size

                ##########################
                ### Forward + Backward ###
                ##########################

                loss, self.model, opt_state, new_keyTrain, pred, y_metrics_batch = self.__train_step(
                    self.model,
                    x_train_batch,
                    y_train_batch,
                    opt_state,
                    self.keyTrain
                )
                self.keyTrain = new_keyTrain


                metrics = self.__calculate_metrics(pred, y_metrics_batch)

                ########################
                ### Show the Metrics ###
                ########################

                def __make_description():
                    for task in metrics:
                        for item in metrics[task]:
                            if len(metrics) > 1:
                                epoch_metrics[f"{task}_{item[0]}"].append(item[-1].item())
                            else:
                                epoch_metrics[f"{item[0]}"].append(item[-1].item())
                    
                    epoch_loss.append(float(loss.item()))
                    tqdm_desc = f"Loss: {np.mean(epoch_loss):.4f}"

                    dec = f" | "
                    for item in epoch_metrics:
                        dec += f"{item} : {np.mean(epoch_metrics[item]):.4f} "

                    tqdm_desc += dec
                    return tqdm_desc
                
                if verbose == True:
                    tqdm_desc = __make_description()
                    pbar.set_description(desc = tqdm_desc)
                    pbar.update(1)
                if step == steps_for_epoch:
                    if verbose == True:
                        pbar.close()

                    ##################
                    ### Evaluation ###
                    ##################

                    eval_loss, eval_metrics = self.__evaluation(
                        data_eval,
                        x_eval,
                        y_eval,
                        batch_size,
                        steps_for_eval,
                        verbose,
                        epoch
                    )

                    ###################
                    ### Epoch Ended ###
                    ###################

                    self.__call_callback(
                        "_on_epoch_end",
                        epoch,
                        loss = np.mean(eval_loss).item(),
                        metrics = eval_metrics,
                        masterKey = self.master_key,
                        trainKey  = self.keyTrain,
                        model = self.model,
                        opt_state = opt_state
                    )
            
            for i in epoch_metrics:
                total_metrics_list[i].append(np.mean(epoch_metrics[i]).item())
            for i in eval_metrics:
                total_val_metrics_list[i].append(np.mean(eval_metrics[i]).item())

            total_loss_list.append(np.mean(epoch_loss).item())
            total_val_loss_list.append(np.mean(eval_loss).item())            

            ###################
            ### Train Ended ###
            ###################

            if epoch == epochs:
                self.__call_callback("_on_train_end")
                
                history["loss"] = total_loss_list
                for i in total_metrics_list:
                    history[i] = total_metrics_list[i]

                history["val_loss"] = total_val_loss_list
                for i in total_val_metrics_list:
                    history[f"val_{i}"] = total_val_metrics_list[i]
                
                return history

    def __evaluation(
           self,
           data_eval,
           x_eval: tuple|jnp.ndarray|np.ndarray|pd.DataFrame|None,
           y_eval: tuple|jnp.ndarray|np.ndarray|pd.DataFrame|None,
           batch_size: int = 32,
           steps_for_eval: int = 0,
           verbose: bool = True,
           epoch: int = 0
    ):
        key = self.eval_key
        x_eval, y_eval = self.__check_data(x_eval, y_eval)

        if x_eval == None and data_eval == None:
            raise ValueError("At least one of x_train and data has to not be None.")
        
        if x_eval != None and data_eval != None:
            raise ValueError("The user is using both a generator and a dataset. Only one can be used at a time.")

        # If the user is not using the batch loader, we're going to calculate the steps automatically.
        if not isinstance(data_eval, batchLoader):
            steps_for_eval = self.__check_steps(x_eval, steps_for_eval, batch_size)

        self.__call_callback("_on_evaluation_start", epoch = epoch)
        
        ########################
        ### Data Preparation ###
        ########################

        def __grab_batch(start, *args, **kwargs):
            x_batch: list = []
            y_batch: list = []
            for item in x_eval:
                x = item[start:start + batch_size]
                x = jnp.array(x)
                x_batch.append(x)
            for item in y_eval:
                y = item[start:start + batch_size]
                y = jnp.array(y)
                y_batch.append(y)
            
            # This just returns the batched for each task
            return tuple(x_batch), tuple(y_batch)

        def __grab_next(*args, **kwargs):
            # This will return a tuple for x and y each.
            x, y = next(data_iter)
            return x, y

        batch_function = __grab_batch
        
        if isinstance(data_eval, batchLoader):
            data_iter = iter(data_eval)
            batch_function = __grab_next

        start: int = 0

        eval_losses: list = []
        eval_metrics_list = defaultdict(list)

        ###########################
        ### Starting Evaluation ###
        ###########################

        if verbose == True:
            print(f"Starting Evaluation...")
            ebar = tqdm(total = int(steps_for_eval), unit_scale = 1, desc = "...", colour = "green")
        for step in range(1, steps_for_eval + 1):
            key, subkey = jr.split(key, 2)
            try:
                x, y = batch_function(start = start)
            except StopIteration:
                print(f"Dataset exhausted at step {step}/ {steps_for_eval}")
                ebar.close()
                break

            ####################
            ### Eval Forward ###
            ####################

            eval_loss, eval_metrics = self.__eval_step(self.model, x, y, subkey)

            ###############
            ### Metrics ###
            ###############

            def __make_description():
                for task in eval_metrics:
                    for item in eval_metrics[task]:
                        if len(eval_metrics) > 1:
                            eval_metrics_list[f"{task}_{item[0]}"].append(item[-1].item())
                        else:
                            eval_metrics_list[f"{item[0]}"].append(item[-1].item())
                
                eval_losses.append(float(eval_loss.item()))
                tqdm_desc = f"Loss: {np.mean(eval_losses):.4f}"

                dec = f" | "
                for item in eval_metrics_list:
                    dec += f"{item} : {np.mean(eval_metrics_list[item]):.4f} "

                tqdm_desc += dec
                return tqdm_desc
            
            if verbose == True:
                tqdm_desc = __make_description()
                ebar.set_description(desc = tqdm_desc)
                ebar.update(1)
            
            ###########
            ### End ###
            ###########

            if not isinstance(x_eval, batchLoader):
                start += batch_size

            if step == steps_for_eval:
                self.__call_callback("_on_evaluation_end", epoch = epoch)
                ebar.close()
        
        return eval_losses, eval_metrics_list
    
    def predict(self, x, batch_size: int = 32, steps: int = None, verbose: bool = True):
        """
        Just like tensorflow's predict function. Given your x it will produce the predicted output.

        Args:
        - x: data to predict.
        - batch_size: can be None. If x has more than one data, this can be used the same as in the training procedure.
        - steps: can be None. Amount of steps just like the training procedure.
        - verbose: Show the progress bar.
        """
        
        # Getting evaluation key
        key = self.eval_key
        
        x = self.__check_data(x)

        steps_for_predict = self.__check_steps(x, steps, batch_size)

        ########################
        ### Data Preparation ###
        ########################

        def __grab_batch(start, *args, **kwargs):
            x_batch: list = []
            for item in x:
                x_ = item[start:start + batch_size]
                x_ = jnp.array(x_)
                x_batch.append(x_)
            
            # This just returns the batched for each task
            return tuple(x_batch)

        batch_function = __grab_batch

        start: int = 0
        res = None

        ################
        ### Starting ###
        ################

        if verbose == True:
            print(f"Predicting...")
            ebar = tqdm(total = int(steps_for_predict), unit_scale = 1, desc = "...", colour = "blue")

        for step in range(1, steps_for_predict + 1):
            # Split the key
            key, subKey = jr.split(key, 2)

            try:
                x = batch_function(start = start)
            except StopIteration:
                print(f"Dataset exhausted at step {step}/ {steps_for_predict}")
                ebar.close()
                break

            if not isinstance(x, batchLoader):
                start += batch_size
            
            ####################
            ### Pred Forward ###
            ####################

            # Technically it should already have all logits in order and divided by tasks
            logits = self.__pred_step(self.model, x, subKey)
            res = logits

            ###########
            ### End ###
            ###########

            ebar.update(1)
            if step == steps_for_predict:
                ebar.close()
        return res

    def load_model(self, model_path, load_weights_only: bool = False):
        if load_weights_only == False:
            
            template = ModelState(
                # It's a new one, but will get overwritten
                model = self.model,
                masterKey = jnp.zeros(2, dtype = jnp.uint32),
                trainKey = jnp.zeroes(2, dtype = jnp.uint32),
                optState = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
            )

            with open(model_path, "rb") as f:
                # Basically we need the same structure to give the weights and stuff back, so we initialize one
                # with random/starting values so it's constructed in the same way as the one the user is loading
                model_state = eqx.tree_deserialise_leaves(f, template)

                # Now we import back the loaded model states and hyperparameters
                self.model = model_state.model
                self.master_key = model_state.masterKey
                self.keyTrain = model_state.trainKey
                self.optimizerState = model_state.optState
        else:
            with open(model_path, "rb") as f:
                model_weights = eqx.tree_deserialise_leaves(f, self.model)
                self.model = model_weights


class stripped_model:
    model: eqx.Module
    key: jr.PRNGKey
    eval_key: jr.PRNGKey

    def __init__(self, model: eqx.Module, *model_args, seed: int|None = None, **model_kwargs):
        """
        Initializes the custom model the user passed in the "model" argument.
        Note: For now it does not suggest the custom model parameters, idk how.
        Args:
        - model: eqx.Module = The model the user wants to use.
        - *model_args = The model's arguments.
        - seed: int|None = The seed for initializing the user's model. If not given defaults to None.
        - **model_kwargs = Same as *model_args, but you can specify the name of the argument.
        """

        # To make a model work without the need for the user to specify its keys,
        # we can just make the general key like this, this key will be the one used as master.
        if seed != None:
            self.master_key = jr.PRNGKey(seed)
        else:
            seed = np.random.randint(2**64 - 1, dtype = np.uint64)
            self.master_key = jr.PRNGKey(seed)

        # Also, an evaluation key, which does not need to change.
        self.eval_key = jr.PRNGKey(42)

        try:
            self.model = model(*model_args, **model_kwargs, key = self.master_key)
        except Exception as e:
            raise e
    
    def compile(
            self,
            optimizer: Callable,
            loss: list|Callable,
            metrics: list,
            callbacks: list,
            gradAccSteps: int|None = None
    ):
        """
        Initializes the necessary arguments to make the model fit as intended. Similarly to TensorFlow, call "compile" before running the "fit" function.
        Args:
        - optimizer: Callable = The optimizer you want to use for the model training.
        - loss: list|Callable = A list of tuples containing loss_fn:Callable, weight:float as its elements. You can also write:
                [loss_fn, loss_fn2,...] : for multiple loss functions, when not specified the default weight of 1.0 it's applied.
                loss_fn : for a single loss function.
                Automatically converts everything in a [(loss_fn, weight), ] format.
        - metrics: list = A list containing all the callables for the metrics you want to monitor.
        - callbacks: list = A list containing all the callbacks you want the model to call during training.
        - gradAccSteps: int|None = The value to automatically use Gradient Accumulation for gradAccSteps k's.
                If you don't want to use Gradient Accumulation keep it set to None.
        """

        #################
        ### Optimizer ###
        #################

        # If the user wants the gradient accumulation, they should put an intager greater than 1.
        # If the user does not want the gradient accumulation, they should leave it as None.
        # if the user gave either 1 or 0, they get a warning, and the optmizer is set without gradient accumulation.
        if gradAccSteps != None:
            if gradAccSteps <= 1:
                print(f"Argument \"gradAccSteps\" cannot be 1 or 0, compiling without Gradient Accumulation.")
                self.optimizer = optimizer
            elif gradAccSteps > 1:
                self.optimizer = optax.MultiSteps(optimizer, every_k_schedule = gradAccSteps)
        else:
            self.optimizer = optimizer

        ############
        ### Loss ###
        ############

        # This just checks if the loss is in the correct format, you'll probably find redundant code here.
        def __define_loss():
            # The loss should be a list of losses, the list containing a tuple containing the loss function and its weight.
            # this is for compatibility for Multi Task models.
            loss_list = []
            # If the loss is a list, we iter through all the items inside
            if isinstance(loss, list):
                for loss_fn in loss:

                    # If the item is not a tuple, but it's just a Callable (expected) we just append it to the loss_list
                    # with the default weight of 1.0
                    if not isinstance(loss_fn, tuple) and isinstance(loss_fn, Callable):
                        loss_list.append((loss_fn, 1.0))
                        continue

                    # This just checks if it's neither a tuple nor a callable, and raises an error.
                    elif not isinstance(loss_fn, tuple) and not isinstance(loss_fn, Callable):
                        raise ValueError(
                            f"Argument \"loss\" should either be a tuple with (loss_fn:Callable, weight:int) or a single Callable loss_fn. Found: {type(loss_fn)}"
                        )
            
                    # If it's a tuple, and the len is 2, it's correct. But let's inspect the inner items.
                    if isinstance(loss_fn, tuple) and len(loss_fn) == 2:
                        # This is correct!
                        if isinstance(loss_fn[-1], float) and isinstance(loss_fn[0], Callable):
                            loss_list.append(loss_fn)
                            continue
                        
                        # These two are not correct.
                        elif not isinstance(loss_fn[-1], float):
                            raise ValueError(
                                f"Values inside the loss tuple should be (Callable, float), last item in tuple was {type(loss_fn[-1])}."
                            )
                        elif not isinstance(loss_fn[0], Callable):
                            raise ValueError(
                                f"Values inside the loss tuple should be (Callable, float), last item in tuple was {type(loss_fn[0])}."
                            )
                    elif isinstance(loss_fn, tuple) and len(loss_fn) == 1:
                        if isinstance(loss_fn[0], Callable):
                            loss_list.append((loss_fn[0], 1.0))
                            continue
                        else:
                            raise ValueError(
                                f"Values inside the loss tuple should be (Callable, float) and of length 2, found {type(loss_fn[0])}"
                            )
                    else:
                        raise ValueError(
                            f"Values inside the loss tuple should be (Callable, float) and of lenght 2."
                        )
            else:
                # If it's just a callable outside of the list, it works anyway, we just convert it into a tuple
                # with the default weight of 1.0
                if isinstance(loss, Callable):
                    loss_list.append((loss, 1.0))
                elif isinstance(loss, tuple) and len(loss) == 1:
                    loss_list.append((loss[0], 1.0))
                else:
                    raise ValueError(
                        f"Argument \"loss\" should either be a list with inner tuples (loss_fn:Callable, weight:float), or just loss_fn:Callable. Found {type(loss)}"
                    )
            return loss_list
        
        loss_list = __define_loss()
        if len(loss_list) > 0:
            self.loss = loss_list
        else:
            self.loss = loss

        ###############
        ### Metrics ###
        ###############

        # Now we're going to check for correct metrics format.
        # We'll have each metric for each task in case of MTL models, just for ease of work to get it up and going.
        def __define_metrics():
            metrics_final = []
            # Good!
            if isinstance(metrics, list):
                if len(metrics) != 0:
                    for item in metrics:
                        if isinstance(item, Callable):
                            metrics_final.append(item)
                        else:
                            raise ValueError(f"Metrics are supposed to be Callables!")
                else:
                    return []
            else:
                raise ValueError(f"Metrics is supposed to be a list of Callables!")
            return metrics_final
        
        metrics_list = __define_metrics()
        self.metrics = metrics_list

        self.callbacks = callbacks

    def __check_data(self, x_train, y_train = None):
        def __convert(d):
            if d is None:
                return None
            # Not using a match case cuz i kinda need it progressively and i'm honestly too lazy to check if it works.
            if isinstance(d, tuple):
                new_d = []
                for item in d:
                    if isinstance(item, pd.Series) or isinstance(item, pd.DataFrame) or isinstance(item, np.ndarray):
                        item = jnp.array(item)
                    if isinstance(item, jnp.ndarray):
                        new_d.append(item)
                return tuple(new_d)
            else:
                if isinstance(d, pd.Series) or isinstance(d, pd.DataFrame) or isinstance(d, np.ndarray):
                    d = jnp.array(d)
                else:
                    raise ValueError(f"Data has to be pandas Dataframe, jax/numpy array or a tuple containing these types! Found: {type(d)}")
                return (d, ) if not isinstance(d, tuple) else d
        # We absolutely need all of these to be tuples containing either 1 element or n elements for n tasks.
        x_train = __convert(x_train)
        y_train = __convert(y_train)
        if y_train == None:
            y_train = x_train

        return x_train, y_train

    def __check_steps(self, d, steps_for_epoch, batch_size):
        if steps_for_epoch == None:
            # We already know that x_train and y_train are tuples after the conversion, so we just pick the first item.
            steps = jnp.ceil(len(d[0]) // batch_size)
            return steps
        else:
            assert steps_for_epoch == jnp.ceil(len(d[0]) // batch_size), f"Argument \"steps_for_epoch\" has incompatible value for data len. Optimal steps should be: {jnp.ceil(len(d[0]) // batch_size)}"
        
        return steps_for_epoch

    @eqx.filter_jit
    def __compute_loss(
        self,
        model,
        x,
        y,
        key: jr.PRNGKey,
        is_inference: bool = False
    ):
        if is_inference == False:

            new_keyTrain, splitKey = jr.split(key, 2)
            splitKey = jr.split(splitKey, x[0].shape[0])
            axs = (0, ) * len(x) + (0,)

            pred = jax.vmap(model, in_axes = axs)(*x, splitKey)
        
        if not isinstance(pred, tuple):
            pred = (pred,)

        losses: float = 0.0
        for idx, (loss_fn, weight) in enumerate(self.loss):
            l = loss_fn(pred[idx], y[idx])
            losses += weight * jnp.mean(l)

        if is_inference == False:
            return jnp.sum(losses), (new_keyTrain)

    @eqx.filter_jit
    def __train_step(
            self,
            model,
            x,
            y,
            opt_state,
            key: jr.PRNGKey
    ):
        # In here we basically run the forward step, backward step, calculate loss and the gradients. Then apply the changes.
        (loss, (new_keyTrain)), grads = eqx.filter_value_and_grad(
            self.__compute_loss,
            has_aux = True
        )(model, x, y, key, is_inference = False)

        updates, opt_state = self.optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        
        return loss, model, opt_state, new_keyTrain

    def fit(
            self,
            x,
            y,
            batch_size: int = 32,
            epochs: int = 1,
            steps_for_epoch: int|None = None       
    ):
        x_train, y_train = self.__check_data(x, y)

        steps_for_epoch = self.__check_steps(x_train, steps_for_epoch, batch_size)

        def __grab_batch(start, *args, **kwargs):
            x_batch: list = []
            y_batch: list = []
            for item in x_train:
                x = item[start:start + batch_size]
                x = jnp.array(x)
                x_batch.append(x)
            for item in y_train:
                y = item[start:start + batch_size]
                y = jnp.array(y)
                y_batch.append(y)

            return tuple(x_batch), tuple(y_batch)

        batch_function = __grab_batch
        opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))

        self.keyTrain, _ = jr.split(self.master_key)

        for epoch in range(1, epochs + 1):
            print(f"Starting epoch {epoch} / {epochs}")
            pbar = tqdm(total = int(steps_for_epoch), unit_scale = 1, desc = "...")

            start:int = 0
            epoch_loss = []
            for step in range(1, steps_for_epoch + 1):
                x_batch, y_batch = batch_function(start = start)

                # I gotta stack the data so that they can be used with the key
                x_train_batch = []
                for task in x_batch:
                    x_train_batch.append(jnp.stack(task))
                x_train_batch = tuple(x_train_batch)
                
                y_train_batch = []
                for task in y_batch:
                    y_train_batch.append(jnp.stack(task))
                y_train_batch = tuple(y_train_batch)

                start += batch_size

                loss, self.model, opt_state, new_keyTrain = self.__train_step(
                    self.model,
                    x_train_batch,
                    y_train_batch,
                    opt_state,
                    self.keyTrain
                )
                self.keyTrain = new_keyTrain

                epoch_loss.append(float(loss.item()))
                tqdm_desc = f"Loss: {np.mean(epoch_loss):.4f}"
                pbar.set_description(desc = tqdm_desc)
                pbar.update(1)
                if step == steps_for_epoch:
                    pbar.close()
