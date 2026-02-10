from neutron.core._tracer import Tracer
from neutron.core._module import Module, _check_for_static
from neutron.core._layers import ignore_list

from typing import Any, Callable
from pprint import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm

######################################
### Forwards and backwards handler ###
######################################

def autograd(model: Module, inputs, optimizer) -> None:
    """
    Autograd is the function that gets called when the used starts the training process.
    Handles both the Forwards and Backwards process.

    :param model: The custom model you want to train.
    :type model: Module
    :param inputs: The inputs for the model, can also be a tuple to divide in multiple inputs.
    :type inputs: np.ndarray|tuple(np.ndarray)
    """
    def forwards() -> None:
        """
        Simply calls the model class to proceed with the forwards process.
        Retrieves the final output as a Tracer object, which initializes the backwards process.
        """
        final_tracer: Tracer = model(*inputs)
        backwards(final_tracer, model)
        return final_tracer

    def backwards(final_tracer: Tracer, model):
        final_tracer.backwards()

        updates: dict = get_tree(model)

        # Optimizer here
        new_updates: dict = optimizer(updates)

        update(new_updates)

        return final_tracer
        
    final_tracer = forwards()
    return final_tracer

##########################
### Update the tracers ###
##########################

def update(new_updates: dict) -> None:
    """
    Updates all instances with specified values and gradients.

    :param new_updates: A dict containing instances as key and an inner dict with key/value for value and gradient.
    :type new_updates: dict
    """
    for instance in new_updates:
        instance.value      = new_updates[instance]["value"]
        instance.gradient   = new_updates[instance]["gradient"]
    
    return

#############################
### Prints the whole tree ###
#############################

def get_tree(model: Module, instances_only: bool = True) -> dict:
    """
    Just makes a big dict for all variables and their gradient and value.\n
    Returns a dict.

    :param model: The model class to get the updates from.
    :type model: Module
    """
    def extract_inside(model: Module):
        variables   : dict = vars(model)
        updates     : dict = {}

        for variable in variables:
            variable_instance = getattr(model, variable)
            
            # Extracts instance from Tracer
            if (
                _check_for_static(model, variable) == False and
                isinstance(variable_instance, Tracer)
            ):
                updates[variable] = {
                    "instance"  : variable_instance
                }
        
            # If it's a Module but not a layer, just extract the stuff inside.
            if (
                isinstance(variable_instance, Module) and
                type(variable_instance) not in ignore_list
            ):
                module_updates: dict    = extract_inside(variable_instance)
                updates[variable]       = module_updates
            
            # If it's a Module and a layer, give just the instance.
            elif (
                isinstance(variable_instance, Module) and
                type(variable_instance) in ignore_list
            ):
                updates[variable] = {
                    "instance"  : variable_instance
                }
        
        return updates
    
    def extract_instances(model: Module):
        updates: list = []
        variables   : dict = vars(model)
        for variable in variables:
            variable_instance = getattr(model, variable)

            # Extracts instance from Tracer
            if (
                _check_for_static(model, variable) == False and
                isinstance(variable_instance, Tracer)
            ):
                updates.append(variable_instance)

            # If it's a Module but not a layer, just extract the stuff inside.
            if (
                isinstance(variable_instance, Module) and
                type(variable_instance) not in ignore_list
            ):
                module_updates: list    = extract_instances(variable_instance)

                updates.extend(module_updates)

            # If it's a Module and a layer, give just the instance.
            elif (
                isinstance(variable_instance, Module) and
                type(variable_instance) in ignore_list
            ):
                updates.append(variable_instance)
        return updates

    updates_list = extract_inside(model) if instances_only == False else extract_instances(model)
    return updates_list

class Model:
    model: Module

    def __init__(
            self,
            model: Module,
            *model_args,
            seed: int|None = None,
            **model_kwargs
    ):
        if seed != None:
            np.random.seed(seed)
        try:
            self.model = model(*model_args, **model_kwargs)
        except Exception as e:
            raise e
        
    def compile(
            self,
            optimizer: Callable,
            loss: list|Callable,
            metrics: list,
            callbacks: list
    ):
        self.optimizer = optimizer
        
        def __define_loss() -> list:
            loss_list: list = []

            if not isinstance(loss, list):
                if isinstance(loss, Callable):
                    loss_list.append((loss, 1.0))
                elif isinstance(loss, tuple) and len(loss) == 1:
                    loss_list.append((loss[0], 1.0))
                else:
                    raise ValueError(
                        f"Argument \"loss\" should either be a list with inner tuples (loss_fn:Callable, weight:float), or just loss_fn:Callable. Found {type(loss)}"
                    )
            for loss_fn in loss:
                if not isinstance(loss_fn, tuple) and isinstance(loss_fn, Callable):
                    loss_list.append((loss_fn, 1.0))
                    continue

                elif not isinstance(loss_fn, tuple) and not isinstance(loss_fn, Callable):
                    raise ValueError(
                        f"Argument \"Loss\" should be a tuple (loss_fn, weight) or a single Callable object. Found: {type(loss_fn)}"
                    )
                
                if isinstance(loss_fn, tuple) and len(loss_fn) == 2:
                    if not isinstance(loss_fn[-1], float):
                        raise ValueError(
                            f"Values inside the loss tuple should be (Callable, float). Found: {type(loss_fn[-1])}"
                        )
                    if not isinstance(loss_fn[0], Callable):
                        raise ValueError(
                            f"Values inside the loss tuple should be (Callable, float). Found: {type(loss_fn[0])}"
                        )
                    
                    loss_list.append(loss_fn)
                
                elif isinstance(loss_fn, tuple) and len(loss_fn) == 1:
                    if isinstance(loss_fn[0], Callable):
                        continue
                    else:
                        raise ValueError(
                            f"Values inside a loss tuple should be (Callable, float) with len 2. Found: {type(loss_fn[0])}"
                        )
                else:
                    raise ValueError(
                        f"Values inside a loss tuple should be (Callable, float) with len 2. Found: {type(loss_fn[0])}"
                    )
            return loss_list

        def __define_metrics():
            metrics_final = []
            if isinstance(metrics, list):
                if len(metrics) == 0:
                    return
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
        
        loss_list = __define_loss()
        if len(loss_list) > 0:
            self.loss = loss_list
        else:
            self.loss = loss

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
                    if isinstance(item, pd.Series):
                        item = item.to_numpy()
                    if isinstance(item, pd.DataFrame):
                        item = item.to_numpy()
                    if isinstance(item, np.ndarray):
                        new_d.append(item)
                return tuple(new_d)
            else:
                if isinstance(d, pd.Series):
                    d = d.to_numpy()
                if isinstance(d, pd.DataFrame):
                    d = d.to_numpy()
                if isinstance(d, np.ndarray):
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
            steps = np.ceil(len(d[0]) // batch_size)
            return steps
        else:
            assert steps_for_epoch == np.ceil(len(d[0]) // batch_size), f"Argument \"steps_for_epoch\" has incompatible value for data len. Optimal steps should be: {np.ceil(len(d[0]) // batch_size)}"
        
        return steps_for_epoch
    
    def fit(
            self,
            data: None,
            x_train: tuple|np.ndarray|pd.DataFrame|None = None,
            y_train: tuple|np.ndarray|pd.DataFrame|None = None,
            data_eval: None = None,
            x_eval: tuple|np.ndarray|pd.DataFrame|None = None,
            y_eval: tuple|np.ndarray|pd.DataFrame|None = None,
            batch_size: int = 32,
            epochs: int = 1,
            steps_for_epoch: int|None = None,
            steps_for_eval: int|None = None,
            verbose: bool = True,
            starting_epoch: int = 1,
            starting_step_train: int = 1
    ):
        print("Starting up...")

        x_train, y_train = self.__check_data(x_train, y_train)

        if x_train == None and data == None:
            raise ValueError("At least one of x_train and data has to not be None.")
        
        if x_train != None and data != None:
            raise ValueError("The user is using both a generator and a dataset. Only one can be used at a time.")

        steps_for_epoch = self.__check_steps(x_train, steps_for_epoch, batch_size)

        def __grab_batch(start, *args, **kwargs):
            x_batch: list = []
            y_batch: list = []
            for item in x_train:
                x = item[start:start + batch_size]
                x = np.array(x)
                x_batch.append(x)
            for item in y_train:
                y = item[start:start + batch_size]
                y = np.array(y)
                y_batch.append(y)
            
            # This just returns the batched for each task
            return tuple(x_batch), tuple(y_batch)
        
        batch_function = __grab_batch

        for epoch in range(starting_epoch, epochs + 1):
            if verbose == True:
                print(f"Starting Epoch {epoch} / {epochs}")
                pbar = tqdm(total = int(steps_for_epoch), unit_scale = 1, desc = "...")
            
            start: int = 0

            epoch_loss: list = []
            
            for step in range(starting_step_train, steps_for_epoch + 1):
                try:
                    x_batch, y_batch = batch_function(start = start)
                except StopIteration:
                    print(f"Dataset exhausted at step {step} / {steps_for_epoch}")
                    pbar.close()
                    break

                x_train_batch: list = []
                for task in x_batch:
                    x_train_batch.append(np.stack(task))
                x_train_batch = tuple(x_train_batch)

                y_train_batch: list = []
                for task in y_batch:
                    y_train_batch.append(np.stack(task))
                y_train_batch = tuple(y_train_batch)

                start += batch_size

                result  = self.model(*x_train_batch)

                if not isinstance(result, tuple):
                    result = (result,)

                # If we have multiple losses, we just add them together, adding more to
                # the graph, still, making each loss affect each other.
                total_loss: Tracer = None

                for idx, (loss_fn, weight) in enumerate(self.loss):
                    loss_tracer = loss_fn(result[idx], y_train_batch[idx]) * weight

                    if total_loss == None:
                        total_loss = loss_tracer
                    else:
                        total_loss = total_loss + loss_tracer
                
                epoch_loss.append(np.mean(total_loss.value))

                total_loss.backwards()

                updates: dict = get_tree(self.model)

                # Optimizer here
                new_updates: dict = self.optimizer(updates)

                update(new_updates)

                tqdm_desc = f"Loss: {np.mean(epoch_loss):.4f}"

                if verbose == True:
                    pbar.set_description(desc = tqdm_desc)
                    pbar.update(1)
                
                if step == steps_for_epoch:
                    if verbose == True:
                        pbar.close()
            
            if epoch == epochs:
                return result