#####
#
# This main loop is intended to be run inside of databricks
# ddp mode isn't supported in interactive mode so is run via %sh
# running via %sh requires that a terminal session is started and databricks cli configured
# running via %sh won't log the code against the notebook

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from typing import Type

# do tensorboard profiling
import torch.profiler
# log hardware stats
from pytorch_lightning.callbacks import DeviceStatsMonitor

# Adding mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# pytorch profiling
from pytorch_lightning.profiler import PyTorchProfiler
import os

# spark horovod
from sparkdl import HorovodRunner
import horovod.torch as hvd


def main_hvd(mlflow_db_host:str, mlflow_db_token:str, 
            data_module:Type[LightningDataModule], model:Type[LightningModule], 
            root_dir:str, epochs: int, run_name:str, experiment_id:int, *args, **kwargs):

    """
    
    In order to leverage horovod we need to wrap the main train function with a hvd.init
    
    Args:
        mlflow_db_host: the url for your workspace 
        mlflow_db_token: A valid access token for your workspace see: https://docs.databricks.com/dev-tools/api/latest/authentication.html
        data_module: pl LightningDataModule
        model: pl LightningModule
        root_dir: We need to set this to a /dbfs/ folder
        epochs:
        run_name: This name is used for MLFlow and also for the profiler logs
        experiment_id:
        args/kwargs: Args/Kwargs get fed into the pl.Trainer class
    
    """

    hvd.init()

    # mlflow workaround for python subprocess not receiving notebook token and workspace url
    mlflow.set_tracking_uri("databricks")
    os.environ['DATABRICKS_HOST'] = mlflow_db_host
    os.environ['DATABRICKS_TOKEN'] = mlflow_db_token

    return main_train(data_module=data_module, model=model, strat='horovod', num_gpus=1, node_id=hvd.rank(), 
                    root_dir=root_dir, epoch=epochs, run_name=run_name, experiment_id=experiment_id, *args, **kwargs)


def build_trainer(num_gpus:int, root_dir:str, epoch:int=3, strat:str='ddp', node_id:int=0,
                run_name:str=None, *args, **kwargs):
    
    """
    We split out the function that builders the Trainer object so that we can customise it if needed and change the train procedure

    Args:
        data_dir: data module to fit in
        model: model to train on
        num_gpus: number of gpus to train on
        root_dir: 
        epoch
        strat
        node_id: the number of the node
        run_name:
        experiment_id:
    
    """

    # set saving folders
    log_dir = os.path.join(root_dir, 'logs')

    # start mlflow
    ## manually trigger log models later as there seems to be a pickling area with logging the model
    if node_id==0:
        mlflow.pytorch.autolog(log_models=False)

    # Loggers
    loggers = []
    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(save_dir=log_dir, name=run_name,log_graph=True)

    loggers.append(tb_logger)

    # Callbacks
    # Device Stats Monitor is a bit hard to understand right now
    # April 2022 - looking to v1.7 to address
    callbacks = []
    device_stats = DeviceStatsMonitor()

    callbacks.append(device_stats)


    # Profilers - This is the key part to make sure that we can log perf stats and debug what is going wrong.
    # See: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

    profiler = PyTorchProfiler(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(log_dir,run_name), worker_name='worker'+str(node_id)),
        record_shapes=True,
        profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        with_stack=True)

    # main pytorch lightning trainer
    # we also feed all the args/kwargs in
    # See: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    trainer = pl.Trainer(
        max_epochs=epoch,
        log_every_n_steps=100,
        gpus=num_gpus,
        callbacks=callbacks,
        logger=loggers,
        strategy=strat,
        profiler=profiler,
        default_root_dir=root_dir, #otherwise pytorch lightning will write to local
        *args,
        **kwargs
        #profiler=profiler # for tensorboard profiler
    )


    return trainer


def main_train(data_module:Type[LightningDataModule], model:Type[LightningModule], 
                num_gpus:int, root_dir:str, epoch:int=3, strat:str='ddp', node_id:int=0,
                run_name:str=None, experiment_id:int=None, *args, **kwargs):

    """
    Main training Loop

    Args:
        data_dir: data module to fit in
        model: model to train on
        num_gpus: number of gpus to train on
        root_dir: 
        epoch:
        strat:
        node_id: the number of the node
        run_name:
        experiment_id:
        args/kwargs:
    
    """

    trainer = build_trainer(num_gpus, root_dir, epoch, strat, node_id,
                run_name, *args, **kwargs)

    
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    if node_id == 0:
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            
            mlflow.log_param("model", model.model_tag)

            trainer.fit(model, data_module)

            # log model
            mlflow.pytorch.log_model(model, "models")

        return trainer

    else:
        trainer.fit(model, data_module)

    

if __name__ == '__main__':

    #main_train(data_path, AVAIL_GPUS)
    # We can use %sh to run this as a script too
    # See: https://docs.databricks.com/notebooks/notebooks-use.html#mix-languages

    hr = HorovodRunner(np=2, driver_log_verbosity='all')

    model = hr.run(main_hvd())