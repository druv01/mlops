import kfp.dsl as dsl
from kubernetes import client as k8s_client

@dsl.pipeline(
    name='Model Training Pipeline',
    description='A pipeline for data preprocessing, model training, and model evaluation'
)
def model_training_pipeline():
    data_preprocessing = dsl.ContainerOp(
        name='data-preprocessing',
        image='trsanthosh/data-preprocessing',
        command=['python', '/real_estate.py'],
        pvolumes={'data-volume': k8s_client.V1Volume(volume_source=k8s_client.V1VolumeSource(host_path=k8s_client.V1HostPathVolumeSource(path='/data')))}
    )
    
    model_training = dsl.ContainerOp(
        name='model-training',
        image='trsanthosh/model-training',
        command=['python', '/real_estate.py'],
        pvolumes={'models-volume': k8s_client.V1Volume(volume_source=k8s_client.V1VolumeSource(host_path=k8s_client.V1HostPathVolumeSource(path='/models')))}
    )
    model_training.after(data_preprocessing)
    
    model_evaluation = dsl.ContainerOp(
        name='model-evaluation',
        image='trsanthosh/model-evaluation',
        command=['python', '/real_estate.py'],
        pvolumes={'models-volume': k8s_client.V1Volume(volume_source=k8s_client.V1VolumeSource(host_path=k8s_client.V1HostPathVolumeSource(path='/models')))}
    )
    model_evaluation.after(model_training)

# Create an experiment
experiment_name = 'real_estate'
client = kfp.Client()
experiment = client.create_experiment(name=experiment_name)

# Submit the pipeline for execution
pipeline_func = model_training_pipeline
pipeline_filename = 'model_training_pipeline.yaml'
client.upload_pipeline(pipeline_filename, pipeline_func=pipeline_func)
pipeline = client.create_run_from_pipeline_func(pipeline_func, experiment_name=experiment_name)
