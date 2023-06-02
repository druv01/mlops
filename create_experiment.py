import kfp
experiment_name = 'real_estate'
client = kfp.Client()
experiment = client.create_experiment(name=experiment_name)
