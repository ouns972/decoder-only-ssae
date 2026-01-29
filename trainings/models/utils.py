def import_model(model_name):
    if model_name == "model_trainable_inputs":
        from trainings.models.model_trainable_inputs import Decoder
    elif model_name == "model_avg_feature":
        from trainings.models.model_avg_feature import Decoder
    else:
        raise Exception("model name do not exist.")

    return Decoder
