from argotech.serving.container import model_manager, feature_store


def get_model_manager():
    return model_manager


def get_feature_store():
    return feature_store
