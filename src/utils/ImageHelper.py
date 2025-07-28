
def preprocess_image(array):
    max_value = array.max()
    if max_value > 1:
        array = array / max_value
    return array