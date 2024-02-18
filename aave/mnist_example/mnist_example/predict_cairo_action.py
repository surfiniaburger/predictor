import numpy as np
import torch
import torch.nn.functional as F
from giza_actions.action import Action, action
from giza_actions.model import GizaModel
from giza_actions.task import task
from PIL import Image

#from mnist_example.predict_onnx_action import preprocess_image

MODEL_ID = 306  # Update with your model ID
VERSION_ID = 1  # Update with your version ID





@task(name="Preprocess Image")
def preprocess_image(image_path):
    # Load image, convert to grayscale, resize and normalize
    image = Image.open(image_path).convert("L")
    # Resize to match the input size of the model
    image = image.resize((14, 14))
    image = np.array(image).astype("float32") / 255
    image = image.reshape(1, 196)  # Reshape to (1, 196) for model input
    return image

@task(name='Prediction with Cairo')
def prediction(image, model_id, version_id):
    model = GizaModel(id=model_id, version=version_id)

    try:
        print("Input image shape:", image.shape)  # Check the shape of the input image
        result, request_id = model.predict(
            input_feed={"image": image}, verifiable=True, output_dtype="Tensor<FP16x16>"
        )

        print("Result from model.predict:", result)
        print("Request ID from model.predict:", request_id)

        if result is None:
            raise ValueError("Model prediction returned None.")

        # Convert result to a PyTorch tensor
        probabilities = torch.tensor(result)
        # Use argmax to get the predicted class
        predicted_class = torch.argmax(probabilities, dim=1)

        return predicted_class.item(), request_id
    except Exception as e:
        # If an error occurs during prediction, return None
        print(f"An error occurred during prediction: {e}")
        return None, None


@action(name="Execution: Prediction with Cairo", log_prints=True)
def execution():
    image = preprocess_image("./zero.jpg")
    (result, request_id) = prediction(image, MODEL_ID, VERSION_ID)
    print(f"Predicted Digit: {result}")
    print(f"Request ID: {request_id}")

    return result, request_id


if __name__ == "__main__":
    action_deploy = Action(entrypoint=execution, name="pytorch-mnist-cairo-action")
    action_deploy.serve(name="pytorch-mnist-cairo-deployment")
