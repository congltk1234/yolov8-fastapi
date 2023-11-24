from PIL import Image
import io
import pandas as pd
import numpy as np
import faiss 
from typing import Optional

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Initialize the models
detect_model = YOLO("./models/best_30epochs.pt")

# Load model for embedding
from transformers import AutoFeatureExtractor, AutoModel
from transformers import ViTImageProcessor, ViTModel

extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
embed_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
hidden_dim = embed_model.config.hidden_size

def extract_embeddings(image):
    image_pp = extractor(image, return_tensors="pt")
    features = embed_model(**image_pp).last_hidden_state[:, 0].detach().numpy()
    return features.squeeze()





def get_image_from_bytes(binary_image: bytes) -> Image:
    """Convert image from bytes to PIL RGB format
    
    Args:
        binary_image (bytes): The binary representation of the image
    
    Returns:
        PIL.Image: The image in PIL RGB format
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image


def get_bytes_from_image(image: Image) -> bytes:
    """
    Convert PIL image to Bytes
    
    Args:
    image (Image): A PIL image instance
    
    Returns:
    bytes : BytesIO object that contains the image in JPEG format with quality 85
    """
    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=85)  # save the image in JPEG format with quality 85
    return_image.seek(0)  # set the pointer to the beginning of the file
    return return_image

def transform_predict_to_df(results: list, labeles_dict: dict) -> pd.DataFrame:
    """
    Transform predict from yolov8 (torch.Tensor) to pandas DataFrame.

    Args:
        results (list): A list containing the predict output from yolov8 in the form of a torch.Tensor.
        labeles_dict (dict): A dictionary containing the labels names, where the keys are the class ids and the values are the label names.
        
    Returns:
        predict_bbox (pd.DataFrame): A DataFrame containing the bounding box coordinates, confidence scores and class labels.
    """
    # Transform the Tensor to numpy array
    predict_bbox = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax','ymax'])
    # Add the confidence of the prediction to the DataFrame
    predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
    # Add the class of the prediction to the DataFrame
    predict_bbox['class'] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
    # Replace the class number with the class name from the labeles_dict
    predict_bbox['name'] = predict_bbox["class"].replace(labeles_dict)
    return predict_bbox

def get_model_predict( input_image: Image) -> pd.DataFrame:
    """
    Get the predictions of a model on an input image.
    
    Args:
        model (YOLO): The trained YOLO model.
        input_image (Image): The image on which the model will make predictions.
        save (bool, optional): Whether to save the image with the predictions. Defaults to False.
        image_size (int, optional): The size of the image the model will receive. Defaults to 1248.
        conf (float, optional): The confidence threshold for the predictions. Defaults to 0.5.
        augment (bool, optional): Whether to apply data augmentation on the input image. Defaults to False.
    
    Returns:
        pd.DataFrame: A DataFrame containing the predictions.
    """
    # Make predictions
    predictions = detect_model.predict(
                        imgsz=640,
                        source=input_image,
                        conf=0.3,
                        save=False,
                        augment=True,
                        save_crop=False,
                        flipud= 0.0,
                        fliplr= 0.0,
                        mosaic = 0.0,
                        )
    
    # Transform predictions to pandas dataframe
    predictions = transform_predict_to_df(predictions, detect_model.model.names)
    return predictions


################################# BBOX Func #####################################

def add_bboxs_on_img(image: Image, predict: pd.DataFrame()) -> Image:
    """
    add a bounding box on the image

    Args:
    image (Image): input image
    predict (pd.DataFrame): predict from model

    Returns:
    Image: image whis bboxs
    """
    # Create an annotator object
    annotator = Annotator(np.array(image))
    # sort predict by xmin value
    predict = predict.sort_values(by=['xmin'], ascending=True)
    # iterate over the rows of predict dataframe
    for i, row in predict.iterrows():
        # create the text to be displayed on image
        text = f"{row['name']}: {int(row['confidence']*100)}%"
        # get the bounding box coordinates
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        # add the bounding box and text on the image
        annotator.box_label(bbox, text, color=colors(row['class'], True))
    # convert the annotated image to PIL image
    return Image.fromarray(annotator.result())


################################# Models #####################################

def canAddItem(existingArray, newType):
    bottoms = {'pants', 'shorts', 'skirt'}
    top = {'jacket', 'shirt'}
    newType = newType.lower()
    # Don't add the same item type twice
    if newType in existingArray:
        return False
    if newType == "shoe":
        return True
    # You can't wear both a top and a dress
    if newType in top and (len(top.intersection(existingArray)) or "dress" in existingArray):
        return False    
    if newType == 'dress' and (len(top.intersection(existingArray)) or len(bottoms.intersection(existingArray))):
        return False
    # Only add one type of bottom (pants, skirt, etc)
    if newType in bottoms and (len(bottoms.intersection(existingArray)) or "dress" in existingArray):
        return False

    return True


def get_outfit_from_detect(df)->pd.DataFrame():
    outfit =[]
    addedTypes = []

    df = df.sort_values(by=['confidence'], ascending=False)
    for item in df.values:
        itemType = item[-1] # i.e. shorts, top, etc
        if canAddItem(addedTypes, itemType):
            addedTypes.append(itemType)
            outfit.append(item)
    df = df.head(0)

    for i in outfit:
        df.loc[len(df)] = i
     
    return df




#################
def get_similar_item(crop_image, label,k):
    new_embed = extract_embeddings(crop_image)
    new_embed = np.expand_dims(new_embed, axis=0)
    index_path = './assets/faiss/'+label+'.bin'
    load_jacket_embed = faiss.read_index(index_path)
    f_distances, f_ids = load_jacket_embed.search(new_embed, k=k)  # search k product
    return f_ids[0],f_distances[0]


def make_recommend_outfit_list(prediction_df):
    outfit = get_outfit_from_detect(prediction_df)
    
    pass