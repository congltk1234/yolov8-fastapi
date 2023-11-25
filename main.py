####################################### IMPORT #################################
import json
import pandas as pd
from PIL import Image
from loguru import logger
import sys

from fastapi import FastAPI, File, status, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException

from io import BytesIO

from app import *

####################################### logger #################################

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level=10,
)
logger.add("log.log", rotation="1 MB", level="DEBUG", compression="zip")

###################### FastAPI Setup #############################

# title
app = FastAPI(
    title="Object Detection FastAPI Template",
    description="""Obtain object value out of image
                    and return image and json result""",
    version="2023.1.31",
)

# This function is needed if you want to allow client requests 
# from specific domains (specified in the origins argument) 
# to access resources from the FastAPI server, 
# and the client and server are hosted on different domains.
origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def save_openapi_json():
    '''This function is used to save the OpenAPI documentation 
    data of the FastAPI application to a JSON file. 
    The purpose of saving the OpenAPI documentation data is to have 
    a permanent and offline record of the API specification, 
    which can be used for documentation purposes or 
    to generate client libraries. It is not necessarily needed, 
    but can be helpful in certain scenarios.'''
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)

# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get('/healthcheck', status_code=status.HTTP_200_OK)
def perform_healthcheck():
    '''
    It basically sends a GET request to the route & hopes to get a "200"
    response code. Failing to return a 200 response code just enables
    the GitHub Actions to rollback to the last version the project was
    found in a "working condition". It acts as a last line of defense in
    case something goes south.
    Additionally, it also returns a JSON response in the form of:
    {
        'healtcheck': 'Everything OK!'
    }
    '''
    return {'healthcheck': 'Everything OK!'}


######################### Support Func #################################




######################### MAIN Func #################################
def image_grid(imgs, rows, cols):
            w,h = imgs[1].size
            grid = Image.new('RGB', size=(cols*w, rows*h))
            for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
            return grid

def expand2square(pil_img, background_color=(0, 0, 0)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

        
@app.post("/img_object_detection_outfit")
def img_object_detection_outfit(file: bytes = File(...)):
    """
    Object Detection from an image.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        dict: JSON format containing the Objects Detections.
    """
    # Step 2: Convert the image file to an image object
    input_image = get_image_from_bytes(file)
    # Step 3: Predict from model
    predict = get_model_predict(input_image)

    outfit = get_outfit_from_detect(predict)

    for i in range(outfit.shape[0]):
        crop_bbox = outfit[['xmin', 'ymin', 'xmax','ymax']].iloc[i].values
        img_crop = input_image.crop(crop_bbox)   # Crop
        label=outfit['name'].iloc[i]
        f_ids, f_distances = get_similar_item(img_crop, label=label,k=3)
        
        name_file = './assets/csv/'+label+'.csv'
        articles = pd.read_csv(name_file)
        id_articles = []
        retrieved_examples = []
        for i in f_ids:
            name_file = './assets/img/'+articles['path'][i]
            id_articles.append(articles['article_id'][i])
            list_image = Image.open(name_file)
            retrieved_examples.append(list_image)
            output_list = [expand2square(img_crop)]
            output_list.extend(retrieved_examples)
            recommend = image_grid(output_list, 1, len(output_list))
        
        
        print(label,': ',id_articles)
        recommend.show()    
    # print(outfit)
    result=json.dumps(f_ids.tolist())
    # objects = outfit['name'].values
    # result['detect_objects_names'] = ', '.join(objects)
    # result['detect_objects'] = json.loads(outfit.to_json(orient='records'))
    # Step 5: Logs and return
    logger.info("results: {}", result)
    return result


@app.post("/img_object_detection_to_json")
def img_object_detection_to_json(file: bytes = File(...)):
    """
    Object Detection from an image.

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        dict: JSON format containing the Objects Detections.
    """
    # Step 1: Initialize the result dictionary with None values
    result={'detect_objects': None}
    # Step 2: Convert the image file to an image object
    input_image = get_image_from_bytes(file)
    # Step 3: Predict from model
    predict = get_model_predict(input_image)
    # Step 4: Select detect obj return info
    # here you can choose what data to send to the result
    detect_res = predict
    objects = detect_res['name'].values
    result['detect_objects_names'] = ', '.join(objects)
    result['detect_objects'] = json.loads(detect_res.to_json(orient='records'))
    # Step 5: Logs and return
    logger.info("results: {}", result)
    return result


@app.post("/img_object_detection_to_img")
def img_object_detection_to_img(file: bytes = File(...)):
    """
    Object Detection from an image plot bbox on image

    Args:
        file (bytes): The image file in bytes format.
    Returns:
        Image: Image in bytes with bbox annotations.
    """
    # get image from bytes
    input_image = get_image_from_bytes(file)
    # model predict
    predict = get_model_predict(input_image)
    # add bbox on image
    final_image = add_bboxs_on_img(image = input_image, predict = predict)
    # return image in bytes format
    return StreamingResponse(content=get_bytes_from_image(final_image), media_type="image/jpeg")

