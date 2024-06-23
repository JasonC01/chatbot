import base64
import awsgi
from hello_world.app import app

def lambda_handler(event, context):
    # event['image'] = event['image'].encode('utf-8')
    if 'image' in event:
        try:
            event['image'] = base64.b64decode(event['image'])
        except base64.binascii.Error as error:
            print(f"Decoding error: {error}")
            return {"statusCode": 400, "body": "Invalid base64 image data"}
    print("hello")
    # print(event['image'])
    return awsgi.response(app, event, context)