import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        steering_angle_vect = model.predict(image_array[None, :, :, :], batch_size=1)
        
        drive_UK = True
        if drive_UK:
            steering_angle = float(steering_angle_vect[0][0])
        else:
            steering_angle = float(steering_angle_vect[0][1])
            
        
        # Fun experimentation with a Constant controller to roughly maintain
        # a speed and reduce the throttle when turning the vehicle
        if abs(steering_angle)<0.2:
            target_speed = 20
        elif abs(steering_angle)<0.5:
            target_speed = 15
        else:
            target_speed = 10
            
        throttle_gain = 0.1
        throttle = min(max(throttle_gain*(target_speed-float(speed)), -1.0), 1.0)
            
        print(steering_angle_vect, throttle)
        send_control(1*steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model pickle file (YAML). Model should be on the same path.'
    )
    parser.add_argument(
        'model_weights',
        type=str,
        help='Path to model weights (H5 files). Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()


    # model reconstruction from YAML
    import pickle
    with open(args.model, mode='rb') as f:
        yaml_string = pickle.load(f)
        
    from keras.models import model_from_yaml
    model = model_from_yaml(yaml_string)
    model.load_weights(args.model_weights)
    
    #model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
