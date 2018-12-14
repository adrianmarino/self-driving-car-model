import argparse
import base64
from io import BytesIO

import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from flask import Flask

from lib.config import Config
from lib.image_preprocessor import ImagePreprocessor
from lib.model.metrics import rmse
from lib.model.model_factory import ModelFactory

sio = socketio.Server()
app = Flask(__name__)

config = Config('./config.yml')
model = None
image_preprocessor = ImagePreprocessor.create_from(config)


# registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        try:
            steering_angle, throttle = predict(current_camera_frame(data))
            print(f'<< Steering Angle: {steering_angle:0.6f} | Throttle: {throttle:0.6f} >>')
            send_control(steering_angle, throttle)
        except Exception as e:
            print(f'ERROR: {e}')
    else:
        sio.emit('manual', data={}, skip_sid=True)


def predict(image):
    results = model.predict(image, batch_size=1)
    return float(results[0]), float(results[1])


def current_speed(data): return float(data["speed"])


def current_camera_frame(data):
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    return pre_process_image(image)


def pre_process_image(frame):
    # from PIL image to numpy array
    frame = np.asarray(frame)
    frame = image_preprocessor.process(frame)
    # the model expects 4D array
    return np.array([frame])


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

    
def send_control(steering_angle, throttle):
    sio.emit(
        'steer',
        data={'steering_angle': steering_angle.__str__(), 'throttle': throttle.__str__()},
        skip_sid=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'weights',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )

    model = ModelFactory.create_nvidia_model(metrics=[rmse])
    model.load_weights(parser.parse_args().weights)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
