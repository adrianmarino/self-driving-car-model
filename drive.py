import argparse
import base64
import numpy as np
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from keras.models import load_model
from lib.image_preprocessor import ImagePreprocessor
from lib.config import Config

# initialize our server
sio = socketio.Server()

# our flask (web) app
app = Flask(__name__)

model = None
prev_image_array = None

k_p = 0.15
target_speed = 15


# registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        try:
            next_steering_angle = float(model.predict(
                current_camera_image(data),
                batch_size=1
            ))
            throttle = next_throttle_value(get_speed(data))

            print(f'Angle: {next_steering_angle}, Throttle: {throttle}')
            send_control(next_steering_angle, throttle)
        except Exception as e:
            print(f'ERROR: {e}')
    else:
        sio.emit('manual', data={}, skip_sid=True)


def next_throttle_value(current_speed):
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    # A proportional controller to ease up on the speed and angles
    # https://en.wikipedia.orgss/wiki/Proportional_control
    # k_p = Proportional gain
    error = target_speed - current_speed
    throttle = k_p * error
    return throttle


def get_speed(data): return float(data["speed"])


def current_camera_image(data):
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    return pre_process_image(image)


def pre_process_image(image):
    # from PIL image to numpy array
    image = np.asarray(image)
    image = image_preprocessor.process(image)  # apply the preprocessing
    # the model expects 4D array
    return np.array([image])


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={'steering_angle': steering_angle.__str__(), 'throttle': throttle.__str__()},
        skip_sid=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )

    args = parser.parse_args()
    model = load_model(args.model)

    cfg = Config('./config.yml')

    image_preprocessor = ImagePreprocessor(
        top_offset=cfg['train']['preprocess']['crop']['top_offset'],
        bottom_offset=cfg['train']['preprocess']['crop']['bottom_offset'],
        input_shape=(
            cfg['network']['input_shape']['height'],
            cfg['network']['input_shape']['width'],
            cfg['network']['input_shape']['channels']
        )
    )

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
