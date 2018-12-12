import argparse
import base64
import numpy as np
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from keras.models import load_model
from keras.optimizers import Adam

from lib.image_preprocessor import ImagePreprocessor
from lib.config import Config
from lib.metrics import rmse
from lib.model_factory import ModelFactory

cfg = Config('./config.yml')
    
# initialize our server
sio = socketio.Server()

# our flask (web) app
app = Flask(__name__)

model = None
prev_image_array = None

k_p = float(cfg['simulator']['k_p'])
target_speed = float(cfg['simulator']['speed'])


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 20
controller.set_desired(set_speed)


# registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        try:
            steering_angle = float(model.predict(current_camera_image(data), batch_size=1))
            throttle = controller.update(get_speed(data))
            print(f'Angle: {steering_angle}, Throttle: {throttle}')
            send_control(steering_angle, throttle)
        except Exception as e:
            print(f'ERROR: {e}')
    else:
        sio.emit('manual', data={}, skip_sid=True)


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
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )

    model = ModelFactory.create_nvidia_model(metrics=[rmse])
    args = parser.parse_args()
    model.load_weights(args.model)

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
