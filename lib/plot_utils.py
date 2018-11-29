from IPython.display import Image
from IPython.display import display
import matplotlib.pyplot as plt
import cv2


def show_img(path):
    display(Image(path))


def show_example(features, labels):
    ordered_paths = [features[1], features[0], features[2]]
    grid_display(
        images=[cv2.imread(path) for path in ordered_paths],
        titles=['LEFT CAMERA', 'CENTER CAMERA', 'RIGHT CAMERA'],
        columns=3,
        figure_size=(30, 35)
    )
    print("Steering Angle: ", labels[0])


def grid_display(
        images, titles, columns=2, figure_size=(10, 10)):
    fig = plt.figure(figsize=figure_size)
    column = 0
    for i in range(len(images)):
        column += 1
        #  check for end of column and create a new figure
        if column == columns + 1:
            fig = plt.figure(figsize=figure_size)
            column = 1
        fig.add_subplot(1, columns, column)
        plt.imshow(images[i], cmap=plt.cm.Greys)
        plt.axis('off')
        if len(titles) >= len(images):
            plt.title(titles[i])
    plt.show()
