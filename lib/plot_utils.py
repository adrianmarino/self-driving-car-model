import matplotlib.pyplot as plt
from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot
import seaborn as sns


def show_sample(image, angle=None):
    title = f'Shape: {image.shape}'

    if angle is not None:
        title += f'  -  Steering Angle: {angle:.6f}'

    grid_display(
        images=[image],
        titles=[title],
        columns=1,
        figure_size=(6, 6),
        font_size=14
    )


def show_distribution(dataset, title='Title'):
    sns.set(rc={'figure.figsize': (22, 5)})

    b = sns.distplot(dataset.labels, hist=True, kde=False)

    b.axes.set_title(f'{title} (Samples: {len(dataset)})', fontsize=15)
    b.set_xlabel("Steering angle", fontsize=15)
    b.set_ylabel("Number of occurrences", fontsize=15)

    plt.show()


def graph_model(model):
    converter = model_to_dot(model, show_shapes=True, show_layer_names=True)
    image = converter.create(prog='dot', format='svg')
    display(SVG(image))


def show_image(image, size=(17, 17)):
    grid_display(
        images=[image],
        titles=[],
        columns=1,
        figure_size=size
    )


def grid_display(
        images,
        titles,
        columns=2,
        figure_size=(10, 10),
        font_size=28
):
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
            plt.title(titles[i], {'fontsize': font_size})
    plt.show()
