import matplotlib.pyplot as plt
from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot
import seaborn as sns

from lib.utils.array_utils import wrap


def show_sample(sample):
    show_features(sample)
    print(sample.str_labels())


def show_features(
        sample,
        image_features_columns=['left', 'center', 'right']
):
    non_image_features_columns = list(set(sample.feature_columns) - set(image_features_columns))
    print(sample.str_features(non_image_features_columns))
    print('\t- Images:')
    images = [sample.feature_image(feature) for feature in image_features_columns]
    titles = [f'{label.capitalize()} Camera {images[0].shape}' for label in sample.feature_columns]
    grid_display(
        images=images,
        titles=titles,
        columns=len(titles),
        figure_size=(40, 40),
        font_size=22
    )



def show_sample_data(image, angle=None):
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


def histograms(
        values,
        x_labels,
        titles,
        y_label="Number of occurrences",
        size=(20, 5),
        vertical=False
):
    f, axes = plt.subplots(1, len(values), figsize=size, sharex=False)

    axes = wrap(axes)
    for index, value in enumerate(values):
        if vertical:
            x_label, y_label = y_label, x_labels[index]
        else:
            x_label, y_label = x_labels[index], y_label

        b = sns.distplot(value, ax=axes[index], hist=True, kde=False, vertical=vertical)
        b.axes.set_title(f'{titles[index]} - Size: {len(value)}', fontsize=15)
        b.set_xlabel(x_label, fontsize=15)
        b.set_ylabel(y_label, fontsize=15)


def graph_model(model):
    converter = model_to_dot(model, show_shapes=True, show_layer_names=True)
    image = converter.create(prog='dot', format='svg')
    display(SVG(image))


def show_image(image, size=(10, 5)):
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


def show_values(labels, values):
    print()
    for label, value in zip(labels, values):
        print(f'- {label}: {value}')
    print()
