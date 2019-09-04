"""
This file holds a specialized function to display a confusion matrix for use in the paper.
"""

import matplotlib.pyplot as plt
import numpy as np

from src import em_constants as emc

CM = np.array([
        [[1.00, 0.00],
         [0.94, 0.06]],
        [[1.00, 0.00],
         [0.97, 0.03]],
        [[1.00, 0.00],
         [0.87, 0.13]],
        [[1.00, 0.00],
         [0.79, 0.21]],
        [[1.00, 0.00],
         [0.97, 0.03]],
        [[1.00, 0.00],
         [0.86, 0.14]],
        [[1.00, 0.00],
         [0.53, 0.47]],
    ])



def visualize_confusion_matrix():
    """
    Visualizes the test set confusion matrix.
    """
    plt.rcParams.update({'font.size': 21})

    # Display the confusion matrix for each emotion
    labels = ["Absent", "Present"]

    for em_index in range(0, emc.NUM_EMOTIONS):
        # Normalize the confusion matrix
        em_cm = CM[em_index]
        em_cm = em_cm.astype('float') / em_cm.sum(axis=1)[:, np.newaxis]

        # Plot the confusion matrix
        fig, ax = plt.subplots()
        im = ax.imshow(em_cm, interpolation='nearest',
                       cmap=plt.get_cmap("Blues"))
        ax.figure.colorbar(im, ax=ax)

        # Set the x and y axis labels
        em_label = emc.INVERT_EMOTION_MAP[em_index].title()
        ax.set(xticks=np.arange(em_cm.shape[1]),
               yticks=np.arange(em_cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=labels, yticklabels=labels,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax.set_ylim(len(em_cm) - 0.5, -0.5)
        ax.set_title('Normalized Confusion Matrix for {}'.format(em_label),
                     pad=20)

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = em_cm.max() / 2.
        for i in range(em_cm.shape[0]):
            for j in range(em_cm.shape[1]):
                ax.text(j, i, format(em_cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if em_cm[i, j] > thresh else "black")

        fig.tight_layout()
        fig.set_size_inches(8, 8)
        fig.savefig('{}.png'.format(em_label), dpi=600)
        # plt.show()


def main():
    visualize_confusion_matrix()


if __name__ == "__main__":
    main()
