import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 20))


def show_grid_results_clip(images, text_classes, probabilities_clip):
    for idx in range(len(images)):
        # show original image
        fig.add_subplot(len(images), 2, 2 * (idx + 1) - 1)
        plt.imshow(images[idx])
        plt.xticks([])
        plt.yticks([])

        # show probabilities
        fig.add_subplot(len(images), 2, 2 * (idx + 1))
        plt.barh(
            range(len(probabilities_clip[0].detach().numpy())),
            probabilities_clip[idx].detach().numpy(),
            tick_label=text_classes,
        )
        plt.xlim(0, 1.0)

        plt.subplots_adjust(
            left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.8
        )

    plt.show()
