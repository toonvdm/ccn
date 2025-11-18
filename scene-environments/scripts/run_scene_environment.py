from scene_environments import SceneEnvironment
import matplotlib.pyplot as plt


if __name__ == "__main__":
    objects = [
        "002_master_chef_can",
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
    ]

    env = SceneEnvironment(
        objects,
        n_objects=3,
        seed=1,
        render_table=True,
        eval=True,
        resample_objects=True,
        random_table_color=True,
        resample_table_color=True,
    )

    state = env.state

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(state["rgb"])
    ax[0].set_title("RGB")
    ax[1].imshow(state["goal_rgb"])
    ax[1].set_title("Goal")
    [a.axis("off") for a in ax.flatten()]
    plt.show()
