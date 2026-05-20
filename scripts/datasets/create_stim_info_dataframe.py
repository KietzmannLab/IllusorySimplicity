import pandas as pd
import numpy as np
from os import path

stimnames_file = path.join("Data", "stim_names.txt")
# load
stim_info = pd.read_csv(stimnames_file, header=None, names=["filenames"])
stim_info["im_id"] = np.arange(1, len(stim_info) + 1)

stim_info["category"] = stim_info["filenames"].apply(
    lambda x: "_".join(x.split("_")[:-1])
)

unique_categories = stim_info["category"].unique()
category_id_map = {c: i + 1 for i, c in enumerate(unique_categories)}
stim_info["cat_id"] = stim_info["category"].apply(lambda x: category_id_map[x])

body_parts = ["arm", "ankle", "leg", "hand", "chest1", "elbow", "mouth"]
human_face = [
    "man",
    "woman",
    "baby",
]

mammal = [
    "monkey",
    "elephant",
    "dog",
    "cat",
    "pig",
    "tiger",
    "zebra",
    "mouse1",
    "alpaca",
    "bear",
    "horse",
    "rabbit",
    "cow",
    "beaver",
    "chipmunk",
]
non_mammal = [
    "bird",
    "snake",
    "spider",
    "alligator",
    "ant",
    "bee",
    "lizard",
    "bug",
    "fly",
    "beetle",
    "dragonfly",
]

fruit = [
    "apple",
    "banana",
    "grape",
    "blackberry",
    "cantaloupe",
    "raspberry",
    "kiwi",
]

vegetable = [
    "lettuce",
    "cucumber",
    "bell_pepper",
    "broccoli",
    "carrot",
    "beet",
]

other_food = [
    "nut",
    "almond",
    "donut",
    "marshmallow",
]


plants = [
    "bamboo",
    "tree",
    "cactus",
    "fern",
    "flower",
    "bush",
    "weed",
    "tumbleweed",
    "aloe",
]

other_natural = [
    "fire",
    "shell2",
    "rock",
    "stick",
    "stalagmite",
    "pinecone",
]


artificial_small_other = [
    "jar",
    "backgammon",
    "baseball",
    "bolt",
    "beachball",
    "kazoo",
]

tools = [
    "axe",
    "joystick",
    "spoon",
    "key",
    "crayon",
    "hammer",
    "pan",
    "ashtray",
    "umbrella",
    "dustpan",
]


vehicles = [
    "bike",
    "boat",
    "helicopter",
    "car",
    "bulldozer",
]

furniture = [
    "bed",
    "couch",
    "chair",
    "table",
    "coat_rack",
    "crate",
    "cabinet",
]

outside_large = [
    "tent",
    "doghouse",
    "traffic_light",
    "ferris_wheel",
]

human = body_parts + human_face
animal = mammal + non_mammal
animate = human + mammal + non_mammal
food = fruit + vegetable + other_food
natural = food + plants + other_natural
artificial_small = artificial_small_other + tools
artificial_large = vehicles + furniture + outside_large
artificial = artificial_small + artificial_large
inanimate = natural + artificial

superordinate_categories = {
    "body_parts": np.array(body_parts),
    "human_face": np.array(human_face),
    "human": np.array(human),
    "mammal": np.array(mammal),
    "non_mammal": np.array(non_mammal),
    "animal": np.array(animal),
    "animate": np.array(animate),
    "inanimate": np.array(inanimate),
    "fruit": np.array(fruit),
    "vegetable": np.array(vegetable),
    "other_food": np.array(other_food),
    "food": np.array(food),
    "natural": np.array(natural),
    "plants": np.array(plants),
    "other_natural": np.array(other_natural),
    "artificial": np.array(artificial),
    "artificial_small": np.array(artificial_small),
    "tools": np.array(tools),
    "artificial_small_other": np.array(artificial_small_other),
    "vehicles": np.array(vehicles),
    "furniture": np.array(furniture),
    "outside_large": np.array(outside_large),
    "artificial_large": np.array(artificial_large),
}

# for each category, create a column with 1s and 0s denoting membership
for supcat in superordinate_categories.keys():
    is_cat_member = stim_info["category"].apply(
        lambda x: np.isin(x, superordinate_categories[supcat])
    )
    stim_info[supcat] = is_cat_member.astype(int)

stim_info.to_csv(path.join("Data", "stimulus_information.csv"), index=False)
