#!/usr/bin/env python3
"""stacked bar chart"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """stacked bar cart Farrah, Fred, Felicia"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    people = ["Farrah", "Fred", "Felicia"]
    x = np.arange(len(people))
    width = 0.5

    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    plt.bar(people, apples, width, color="red")
    plt.bar(people, bananas, width, bottom=apples, color="yellow")
    plt.bar(people, oranges, width, bottom=apples+bananas, color="#ff8000")
    plt.bar(people, peaches, width,
            bottom=apples+bananas+oranges, color="#ffe5b4")
    plt.legend(["apples", "bananas", "oranges", "peaches"])
    plt.ylim(0, 80)
    plt.ylabel("Quantity of Fruit")
    plt.title("Number of Fruit per Person")
    plt.show()
    