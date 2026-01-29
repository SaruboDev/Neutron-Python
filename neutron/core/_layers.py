"""
Contains general stuff for layers, such as an ignore list with only layers.
"""

from neutron.core._module import Module
import neutron.modules as nn

"""
I need the ignore list because i want to avoid checking inside layers, but i do want to check
inside Module classes that aren't layers.
"""
ignore_list: list = [
    nn.Linear
]


def add_to_ignore_list(class_to_ignore: Module) -> None:
    """
    Literally just appends to the ignore_list.\n
    SHOULD be used if you made a custom layer.
    """

    ignore_list.append(class_to_ignore)

    return