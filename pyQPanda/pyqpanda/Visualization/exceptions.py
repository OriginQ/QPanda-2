# -*- coding: utf-8 -*-

# This code is part of PyQpanda.
#
# (C) Copyright Origin Quantum 2018-2019\n
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exceptions for errors raised by PyQpanda."""


class QError(Exception):
    """Base class for errors raised by PyQpanda."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(' '.join(message))
        self.message = ' '.join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class QIndexError(QError, IndexError):
    """Raised when a sequence subscript is out of range."""
    pass


class QUserConfigError(QError):
    """Raised when an error is encountered reading a user config file."""
    message = "User config invalid"

class CircuitError(QError):
    """Base class for errors raised while processing a circuit."""

    pass