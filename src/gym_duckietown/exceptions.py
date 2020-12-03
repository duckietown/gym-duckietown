__all__ = ["GymDuckietownException", "InvalidMapException", "NotInLane"]

from zuper_commons.types import ZException


class GymDuckietownException(ZException):
    pass


class InvalidMapException(GymDuckietownException):
    pass


class NotInLane(GymDuckietownException):
    """ Raised when the Duckiebot is not in a lane. """
