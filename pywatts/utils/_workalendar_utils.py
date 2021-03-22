
import workalendar.africa
import workalendar.america
import workalendar.asia
import workalendar.europe
import workalendar.oceania
import workalendar.usa

from pywatts.core.exceptions.util_exception import UtilException


def _init_calendar(continent: str, country: str):
    """ Check if continent and country are correct and return calendar object.

    :param continent: Continent where the country or region is located.
    :type continent: str
    :param country: Country or region to use for the calendar object.
    :type country: str
    :return: Returns workalendar object to use for holiday lookup.
    :rtype: workalendar object
    """
    if hasattr(workalendar, continent.lower()):
        module = getattr(workalendar, continent.lower())
        if hasattr(module, country):
            return getattr(module, country)()
        else:
            raise UtilException(f"The country {country} does not fit to the continent {continent}")
    else:
        raise UtilException(f"The continent {continent} does not exist.")