from ctypes import c_char_p, cast
from typing import Dict

from pyglet import gl

__all__ = ["get_graphics_information"]


def get_graphics_information() -> Dict:
    options = {
        "vendor": gl.GL_VENDOR,
        "renderer": gl.GL_RENDERER,
        "version": gl.GL_VERSION,
        "shading-language-version": gl.GL_SHADING_LANGUAGE_VERSION,
        # 'extensions': gl.GL_EXTENSIONS
    }

    results = {}
    for o, code in options.items():
        a = gl.glGetString(code)
        b: bytes = cast(a, c_char_p).value
        res = b.decode()
        results[o] = res
    return results


if __name__ == "__main__":
    print(str(get_graphics_information()))
