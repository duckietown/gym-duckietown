mapping = {
    "house": "3deb34",
    "bus": "ebd334",
    "truck": "961fad",
    "duckie": "cfa923",
    "cone": "ffa600",
    "floor": "000000",
    "grass": "000000",
    "barrier": "000099"
}

mapping = {
    key:
        [int(h[i:i+2], 16) for i in (0,2,4)]
    for key, h in mapping.items()
}