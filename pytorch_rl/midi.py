import mido

# Note: requires python-rtmidi package

inPort = None
outPort = None

# Map of control numbers to values in 0..127
controlCbs = {}

def connect():
    global inPort
    global outPort

    inputNames = mido.get_input_names()
    outputNames = mido.get_output_names()

    for name in inputNames:
        if "Launch Control" in name:
            print("Connecting to %s" % name)
            inPort = mido.open_input(name)

    for name in outputNames:
        if "Launch Control" in name:
            print("Connecting to %s" % name)
            outPort = mido.open_output(name)

    for i in range(0, 8):
        setLED(i, 'off')

def setLED(ledNo, color):

    # template=8 is the first factory template
    template=8

    colorMap = {
        'off': 12,
        'red': 15,
        'green': 60,
        'yellow': 62
    }

    value = colorMap[color]

    msg = mido.Message('sysex', data=[0, 32, 41, 2, 10, 120, template, ledNo, value])
    print(msg)
    outPort.send(msg)

def setControlCb(ctrlNo, cb):
    controlCbs[ctrlNo] = cb

def handleMsgs():
    for msg in inPort.iter_pending():
        if msg.type == 'control_change':
            #print('control #%s = %s' % (msg.control, msg.value))
            if msg.control in controlCbs:
                controlCbs[msg.control](msg.value)
