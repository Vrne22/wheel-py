import time
import frida
js_code = '''
var rotationAddr = ptr("0x7C894F22A0");
var velYAddr = ptr("0x7C893A1038");

function trackValues() {
    try {
        var rotation = rotationAddr.readFloat();
        var velY = velYAddr.readFloat();

        send({ r: rotation, vy: velY});
    } catch (e) {
        console.log(e);
    }

    setTimeout(trackValues, 5);
}

trackValues();
'''


def on_message(message, data):
    payload = message.get("payload", {})
    if "r" in payload and "vy" in payload:
        print(payload["r"], payload["vy"])

def start_frida():
    device = frida.get_usb_device()
    pid = device.get_frontmost_application().pid
    session = device.attach(pid)

    script = session.create_script(js_code)
    script.on('message', on_message())
    script.load()
        
