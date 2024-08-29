# refer to http://blog.sipeed.com/p/680.html

import sensor, image, lcd, time
import KPU as kpu
import gc, sys
from Maix import utils


def main(model_addr="/sd/m.kmodel", lcd_rotation=0, sensor_hmirror=False, sensor_vflip=False):
    gc.collect()

    labels={
    "0":"cat",
    "1":"dog"
    }

    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_windowing((224, 224))
    sensor.set_hmirror(sensor_hmirror)
    sensor.set_vflip(sensor_vflip)
    sensor.run(1)

    lcd.init(type=1)
    lcd.rotation(lcd_rotation)
    lcd.clear(lcd.WHITE)

    print("loading model")
    task = kpu.load(model_addr)
    print("model loaded")
    kpu.set_outputs(task,0,2,1,1)

    try:
        while(True):
            img = sensor.snapshot()
            t = time.ticks_ms()
            img2 = img.resize(128,128)
            img2.pix_to_ai()
            fmap = kpu.forward(task, img2)
            t = time.ticks_ms() - t
            plist=fmap[:]
            pmax=max(plist)
            max_index=plist.index(pmax)
            index_key = str(max_index)
            if index_key in labels.keys() and pmax > 0.80:
                 img.draw_string(0,0, " %.2f\n %s" %(pmax, labels[index_key]), scale=2, color=(0, 0, 0))
                 img.draw_string(3,3, " %.2f\n %s" %(pmax, labels[index_key]), scale=2, color=(255, 255, 255))
                 img.draw_string(0, 200, " t:%dms" %(t), scale=2, color=(255, 0, 0))
            else:
                  img.draw_string(0,0, " %.2f\n %s" %(pmax, "NA"), scale=2, color=(0, 0, 0))
            lcd.display(img)
    except Exception as e:
        sys.print_exception(e)
    finally:
        kpu.deinit(task)


if __name__ == "__main__":
    try:
        main(model_addr=0x300000, lcd_rotation=0, sensor_hmirror=False, sensor_vflip=True)
    except Exception as e:
        sys.print_exception(e)
    finally:
        gc.collect()
