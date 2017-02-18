#!/usr/python

#!/usr/python
from eventlet.green import zmq
import eventlet
 
CTX = zmq.Context(1)

 
def bob_client(ctx):
    print("STARTING BOB")
    bob = zmq.Socket(CTX, zmq.REQ)
    bob.connect("ipc:///tmp/test")
    
    bob.send(str(drive_UK).encode('ascii'))
    print("BOB GOT:", bob.recv())
        



import tkinter as tk
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()
        print("STARTING BOB")
        self.bob = zmq.Socket(CTX, zmq.REQ)
        self.bob.connect("ipc:///tmp/test")
        self.drive_UK = False
        #bob = eventlet.spawn(bob_client, CTX)
        #bob.wait()

    def create_widgets(self):
        self.us_button = tk.Button(self)
        self.us_button["text"] = "US\n(click me)"
        self.us_button["command"] = self.say_us
        self.us_button.pack(side="top")
        
        self.uk_button = tk.Button(self)
        self.uk_button["text"] = "UK\n(click me)"
        self.uk_button["command"] = self.say_uk
        self.uk_button.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=root.destroy)
        self.quit.pack(side="bottom")

    def say_uk(self):
        print("UK")
        self.drive_UK = 1
        self.bob.send(str(self.drive_UK).encode('ascii'))
        self.bob.recv()

    def say_us(self):
        print("US")
        self.drive_UK = 0
        self.bob.send(str(self.drive_UK).encode('ascii'))
        self.bob.recv()

root = tk.Tk()
app = Application(master=root)
app.mainloop()



