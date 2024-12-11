from datetime import datetime
import tkinter as tk
from PIL import Image, ImageTk
import paho.mqtt.client as mqtt
import threading
import io
import time
import numpy as np
import cv2
import os

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, butter, lfilter

"""
pip install paho-mqtt
pip install opencv-python
"""


###################### Config File Setup #######################
# Config file will be placed in the same directory as this script
def read_or_create_config(file_path, default_config):
    # Check if the config file exists
    if not os.path.exists(file_path):
        # If not, create it with default values
        with open(file_path, "w") as file:
            for key, value in default_config.items():
                file.write(f"{key} = {value}\n")

    # Read config values
    config = {}
    with open(file_path, "r") as file:
        for line in file:
            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()

    return config


# Default configuration values
default_config = {
    "MQTT_SERVER_ADDRESS": "127.0.0.1",
    "MQTT_SERVER_PORT": "1883",
    # "MQTT_SERVER_SUBSCRIBE": "3036264884/esp32camFuwamoco/pics",
    'MQTT_SERVER_SUBSCRIBE': "TESTING2",
    "SHOW_FACE_ROI": "True",
}

# File path for the config file
config_file_path = "setting.config"

# Read or create the config
config = read_or_create_config(config_file_path, default_config)

# Accessing the configuration
MQTT_SERVER_ADDRESS = config["MQTT_SERVER_ADDRESS"]
MQTT_SERVER_PORT = int(config["MQTT_SERVER_PORT"])
MQTT_SERVER_SUBSCRIBE = config["MQTT_SERVER_SUBSCRIBE"]
SHOW_FACE_ROI = config["SHOW_FACE_ROI"] == "True"

######################################################
######################################################


class ImageStreamApp:
    def __init__(self, root):
        print("GUI Created and running ...")
        self.root = root
        self.root.title("ECP32 Camera Stream")
        self.root.geometry("1200x800")  # width x height

        ###################### GUI Layout #######################

        # Create 4 row with 4 columns all in same weight
        self.root.grid_rowconfigure((0, 1, 2, 3), weight=1)
        self.root.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Camera Stream start on left top
        # Canvas for the video stream + Face ROI
        self.canvas = tk.Canvas(root, bd=2, relief=tk.GROOVE)
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=5, pady=5)

        # Chart Placeholder
        self.chart_frame = tk.Frame(root, bd=2, relief=tk.GROOVE)
        self.chart_frame.grid(
            row=0, column=1, rowspan=2, columnspan=3, sticky="nsew", padx=5, pady=5
        )

        # IP and Port Entry
        ip_port_frame = tk.LabelFrame(root, text="Settings", bd=2, relief=tk.GROOVE)
        ip_port_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        tk.Label(ip_port_frame, text="IP Address:").grid(row=0, column=0, sticky="w")
        self.ip_entry = tk.Entry(ip_port_frame)
        self.ip_entry.grid(row=0, column=1, sticky="ew")
        self.ip_entry.insert(0, MQTT_SERVER_ADDRESS)

        tk.Label(ip_port_frame, text="Port:").grid(row=1, column=0, sticky="w")
        self.port_entry = tk.Entry(ip_port_frame)
        self.port_entry.grid(row=1, column=1, sticky="ew")
        self.port_entry.insert(0, MQTT_SERVER_PORT)

        tk.Label(ip_port_frame, text="Subscribe:").grid(row=2, column=0, sticky="w")
        self.sub_entry = tk.Entry(ip_port_frame)
        self.sub_entry.grid(row=2, column=1, sticky="ew")
        self.sub_entry.insert(0, MQTT_SERVER_SUBSCRIBE)

        # Create the Label for 'Face ROI'
        tk.Label(ip_port_frame, text="Face ROI:").grid(row=3, column=0, sticky="w")
        # Create a variable to track the state of the checkbox
        self.show_face_roi_var = tk.BooleanVar()
        self.show_face_roi_var.set(SHOW_FACE_ROI)  # Set default value to True
        # Create the Checkbutton for showing/hiding Face ROI
        self.show_face_roi_checkbox = tk.Checkbutton(
            ip_port_frame, variable=self.show_face_roi_var
        )
        self.show_face_roi_checkbox.grid(row=3, column=1, sticky="w")

        # Start and Stop Buttons will be in the same column as IP+Port but in different rows
        self.start_button = tk.Button(
            root, text="Start Listening", command=self.start_listening
        )
        self.start_button.grid(row=2, column=2, sticky="ew", padx=5)

        self.stop_button = tk.Button(
            root, text="Stop Listening", command=self.stop_listening
        )
        self.stop_button.grid(row=2, column=3, sticky="ew", padx=5)

        # Log Panel
        log_frame = tk.Frame(root, bd=2, relief=tk.GROOVE)
        log_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
        self.log = tk.Text(
            log_frame, state="disabled", height=6
        )  # Set the height to 5 lines
        self.log.pack(side=tk.LEFT, fill=tk.X, expand=True)

        scrollbar = tk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log["yscrollcommand"] = scrollbar.set

        #########################################################
        ###################### Variables ########################

        # Flag to indicate if the system is "listening"
        self.is_listening = True  # Default to True

        # frames per second
        self.fps = 24
        self.step_size = 20
        self.window_size = 180

        self.last_face_roi_position = (0, 0, 0, 0)

        self.bpmStr = ''
        self.getBpmList = []

        self.heartBeatUpdateFrameCounter = 0

        self.heartbeat_signal = []  # green channel signal
        self.forehead_roi_for_heart_beat_queue = []
        self.all_predicted_hrs = []

        # Frame, Face and forehead ROI queue
        self.frame_queue = []
        self.face_roi_queue = []
        self.forehead_roi_queue = []
        
        # Buffer to store signal for heart rate calculation
        self.heart_rate_buffer = []

        # Load the pre-trained face cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        #########################################################
        ###################### On Mounted #######################

        # Chart Setup
        # Create a figure for the plot
        self.fig = Figure(figsize=(6, 4), dpi=100)  # Adjust the size as needed
        self.fig.subplots_adjust(hspace=0.5)
        self.fig.tight_layout()

        # Create subplots
        self.ax_hr = self.fig.add_subplot(211)  # Heart Beat Rate Plot Per Second
        self.ax_time = self.fig.add_subplot(212)  # heartbeat_signal plot

        # Set titles and labels for Hr plot
        self.ax_hr.set_title("Heart Rate")
        self.ax_hr.set_xlabel("Frames")
        self.ax_hr.set_ylabel("Bpm")

        # Set titles and labels for time-domain plot
        self.ax_time.set_title("Pulse Signal")
        self.ax_time.set_xlabel("Frames")
        self.ax_time.set_ylabel("Amplitude")

        # Create a Tkinter canvas to embed the plots
        self.ChartCanvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas_widget = self.ChartCanvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Initialize data for the plots
        self.x_data_hr = np.linspace(0, 2 * np.pi, 100)  # Placeholder data
        self.y_data_hr = np.sin(self.x_data_hr)  # Placeholder data
        self.new_x_data_hr = [0, self.window_size]

        # Calculate the remaining values
        remaining_values = np.arange(self.new_x_data_hr[-1] + self.step_size, 2001, self.step_size)

        # Concatenate the initial and remaining values
        self.new_x_data_hr = np.concatenate([self.new_x_data_hr , remaining_values])

        self.x_data_time = np.arange(len(self.heartbeat_signal))  # Time values
        self.y_data_time = self.heartbeat_signal  # Heartbeat signal data
        
        self.y_predicted_hr_axis = self.all_predicted_hrs  # Placeholder data
        self.x_time_axis = np.arange(1800)  # Time values hard-coded
        self.y_heartbeat_signal_axis = self.heartbeat_signal  # Heartbeat signal data

        # Initialize the plots
        (self.line_hr,) = self.ax_hr.plot(self.x_data_hr, self.y_data_hr)
        (self.line_time,) = self.ax_time.plot(self.x_data_time, self.y_data_time)

        # MQTT setup
        self.client = mqtt.Client()
        self.client.on_message = self.on_message

        # Start MQTT client in the main thread
        self.start_mqtt_client()

        # Update GUI periodically
        self.update_gui()

        # Update widget states
        self.update_widget_states()

        # Schedule the first update
        self.update_chart()

        #########################################################

    ###################### Callbacks ########################
    # MQTT Start
    def start_mqtt_client(self):
        if not self.client.is_connected():
            try:
                self.client.connect(self.ip_entry.get(), int(self.port_entry.get()), 60)
                self.client.subscribe(self.sub_entry.get())
                self.client.loop_start()  # This starts a new thread
            except Exception as e:
                logStr = f"Error connecting to MQTT Broker: {e}"
                self.update_log(logStr)
                print(logStr)

    # MQTT Stop
    def stop_mqtt_client(self):
        def disconnect_mqtt():
            self.client.loop_stop()  # Stop the loop
            self.client.disconnect()  # Disconnect the client
            logStr = f"MQTT client disconnected"
            self.update_log(logStr)
            print(logStr)

        # Use a separate thread to handle the disconnection
        stop_thread = threading.Thread(target=disconnect_mqtt)
        stop_thread.start()

        # Clear the buffers
        self.heart_rate_buffer.clear()

    # MQTT Callback
    def on_message(self, client, userdata, message):
        # Image send in byte format
        payload = message.payload
        decoded_image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)

        # Convert color space from BGR to RGB
        frame = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)

        # If show face ROI is false, still need to detect face to calculate heart rate, just dont show the face ROI
        faces = self.face_cascade.detectMultiScale(
            frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            for x, y, w, h in faces:
                self.last_face_roi_position = (x, y, w, h)

                if(self.show_face_roi_var.get()):
                    cv2.rectangle(decoded_image, (x, y), (x + w, y + h), (255, 0, 0), 3)

                # Define ROI for the face
                face_roi = decoded_image[y : y + h, x : x + w]

                # Define ROI for the forehead (adjust the coordinates based on your preference)
                forehead_roi = decoded_image[
                    y : y + int(h * 0.3), x + int(w * 0.2) : x + int(w * 0.8)
                ]

                # Draw rectangles around the face and forehead regions
                if(self.show_face_roi_var.get()):
                    cv2.rectangle(decoded_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cv2.rectangle(
                        decoded_image,
                        (x + int(w * 0.2), y),
                        (x + int(w * 0.8), y + int(h * 0.3)),
                        (0, 255, 0),
                        3,
                    )

                # Resize face and forehead ROIs to a fixed size
                face_roi = cv2.resize(face_roi, (100, 100))
                forehead_roi = cv2.resize(forehead_roi, (100, 30))

                # Add face and forehead ROIs to their respective queues
                self.face_roi_queue.append(face_roi)

                self.forehead_roi_queue.append(forehead_roi)

                self.forehead_roi_for_heart_beat_queue.append(forehead_roi)

                average_color = np.mean(forehead_roi, axis=(0, 1))  # Calculate average color
                self.heartbeat_signal.append(average_color[1])

        # Display the original image with detected face and forehead ROIs
        # Convert color space from BGR to RGB after processing the image
        # frame_rgb = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
        self.frame_queue.append(decoded_image)
        self.heartBeatUpdateFrameCounter += 1

        # Update log with first 10 bytes of the received data
        self.update_log(message.payload[:10])

    def update_gui(self):
        # Main Video Frame, IF show Face ROI is True, the frame is already include the face ROI
        if self.frame_queue:
            # frame = self.frame_queue.pop(0)
            frame = self.frame_queue[-1]

            # Get the size of the label widget
            label_width = self.canvas.winfo_width()
            label_height = self.canvas.winfo_height()

            # Maintain aspect ratio during resize
            aspect_ratio = frame.shape[1] / frame.shape[0]  # width / height
            new_width = int(label_height * aspect_ratio)
            new_height = label_height

            # Check if new width is larger than the label width
            if new_width > label_width:
                new_width = label_width
                new_height = int(label_width / aspect_ratio)

            # Resize the frame
            frame = cv2.resize(frame, (new_width, new_height))

            # Convert color space from BGR to RGB
            # frame = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update the label with the new image
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # is checkbox is checked
        if self.show_face_roi_var.get():
            # If the queue is not empty
            if len(self.face_roi_queue) >= self.window_size and len(self.forehead_roi_queue) >= self.window_size:
                # Draw the Face ROI on the canvas
                # No need to Draw the Face ROI, because the frame already include the face ROI

                # Update the Chart Data
                # Check if 'heartbeat_signal' is more than 10s, if yes, pop the oldest frame
                if len(self.heartbeat_signal) > self.fps * 10:
                    self.heartbeat_signal.pop(0)
                # Check if 'forehead_roi_for_heart_beat_queue' is more than 10s, if yes, pop the oldest frame  
                if len(self.forehead_roi_for_heart_beat_queue) > self.fps * 10:
                    self.forehead_roi_for_heart_beat_queue.pop(0)
                # Check if 'getBpmList' is more than 10s, if yes, pop the oldest frame
                if len(self.getBpmList) > self.fps * 10:
                    self.getBpmList.pop(0)

                self.update_gui2()
            
            else:
                logStr = f"Face ROI Data is not bigger than window size. Get: {len(self.face_roi_queue)} / {self.window_size}"
                self.update_log(logStr)
                print(logStr)

        self.root.after(1, self.update_gui)

    # Update the heart rate every 20 frames
    def update_gui2(self):
        if self.heartBeatUpdateFrameCounter >= 20:
            # If the queue is not empty
            if len(self.face_roi_queue) >= self.window_size and len(self.forehead_roi_queue) >= self.window_size:
                # Calculate and update heart rate
                current_hr = self.calculate_heart_rate()
                self.getBpmList.append(current_hr)
                if len(self.getBpmList) == 1:
                    self.getBpmList.append(current_hr)

                self.heartBeatUpdateFrameCounter = 0

    def calculate_heart_rate(self):
        # Define the bandpass filter parameters
        lowcut = 0.75  # Lower cutoff frequency in Hz
        highcut = 2.5  # Upper cutoff frequency in Hz
                    
        fs = self.fps  # Sampling frequency, adjust accordingly based on your data

        # Normalize the cutoff frequencies
        low = lowcut / (0.5 * fs)
        high = highcut / (0.5 * fs)

        # Design a Butterworth bandpass filter
        b, a = butter(N=1, Wn=[low, high], btype="band")

        forehead_rois_np = np.array(
            self.forehead_roi_for_heart_beat_queue[-self.window_size :]
        )

        # Calculate average color over time for each pixel in the face ROI
        average_color = np.mean(forehead_rois_np, axis=(1, 2), dtype=np.float64)

        green_channel = average_color[:, 1]

        # Apply the bandpass filter to the green channel data
        green_channel = lfilter(b, a, green_channel)

        predicted_hr = self.get_heart_rate(green_channel)

        return predicted_hr

    def get_heart_rate(self, channel):
        sampling_rate = self.fps
        hr = None

        # Perform FFT on the channel
        fft_result = fft(channel)
        freq = fftfreq(len(fft_result)) * sampling_rate

        # Convert the frequency domain to below 2.5Hz and above 0.75Hz
        fft_result = fft_result[(freq < 2.5) & (freq > 0.75)]
        freq = freq[(freq < 2.5) & (freq > 0.75)]

        # Normalize the signals
        fft_result = fft_result / np.max(np.abs(fft_result))

        # Find peaks in the magnitude spectrum within the specified range
        peaks, _ = find_peaks(np.abs(fft_result), height=0)

        # Sort the peaks by amplitude in descending order
        sorted_peaks = sorted(
            peaks, key=lambda x: np.abs(fft_result[x]), reverse=True
        )

        # Store the frequency and amplitude of the first peak in each window
        if sorted_peaks:
            peak_frequency = freq[sorted_peaks[0]]
            print(f"{peak_frequency=}")
            hr = peak_frequency * 60
            self.all_predicted_hrs.append(hr)
        else:
            print("No peaks found. Using the last saved heart rate.")
            # Handle the case when self.all_predicted_hrs is empty
            if self.all_predicted_hrs:
                self.all_predicted_hrs.append(self.all_predicted_hrs[-1])
                hr = self.all_predicted_hrs[-1]
            else:
                # Provide a default value or continue with a different logic
                # For example, append a default heart rate or None
                default_hr = 60  # or some default value
                self.all_predicted_hrs.append(default_hr)
                hr = default_hr

        return hr

    def update_chart(self):
        # Clear previous annotations
        for txt in self.ax_time.texts:
            txt.remove()
        for txt in self.ax_hr.texts:
            txt.remove()

        # Update the time-domain plot
        self.x_data_time = np.arange(len(self.heartbeat_signal))
        self.y_data_time = self.heartbeat_signal
        self.line_time.set_data(self.x_data_time, self.y_data_time)

        # Update the heart rate plot
        self.x_data_hr = np.arange(len(self.getBpmList))
        self.x_data_hr =  self.new_x_data_hr[:len(self.getBpmList)]
        self.y_data_hr = self.getBpmList
        print(self.x_data_hr)
        print(self.y_data_hr)
        self.line_hr.set_data(self.x_data_hr, self.y_data_hr)

        # Add text annotations for the latest data
        if len(self.y_data_time) > 0:
            self.ax_time.text(self.x_data_time[-1], self.y_data_time[-1], f'{self.y_data_time[-1]:.2f}',
                            color='red', fontsize=10)
        if len(self.y_data_hr) > 0:
            self.ax_hr.text(self.x_data_hr[-1], self.y_data_hr[-1], f'{self.y_data_hr[-1]:.2f}',
                            color='red', fontsize=10)

        # Rescale the plots
        self.ax_time.relim()
        self.ax_time.autoscale_view()
        self.ax_hr.relim()
        self.ax_hr.autoscale_view()

        # Redraw the canvas
        self.ChartCanvas.draw()

        # Timer for updating the chart
        self.root.after(1000, self.update_chart)  # Update every second

    def update_log(self, data):
        """
        Add new Message to the log panel with the received data
        :param:
        data : String
        Example: self.update_log(message.payload)
        """
        # Get now Date Time in format YYYY-MM-DD HH:MM:SS:MS
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.log.configure(state="normal")
        self.log.insert(tk.END, f"{now} : Received: {data}\n")
        self.log.configure(state="disabled")
        self.log.see(tk.END)

    def start_listening(self):
        # Save the current settings to the config file
        self.save_settings_to_config()
        # Start the MQTT client with new settings
        self.start_mqtt_client()
        # Set the flag to True
        self.is_listening = True
        # Update widget states
        self.update_widget_states()

    def stop_listening(self):
        # Stop the MQTT client
        self.stop_mqtt_client()
        # Set the flag to False
        self.is_listening = False
        # Update widget states
        self.update_widget_states()

    def update_widget_states(self):
        # Enable or disable widgets based on the flag
        state = tk.DISABLED if self.is_listening else tk.NORMAL
        self.ip_entry.configure(state=state)
        self.port_entry.configure(state=state)
        self.sub_entry.configure(state=state)
        self.show_face_roi_checkbox.configure(state=state)
        self.start_button.configure(
            state=tk.DISABLED if self.is_listening else tk.NORMAL
        )
        self.stop_button.configure(
            state=tk.NORMAL if self.is_listening else tk.DISABLED
        )

    def save_settings_to_config(self):
        # Gather the current settings
        settings = {
            "MQTT_SERVER_ADDRESS": self.ip_entry.get(),
            "MQTT_SERVER_PORT": self.port_entry.get(),
            "MQTT_SERVER_SUBSCRIBE": self.sub_entry.get(),
            "SHOW_FACE_ROI": self.show_face_roi_var.get(),
        }
        # Write settings to the config file
        with open("setting.config", "w") as config_file:
            for key, value in settings.items():
                config_file.write(f"{key} = {value}\n")


# Create the main window
root = tk.Tk()
app = ImageStreamApp(root)
root.mainloop()
