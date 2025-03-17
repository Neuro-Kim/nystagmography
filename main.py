import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from collections import deque
from scipy.signal import find_peaks
from scipy import signal

class NystagmusDetector:
    def __init__(self, history_size=100):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Eye landmark indices (defined in MediaPipe Face Mesh)
        # Left eye landmarks
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        # Right eye landmarks
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        # Left iris landmarks (provided in the 468 landmark model)
        self.LEFT_IRIS_INDICES = [474, 475, 476, 477]
        # Right iris landmarks
        self.RIGHT_IRIS_INDICES = [469, 470, 471, 472]
        
        # Face reference points for normalization (nose tip and forehead)
        self.NOSE_TIP = 4
        self.FOREHEAD = 10
        
        # Initialize variables for data recording
        self.history_size = history_size
        self.left_eye_x_rel = deque(maxlen=history_size)
        self.left_eye_y_rel = deque(maxlen=history_size)
        self.right_eye_x_rel = deque(maxlen=history_size)
        self.right_eye_y_rel = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        self.start_time = None
        
        # Current camera index
        self.camera_index = 0
        self.available_cameras = [0, 1, 2, 3]  # Camera indices to cycle through
        
        # Set up graph
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 8))
        self.line_left_x, = self.axs[0].plot([], [], 'r-', label='Left Eye X')
        self.line_right_x, = self.axs[0].plot([], [], 'b-', label='Right Eye X')
        self.line_left_y, = self.axs[1].plot([], [], 'r-', label='Left Eye Y')
        self.line_right_y, = self.axs[1].plot([], [], 'b-', label='Right Eye Y')
        
        self.axs[0].set_xlabel('Time (seconds)')
        self.axs[0].set_ylabel('Relative X Position')
        self.axs[0].set_title('Horizontal Eye Movement')
        self.axs[0].legend()
        
        self.axs[1].set_xlabel('Time (seconds)')
        self.axs[1].set_ylabel('Relative Y Position')
        self.axs[1].set_title('Vertical Eye Movement')
        self.axs[1].legend()
        
        self.fig.tight_layout()
        
    def calculate_relative_position(self, landmark_x, landmark_y, face_width, face_height, face_center_x, face_center_y):
        """Calculate position relative to face center and normalized by face size"""
        rel_x = (landmark_x - face_center_x) / face_width
        rel_y = (landmark_y - face_center_y) / face_height
        return rel_x, rel_y
        
    def process_frame(self, frame):
        if self.start_time is None:
            self.start_time = time.time()
            
        # Convert color for MediaPipe (BGR -> RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        
        # Process with Face Mesh
        results = self.face_mesh.process(frame_rgb)
        
        # Draw and analyze results
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Draw iris contours
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )
                
                # Calculate face measurements for normalization
                # Use specific face landmarks to determine face width and height
                face_width_landmarks = [234, 454]  # Left and right cheek
                face_height_landmarks = [10, 152]  # Forehead and chin
                
                face_width_points = [face_landmarks.landmark[idx] for idx in face_width_landmarks]
                face_height_points = [face_landmarks.landmark[idx] for idx in face_height_landmarks]
                
                face_width = abs(face_width_points[0].x - face_width_points[1].x) * w
                face_height = abs(face_height_points[0].y - face_height_points[1].y) * h
                
                # Get face center using nose tip
                nose_tip = face_landmarks.landmark[self.NOSE_TIP]
                face_center_x = nose_tip.x * w
                face_center_y = nose_tip.y * h
                
                # Calculate left iris center
                left_iris_landmarks = [face_landmarks.landmark[i] for i in self.LEFT_IRIS_INDICES]
                left_iris_x = np.mean([landmark.x for landmark in left_iris_landmarks]) * w
                left_iris_y = np.mean([landmark.y for landmark in left_iris_landmarks]) * h
                
                # Calculate right iris center
                right_iris_landmarks = [face_landmarks.landmark[i] for i in self.RIGHT_IRIS_INDICES]
                right_iris_x = np.mean([landmark.x for landmark in right_iris_landmarks]) * w
                right_iris_y = np.mean([landmark.y for landmark in right_iris_landmarks]) * h
                
                # Calculate relative positions
                left_iris_x_rel, left_iris_y_rel = self.calculate_relative_position(
                    left_iris_x, left_iris_y, face_width, face_height, face_center_x, face_center_y
                )
                
                right_iris_x_rel, right_iris_y_rel = self.calculate_relative_position(
                    right_iris_x, right_iris_y, face_width, face_height, face_center_x, face_center_y
                )
                
                # Draw circles at iris centers
                cv2.circle(frame, (int(left_iris_x), int(left_iris_y)), 3, (0, 255, 0), -1)
                cv2.circle(frame, (int(right_iris_x), int(right_iris_y)), 3, (0, 255, 0), -1)
                
                # Draw face center reference
                cv2.circle(frame, (int(face_center_x), int(face_center_y)), 5, (255, 0, 0), -1)
                
                # Record current time
                current_time = time.time() - self.start_time
                
                # Store data
                self.left_eye_x_rel.append(left_iris_x_rel)
                self.left_eye_y_rel.append(left_iris_y_rel)
                self.right_eye_x_rel.append(right_iris_x_rel)
                self.right_eye_y_rel.append(right_iris_y_rel)
                self.timestamps.append(current_time)
                
                # Display information on screen
                cv2.putText(frame, f"Left iris rel: ({left_iris_x_rel:.3f}, {left_iris_y_rel:.3f})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Right iris rel: ({right_iris_x_rel:.3f}, {right_iris_y_rel:.3f})", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Camera: {self.camera_index}", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Press 'q' to quit, 'c' to change camera", 
                            (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def update_plot(self, frame_idx):
        # Update graph
        if len(self.timestamps) > 0:
            timestamps_array = list(self.timestamps)
            
            # Update x-axis data
            self.line_left_x.set_data(timestamps_array, list(self.left_eye_x_rel))
            self.line_right_x.set_data(timestamps_array, list(self.right_eye_x_rel))
            
            # Update y-axis data
            self.line_left_y.set_data(timestamps_array, list(self.left_eye_y_rel))
            self.line_right_y.set_data(timestamps_array, list(self.right_eye_y_rel))
            
            # Adjust graph ranges
            if len(timestamps_array) > 1:
                for ax in self.axs:
                    ax.relim()
                    ax.autoscale_view()
            
        return self.line_left_x, self.line_right_x, self.line_left_y, self.line_right_y
    
    def cycle_camera(self):
        """Switch to next available camera"""
        self.camera_index = (self.camera_index + 1) % len(self.available_cameras)
        return self.available_cameras[self.camera_index]
    
    def start_detection(self, initial_camera=0):
        self.camera_index = initial_camera
        cap = cv2.VideoCapture(self.available_cameras[self.camera_index])
        
        if not cap.isOpened():
            print("Failed to open camera.")
            return
        
        # Create animation
        ani = FuncAnimation(self.fig, self.update_plot, interval=30, blit=True)
        plt.ion()  # Interactive mode
        plt.show()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read from camera {self.camera_index}")
                # Try next camera
                next_camera = self.cycle_camera()
                cap.release()
                cap = cv2.VideoCapture(next_camera)
                if not cap.isOpened():
                    print(f"Failed to open camera {next_camera}")
                    continue
                
            processed_frame = self.process_frame(frame)
            
            cv2.imshow("Nystagmus Detection", processed_frame)
            plt.pause(0.001)  # Update graph
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Cycle to next camera
                next_camera = self.cycle_camera()
                cap.release()
                cap = cv2.VideoCapture(next_camera)
                print(f"Switched to camera {next_camera}")
                if not cap.isOpened():
                    print(f"Failed to open camera {next_camera}")
                    # Try to find any working camera
                    for i in range(len(self.available_cameras)):
                        test_cam = self.cycle_camera()
                        cap = cv2.VideoCapture(test_cam)
                        if cap.isOpened():
                            print(f"Found working camera {test_cam}")
                            break
        
        cap.release()
        cv2.destroyAllWindows()
        plt.close()
        
    def analyze_nystagmus(self):
        """Nystagmus analysis function
        
        This function analyzes collected eye movement data to calculate nystagmus characteristics:
        - Frequency: The frequency of eye movements (oscillations per second)
        - Amplitude: The magnitude of eye movements
        - Direction: Whether primarily horizontal, vertical, or rotational
        """
        if len(self.timestamps) < 10:
            print("Not enough data for analysis.")
            return
        
        # Perform FFT for frequency analysis
        # Prepare for data resampling
        x_data = np.array(list(self.left_eye_x_rel))
        t_data = np.array(list(self.timestamps))
        
        # Resample to uniform time intervals
        t_resampled = np.linspace(t_data.min(), t_data.max(), len(t_data))
        x_resampled = np.interp(t_resampled, t_data, x_data)
        
        # Remove trend (linear detrending)
        x_detrended = signal.detrend(x_resampled)
        
        # Calculate FFT
        sampling_rate = len(t_resampled) / (t_resampled.max() - t_resampled.min())  # Calculate sampling rate
        fft_result = np.fft.rfft(x_detrended)
        freqs = np.fft.rfftfreq(len(x_detrended), 1/sampling_rate)
        
        # Calculate spectrum magnitude
        magnitude = np.abs(fft_result)
        
        # Find peaks (nystagmus frequency)
        peaks, _ = find_peaks(magnitude, height=np.std(magnitude))
        
        if len(peaks) > 0:
            # Frequency (Hz) of the largest peak
            dominant_freq_idx = peaks[np.argmax(magnitude[peaks])]
            dominant_freq = freqs[dominant_freq_idx]
            
            # Calculate amplitude
            amplitude = np.std(x_detrended)
            
            # Output results
            print(f"\nNystagmus Analysis Results:")
            print(f"Dominant Frequency: {dominant_freq:.2f} Hz")
            print(f"Average Amplitude: {amplitude:.4f} (relative to face size)")
            
            # Estimate nystagmus type
            if dominant_freq < 2:
                print("Low-frequency nystagmus: Possibly vestibular nystagmus")
            elif dominant_freq >= 2 and dominant_freq <= 5:
                print("Mid-frequency nystagmus: Possibly congenital nystagmus")
            else:
                print("High-frequency nystagmus: Possibly drug-induced or central nystagmus")
                
            # Display additional graphs
            plt.figure(figsize=(12, 6))
            
            plt.subplot(2, 1, 1)
            plt.plot(t_resampled, x_detrended)
            plt.title('Eye Movement After Detrending')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Position (relative)')
            
            plt.subplot(2, 1, 2)
            plt.plot(freqs, magnitude)
            plt.axvline(x=dominant_freq, color='r', linestyle='--', label=f'{dominant_freq:.2f} Hz')
            plt.legend()
            plt.title('Frequency Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.xlim(0, 15)  # Display 0-15Hz range
            
            plt.tight_layout()
            plt.show()
        else:
            print("No distinct nystagmus pattern detected.")


def main():
    detector = NystagmusDetector(history_size=300)
    print("Starting nystagmus detection program...")
    print("Press 'q' to quit and 'c' to cycle between cameras.")
    detector.start_detection()
    detector.analyze_nystagmus()

if __name__ == "__main__":
    main()