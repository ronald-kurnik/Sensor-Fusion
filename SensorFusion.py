import random
import time
import matplotlib.pyplot as plt

class SimulatedSensor:
    """
    A base class to simulate a sensor with a noisy reading.
    """
    def __init__(self, noise_level):
        self.noise_level = noise_level

    def get_reading(self, true_value):
        """Returns the true value plus random noise."""
        return true_value + random.uniform(-self.noise_level, self.noise_level)

class KalmanFilter:
    """
    A simplified 1D Kalman Filter for fusing sensor data.
    It predicts a state and then updates it with a new measurement.
    """
    def __init__(self, initial_estimate, initial_error_covariance, process_noise, measurement_noise):
        self.estimate = initial_estimate
        self.error_covariance = initial_error_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, motion):
        """
        Prediction step: Predict the new estimate based on motion.
        Motion could be based on an accelerometer or other internal model.
        """
        # New estimate is the old estimate plus the motion
        self.estimate += motion
        # The uncertainty (error_covariance) grows with process noise
        self.error_covariance += self.process_noise

    def update(self, measurement):
        """
        Update step: Correct the estimate using a new measurement.
        The correction is based on the Kalman Gain, which weighs the
        new measurement against the current estimate.
        """
        # Calculate the Kalman Gain
        # The gain determines how much to trust the new measurement vs. our current estimate
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_noise)
        
        # Update the estimate by adding a correction based on the measurement
        self.estimate += kalman_gain * (measurement - self.estimate)
        
        # Update the error covariance
        self.error_covariance = (1 - kalman_gain) * self.error_covariance

def simulate_sensor_fusion():
    """
    Main function to simulate the fusion of GPS and accelerometer data.
    This version includes a visual plot of the results.
    """
    # Initialize a true position and velocity
    true_position = 0.0
    true_velocity = 1.0  # meters per second

    # Initialize simulated sensors with different noise levels
    gps_sensor = SimulatedSensor(noise_level=1.5)  # High noise, but gives absolute position
    accelerometer_sensor = SimulatedSensor(noise_level=0.5)  # Lower noise, but measures motion

    # Initialize the Kalman filter with an initial guess
    # Initial estimate of position, and initial uncertainty
    kf = KalmanFilter(initial_estimate=0.0, initial_error_covariance=1.0,
                      # How much uncertainty grows with each step (process model)
                      process_noise=0.1,
                      # How much uncertainty is in the GPS measurement
                      measurement_noise=gps_sensor.noise_level**2)

    print("Step | True Pos | GPS Reading | Accel Motion | KF Estimate")
    print("-" * 55)

    num_steps = 20
    # Lists to store data for plotting
    steps = []
    true_positions = []
    gps_readings = []
    kf_estimates = []

    for step in range(1, num_steps + 1):
        # Move the true position
        true_position += true_velocity
        
        # Simulate sensor readings
        gps_reading = gps_sensor.get_reading(true_position)
        # Accelerometer measures change in position (velocity * time step)
        accel_motion = accelerometer_sensor.get_reading(true_velocity)
        
        # Perform Kalman Filter prediction and update steps
        kf.predict(accel_motion)
        kf.update(gps_reading)

        # Store data for plotting
        steps.append(step)
        true_positions.append(true_position)
        gps_readings.append(gps_reading)
        kf_estimates.append(kf.estimate)

        # Print results for each step
        print(f"{step:4d} | {true_position:8.2f} | {gps_reading:11.2f} | {accel_motion:12.2f} | {kf.estimate:11.2f}")
        time.sleep(0.1)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(steps, true_positions, label='True Position', color='black', linewidth=3)
    plt.plot(steps, gps_readings, 'o', label='GPS Reading (Noisy)', color='blue')
    plt.plot(steps, kf_estimates, label='Kalman Filter Estimate', color='red', linestyle='--', linewidth=2)
    
    plt.title('Sensor Fusion with a Kalman Filter')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    simulate_sensor_fusion()