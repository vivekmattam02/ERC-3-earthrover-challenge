"""
EarthRover Interface

Interface to communicate with FrodoBots EarthRover Zero via the SDK.

The SDK runs as a local server on http://localhost:8000.
Before using this interface, start the SDK server:
    cd earth-rovers-sdk && hypercorn main:app --reload

Usage:
    from earthrover_interface import EarthRoverInterface

    rover = EarthRoverInterface()
    rover.connect()

    # Get sensor data
    image = rover.get_camera_frame()
    lat, lon, heading = rover.get_pose()

    # Send control
    rover.send_control(linear=0.5, angular=0.0)

    # Stop
    rover.stop()
"""

import base64
import math
import time
import requests
import numpy as np
from io import BytesIO
from PIL import Image


class EarthRoverInterface:
    """
    Interface to EarthRover SDK server.
    """

    def __init__(self, base_url="http://localhost:8000", timeout=5.0):
        """
        Initialize interface.

        Args:
            base_url: URL of the SDK server (default localhost:8000)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.connected = False

        # Cache for latest data
        self._last_data = None
        self._last_data_time = 0
        self._data_cache_duration = 0.05  # 50ms cache

        # Last known pose (for interpolation if needed)
        self._last_lat = None
        self._last_lon = None
        self._last_heading = None

    def connect(self):
        """
        Test connection to the SDK server.

        Returns:
            True if connected, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/data",
                timeout=self.timeout
            )
            if response.status_code == 200:
                self.connected = True
                print(f"Connected to EarthRover SDK at {self.base_url}")
                data = response.json()
                print(f"  Battery: {data.get('battery', 'N/A')}%")
                print(f"  GPS: ({data.get('latitude', 'N/A')}, {data.get('longitude', 'N/A')})")
                print(f"  Orientation: {data.get('orientation', 'N/A')}")
                return True
            else:
                print(f"SDK server returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to SDK server at {self.base_url}")
            print("Make sure the SDK server is running:")
            print("  cd earth-rovers-sdk && hypercorn main:app --reload")
            return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def get_camera_frame(self):
        """
        Get the latest front camera frame.

        Returns:
            numpy array of shape (H, W, 3) with RGB values 0-255,
            or None if failed
        """
        try:
            response = requests.get(
                f"{self.base_url}/v2/front",
                timeout=self.timeout
            )

            if response.status_code != 200:
                print(f"Camera request failed: {response.status_code}")
                return None

            data = response.json()
            frame_b64 = data.get('front_frame')

            if frame_b64 is None:
                print("No front_frame in response")
                return None

            # Decode base64 to image
            image_bytes = base64.b64decode(frame_b64)
            image = Image.open(BytesIO(image_bytes))

            # Convert to RGB numpy array
            image = image.convert('RGB')
            frame = np.array(image, dtype=np.uint8)

            return frame

        except Exception as e:
            print(f"Error getting camera frame: {e}")
            return None

    def get_data(self, use_cache=True):
        """
        Get all sensor data from the robot.

        Args:
            use_cache: If True, return cached data if recent enough

        Returns:
            dict with sensor data, or None if failed
        """
        # Check cache
        if use_cache and self._last_data is not None:
            age = time.time() - self._last_data_time
            if age < self._data_cache_duration:
                return self._last_data

        try:
            response = requests.get(
                f"{self.base_url}/data",
                timeout=self.timeout
            )

            if response.status_code != 200:
                print(f"Data request failed: {response.status_code}")
                return self._last_data  # Return stale data

            data = response.json()

            # Update cache
            self._last_data = data
            self._last_data_time = time.time()

            return data

        except Exception as e:
            print(f"Error getting data: {e}")
            return self._last_data

    def get_pose(self):
        """
        Get current position and heading.

        Returns:
            (latitude, longitude, heading) where heading is in radians.
            Returns (None, None, None) if failed.
        """
        data = self.get_data()

        if data is None:
            return self._last_lat, self._last_lon, self._last_heading

        lat = data.get('latitude')
        lon = data.get('longitude')
        orientation = data.get('orientation')  # 0-360 degrees

        if lat is None or lon is None:
            return self._last_lat, self._last_lon, self._last_heading

        # Convert orientation from degrees to radians
        # SDK returns 0-360, we need radians where 0 = North, pi/2 = East
        if orientation is not None:
            # Assuming orientation is compass heading in degrees
            # 0 = North, 90 = East, 180 = South, 270 = West
            heading = math.radians(orientation)
        else:
            heading = self._last_heading

        # Update cache
        self._last_lat = lat
        self._last_lon = lon
        self._last_heading = heading

        return lat, lon, heading

    def send_control(self, linear, angular, lamp=0):
        """
        Send velocity command to the robot.

        Tries the browser-relay path (/control) first; if that fails, falls
        back to the direct RTM path (/control-legacy) which doesn't depend on
        a live browser/pyppeteer session.

        Args:
            linear: Forward/backward speed (-1 to 1)
            angular: Rotation speed (-1 to 1), positive = left
            lamp: Lamp control (0 = off, 1 = on)

        Returns:
            True if successful, False otherwise
        """
        # Clamp values
        linear = max(-1.0, min(1.0, float(linear)))
        angular = max(-1.0, min(1.0, float(angular)))
        lamp = int(lamp)

        payload = {
            "command": {
                "linear": linear,
                "angular": angular,
                "lamp": lamp
            }
        }

        # Try browser-relay path first (lower latency when browser is alive)
        try:
            response = requests.post(
                f"{self.base_url}/control",
                json=payload,
                timeout=1.0
            )
            if response.status_code == 200:
                return True
        except Exception:
            pass

        # Fallback: direct RTM (no browser/pyppeteer dependency)
        try:
            response = requests.post(
                f"{self.base_url}/control-legacy",
                json=payload,
                timeout=2.0
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending control (both paths failed): {e}")
            return False

    def stop(self):
        """Stop the robot (send zero velocities)."""
        return self.send_control(0.0, 0.0)

    def get_battery(self):
        """Get battery percentage."""
        data = self.get_data()
        if data:
            return data.get('battery')
        return None

    def get_imu(self):
        """
        Get IMU data (accelerometer, gyroscope, magnetometer).

        Returns:
            dict with 'accels', 'gyros', 'mags' arrays, or None if failed
        """
        data = self.get_data()
        if data is None:
            return None

        return {
            'accels': data.get('accels', []),
            'gyros': data.get('gyros', []),
            'mags': data.get('mags', [])
        }

    # ------------------------------------------------------------------
    # Mission API
    # ------------------------------------------------------------------

    def start_mission(self):
        """Start mission and return checkpoint list sorted by sequence.

        Returns:
            list of checkpoint dicts (keys: id, sequence, latitude, longitude),
            or None on failure.
        """
        try:
            response = requests.post(
                f"{self.base_url}/start-mission",
                timeout=self.timeout,
            )
            if response.status_code == 200:
                data = response.json()
                checkpoints = (
                    data.get("checkpoints_list", {})
                    .get("checkpoints_list", [])
                )
                checkpoints.sort(key=lambda x: x["sequence"])
                return checkpoints
            print(f"[mission] start_mission HTTP {response.status_code}: {response.text[:200]}")
            return None
        except Exception as exc:
            print(f"[mission] start_mission error: {exc}")
            return None

    def checkpoint_reached(self):
        """Report current checkpoint reached.

        Returns:
            (True, next_sequence: int)   — server accepted the report
            (False, reason: str)         — server rejected (too far, etc.)
        """
        try:
            response = requests.post(
                f"{self.base_url}/checkpoint-reached",
                json={},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                data = response.json()
                return True, data.get("next_checkpoint_sequence")
            data = response.json()
            detail = data.get("detail", {})
            if isinstance(detail, dict):
                dist = detail.get("proximate_distance_to_checkpoint", "?")
                return False, f"too_far({dist}m)"
            return False, str(detail)
        except Exception as exc:
            return False, str(exc)

    def get_checkpoints_list(self):
        """Get current checkpoint list and latest_scanned_checkpoint.

        Returns:
            dict from SDK, or None on failure.
        """
        try:
            response = requests.get(
                f"{self.base_url}/checkpoints-list",
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    def end_mission(self):
        """Emergency: end mission and release the bot.

        Returns:
            True if the server acknowledged, False otherwise.
        """
        try:
            response = requests.post(
                f"{self.base_url}/end-mission",
                timeout=self.timeout,
            )
            return response.status_code == 200
        except Exception:
            return False


    def start_intervention(self):
        """Report operator intervention start to the SDK.

        Returns:
            (True, payload) on success, (False, reason) on failure.
        """
        try:
            response = requests.post(
                f"{self.base_url}/interventions/start",
                json={},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return True, response.json()
            return False, response.text[:200]
        except Exception as exc:
            return False, str(exc)

    def end_intervention(self):
        """Report operator intervention end to the SDK.

        Returns:
            (True, payload) on success, (False, reason) on failure.
        """
        try:
            response = requests.post(
                f"{self.base_url}/interventions/end",
                json={},
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return True, response.json()
            return False, response.text[:200]
        except Exception as exc:
            return False, str(exc)


# Quick test if run directly
if __name__ == "__main__":
    print("Testing EarthRover Interface...")
    print("=" * 60)

    rover = EarthRoverInterface()

    # Test connection
    print("\n[1] Testing connection...")
    if not rover.connect():
        print("\nCannot connect to SDK server.")
        print("To test without the robot, the SDK server must be running.")
        print("\nInterface code is ready. Start the SDK server and try again:")
        print("  cd earth-rovers-sdk && hypercorn main:app --reload")
        exit(1)

    # Test get_data
    print("\n[2] Testing get_data...")
    data = rover.get_data()
    if data:
        print(f"  Battery: {data.get('battery')}%")
        print(f"  Signal: {data.get('signal_level')}")
        print(f"  GPS Signal: {data.get('gps_signal')}")

    # Test get_pose
    print("\n[3] Testing get_pose...")
    lat, lon, heading = rover.get_pose()
    print(f"  Latitude: {lat}")
    print(f"  Longitude: {lon}")
    print(f"  Heading: {heading} rad ({math.degrees(heading) if heading else 'N/A'} deg)")

    # Test get_camera_frame
    print("\n[4] Testing get_camera_frame...")
    frame = rover.get_camera_frame()
    if frame is not None:
        print(f"  Frame shape: {frame.shape}")
        print(f"  Frame dtype: {frame.dtype}")
        print(f"  Frame range: [{frame.min()}, {frame.max()}]")
    else:
        print("  No frame received")

    # Test send_control (very small movement)
    print("\n[5] Testing send_control (stopping)...")
    success = rover.stop()
    print(f"  Stop command: {'OK' if success else 'FAILED'}")

    print("\n" + "=" * 60)
    print("EarthRover Interface test complete!")
