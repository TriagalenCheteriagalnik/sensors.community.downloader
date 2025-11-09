import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import numpy as np
from scipy import stats
import warnings
import io  # Import io for in-memory file handling
from timezonefinderL import TimezoneFinder # <-- Kept as requested
import pytz
import logging  # --- Import logging
import sys      # --- Import sys for console output
import shutil   # --- Import for disk space check
import psutil   # --- Import for RAM check
import time     # --- Import for polite download delay
import random   # --- NEW: Import for jitter

# ==============================================================================
# Automatic sensor.community file downloader & processor
# ==============================================================================

# ==============================================================================
# 1. USER CONFIGURATION: PLEASE EDIT THESE VALUES
# ==============================================================================

# A list of your sensor IDs (as strings). You may use one or multiple IDs, separated by a comma.
# You may get these by going on sensors.community and clicking on any sensor while in PM2.5 mode.
SENSOR_IDS_TO_DOWNLOAD = [
    "SDS011 ID HERE", "SDS011 ID HERE"
]

# This list should only contain the sensor types for your *main* data (e..g, "sds011" for particles).
# Do NOT add humidity sensors here. The humidity sensor download (Phase 2b) specifically handles "bme280".
# Add the humidity sensors in the pairing list only. 
SENSOR_TYPES_TO_TRY = [
    "sds011"
]

# --- RECOMMENDED: Pair SDS011 air quality sensors with BME280 humidity sensors ---
# This accounts for the reduced sensor accuracy at high humidity.
# Required if you wish to do humidity sensor adjustment (see below)
# You may get the humidity sensor IDs by going on sensors.community and clicking on any sensor while in humidity mode.
# Ideally, the sensors should be co-located, i.e. in the same physical location. The farther way the humidity sensor 
# is from the air quality sensor, the less accurate this is.
# The humidity filter will ONLY apply data from the paired sensor and not any other sensors.
# If an SDS011 sensor is in SENSOR_IDS_TO_DOWNLOAD but *not* in this
# map, it will *not* be filtered for humidity.
# One or more pairs may be added. Several air quality sensors may be linked to a single humidity sensor,
# though this is not best practice.
# Format: { "sds011_sensor_id": "bme280_sensor_id" }

SENSOR_PAIRINGS = {
    "SDS011 ID HERE": "BME280 ID HERE",
    "SDS011 ID HERE": "BME280 ID HERE"
    # Add more pairings here as needed
}

# Your desired date range (inclusive)
# Example: START_DATE = "2024-01-01"
# Example: END_DATE = "2024-05-31"
START_DATE = "2025-01-01"
END_DATE = "2025-01-02"

# --- Download & Politeness settings (AKA how to avoid getting banned for using this script) ---

# ALWAYS REMEMBER: You are querying third-party websites and could potentially cause a lot of requests and use significant bandwidth.
# Them granting us access to this data is not a right, it is a privilege. They do this for free and they don't have to.
# Please, do not download more data than is necessary and ALWAYS properly configure below settings before use. 
# Those that do not risk getting rate limited or banned from accessing their servers. 
# We should treat their servers with due respect. ;) 

# --- Polite User-Agent ---
# This string identifies your script to the sensor.community servers.
# It is good practice and shows you are a researcher, not an anonymous bot,
# which reduces the chance of being rate-limited or banned.
# Default: "Academic-Research-Client"
USER_AGENT = "Academic-Research-Client"

# --- "From" Header (Highly Recommended) ---
# This provides the server admin with a contact email if your script
# causes issues. It is the "gold standard" of polite scraping.
# Leave blank to disable.
FROM_EMAIL = "YOUR EMAIL HERE" # e.g., "your-email@your-institution.edu" - PLEASE CHANGE THIS

# --- Polite Download Delay (seconds) ---
# This adds a pause between *every single file request* to avoid 
# being rate-limited or banned by the server due to spamming requests.
# 1.0 = Very safe 1-second delay (RECOMMENDED FOR LARGE "DAILY" JOBS)
# 0.5 = (Default) Half-second delay
# 0.2 = Aggressive 0.2-second delay (not recommended for large jobs)
POLITE_DOWNLOAD_DELAY_SECONDS = 0.5

# --- Polite Delay Jitter (seconds) ---
# Adds a *random* extra delay (0 to X seconds) to each request.
# Many servers strongly dislike robotic requests that repeat on each tick.
# This helps alleviate this concern. (e.g., 1.0s delay + 0.5s jitter = 1.0s to 1.5s total wait)
POLITE_DOWNLOAD_DELAY_JITTER_SECONDS = 0.5 # Default: 0.5s

# --- Robust Retry Settings ---
# Settings for handling failed downloads (e.g., network errors, 503 server errors)
# This prevents partial data by retrying the same file.
# After this the script skips the file and logs a critical error, but continues with the next one.
MAX_RETRIES = 3 # How many times to retry a single failed file download.

RETRY_INITIAL_DELAY_SECONDS = 5 # How long to wait for the first retry.
# (Wait time doubles each retry: e.g., 5s, 10s, 20s until MAX_RETRIES is hit.)

# --- Warning for large jobs (requests) ---
# If the total sensor-days (sensors * days) exceeds this, the user will be
# shown a warning and asked to confirm before proceeding.
# This helps prevent accidental MemoryErrors on large jobs, as well as excessive queries to the server.
SENSOR_DAY_WARNING_THRESHOLD = 1000

# ----------------------------------------------------------------------------------------
# --- OPTIONAL: Download Time Estimation Enhancer ---
# This time, in seconds, is added for the purposes of download time estimation to the 
# polite download delay. It doesn't delay the download.
# In fact, this doesn't actually change anything. It just helps keep the download time estimator accurate
# by accounting for the latency in actually accessing and downloading the file from the server.
# It is only recommended that you change it if you find the download time estimation to be inaccurate.
# Default and recommended value: 0.5
ESTIMATED_REQUEST_OVERHEAD_SECONDS = 0.5

# ----------------------------------------------------------------------------------------
# ---  k-Köhler Humidity Correction Settings --------------------------------
# This factors in humidity - a crucial factor in PM2.5 and PM10 measurements. 
# You NEED to pair air quality sensors with humidity sensors in the settings above for this to work.
# Implements a 3-tiered hybrid logic based on published recommendations.
# 1. Below LOWER_BOUND: Data is used AS-IS (uncorrected).
#    (Studies show sensors may *underestimate* at low RH, so correction is not applied).
# 2. Between LOWER and UPPER_BOUND: k-Köhler correction is applied.
# 3. At or Above UPPER_BOUND: Data is DISCARDED (assumed fog/saturation).
#
RH_CORRECTION_LOWER_BOUND = 70.0  # Recommended & Default: 70.0
RH_CORRECTION_UPPER_BOUND = 98.0  # Recommended & Default: 98.0 (Model is unstable near 100%)

# --- k (kappa) Parameter ---
# This is the citable hygroscopicity parameter for the k-Köhler correction model.
# 0.27 = (Default) Citable global mean for "continental" aerosol (Pringle et al., 2010).
# 0.20 = Represents more fresh, sooty urban aerosol.
# 0.40 = Represents more aged, sulfate-heavy aerosol.
KAPPA_PARAMETER = 0.27

# --- (DEPRECATED) ---
# The settings below are no longer used and are kept for compatibility checks.
# The logic is now handled by the 3-tiered system above.
HUMIDITY_THRESHOLD = 0.0 # No longer used
HUMIDITY_TIME_WINDOW_MINUTES = 0 # No longer used, merge is done on full data

# ----------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------
# --- OPTIONAL: Spike removal ---

# Spike Filter Multiplier (factor)
#   Meant to remove very incidental and localized factors. (eg. diesel truck, cooking, fireworks near sensor)
#   Also removes statistical outliers. (Though a better way would be a rolling median)
#   Data is flagged as a "potential spike" if it is this many
#   times greater than the previous reading (within the time gap below) and not present in any other sensor.
#   Example: With a value of 10, if a single sensor reading is 3, the following cannot be 30 or greater,
#   unless another sensor also has a "spike" reading.
#   "Spike" data is discarded before analysis and sent to a separate file.
#   Default is 10.0
#   Set to 0 to disable the spike filter entirely.
SPIKE_MULTIPLIER = 10.0

# --- NEW: Spike Filter Time Gap (User Request) ---
# A "potential spike" will only be flagged if the previous reading
# occurred within this many minutes.
# This prevents a sensor coming back online after a long gap (e.g., >16 mins)
# from having its first valid reading incorrectly flagged as a spike
# against a much older, different value.
SPIKE_MAX_TIME_GAP_MINUTES = 16.0

# ----------------------------------------------------------------------------------------
# --- OPTIONAL: Resource Estimation Settings ------------------------------------------------

# These are just rough estimates for your information. If the script finds you are going to go over the estimates,
# it will give you a warning, but attempt to run anyway unless you abort it.
# They show up when you run the script, as well as in the log file.
# Estimated default filesize (KB)
# This is the base assumption for data size (in KB) per sensor, per day.
# Should also take into account discarded data, the log file, and other things.
# It is used for pre-run RAM and Disk space checks.
# Default is 150 
KB_PER_SENSOR_DAY = 150 

# Multiplier for estimating peak RAM usage for in-memory processing. (factor)
# (e.g., 3.0 = 3x the base data size)
# Default is 15.0 - as some statistical functions require many times the filesize briefly during peak load
RAM_PROCESSING_MULTIPLIER = 15.0

# Safety margin added on top of estimating disk usage (factor)
# Example - 1.2 gives you a 20% safety buffer
# Default is 1.20 - a pessimistic buffer to put you on the safe side
DISK_SAFETY_MARGIN = 1.20

# ----------------------------------------------------------------------------------------
# --- IMPORTANT: File names and locations ------------------------------------------------

# Output directory (path)
# This typically works best if you give it an absolute path.
# Also, if you do give it an absolute path, put the lowercase letter "r" right before the opening quotation mark.
# Trust me on this, bro. Otherwise it often breaks.
# Example: r"C:\Users\User1\Documents\SensorData"
OUTPUT_DIR = r"PATH HERE"

# Master file name and location (path)
# This will be used as the *base name* and path for the master file AND the sensor-specific files
# e.g., "combined_sensor_data_part_1.csv"
# e.g., "combined_sensor_data_sensor_15194.csv"
# Example: OUTPUT_DIR = r"C:\Users\User1\Documents\SensorData\combined_sensor_data.csv"
FINAL_CSV_FILE = r"PATH HERE"

# ----------------------------------------------------------------------------------------


# ==============================================================================
# 2. SCRIPT LOGIC: DO NOT EDIT BELOW THIS LINE
# ==============================================================================

# Global list to store cleaning logs from the download phase
all_cleaning_logs = # 


def setup_logging():
    """Configures the logging module to output to file and console."""
    
    # --- Create output directory if it doesn't exist ---
    # This is a good place to do it, so the log file can be created.
    try:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Created output directory: {OUTPUT_DIR}") # Use print here as logging isn't set up
    except Exception as e:
        print(f"CRITICAL: Could not create output directory {OUTPUT_DIR}: {e}")
        print("Please check permissions or create the folder manually.")
        input("Press ENTER to exit...")
        sys.exit(1) # Exit the script
        
    log_file_path = os.path.join(OUTPUT_DIR, "download.log")
    
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=
    )

def format_bytes(bytes_val):
    """Converts a byte value into a human-readable string (KB, MB, GB)."""
    if bytes_val >= 1024**3:
        return f"{bytes_val / 1024**3:.2f} GB"
    if bytes_val >= 1024**2:
        return f"{bytes_val / 1024**2:.2f} MB"
    if bytes_val >= 1024:
        return f"{bytes_val / 1024:.2f} KB"
    return f"{bytes_val} bytes"

def format_time(seconds):
    """Converts seconds into a human-readable 'Xh Ym Zs' string."""
    if seconds <= 0:
        return "0s"
    
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    
    parts = # <-- FIX: Initialized list
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts: # Always show seconds if it's the only unit
        parts.append(f"{seconds}s")
        
    return " ".join(parts)

def check_disk_space(required_bytes):
    """Checks for available disk space against estimated requirements."""
    try:
        total, used, free = shutil.disk_usage(OUTPUT_DIR)
        
        available_formatted = format_bytes(free)
        required_formatted = format_bytes(required_bytes)
        
        logging.info(f"Available disk space on target drive: {available_formatted}")
        if free < required_bytes:
            logging.warning("--- LOW DISK SPACE WARNING ---")
            logging.warning(f"Available space ({available_formatted}) is less than the estimated required space ({required_formatted}).")
            logging.warning("Script will continue, but may fail if disk becomes full.")
        else:
            logging.info(f"Available disk space ({available_formatted}) meets estimated requirement ({required_formatted}).")
    except Exception as e:
        logging.error(f"Could not check disk space: {e}")

def check_ram_space(required_bytes):
    """Checks for available system RAM against estimated requirements."""
    try:
        mem = psutil.virtual_memory()
        available_bytes = mem.available
        
        available_formatted = format_bytes(available_bytes)
        required_formatted = format_bytes(required_bytes)
        
        logging.info(f"Available system RAM: {available_formatted}")
        if available_bytes < required_bytes:
            logging.warning("--- LOW RAM WARNING ---")
            logging.warning(f"Available RAM ({available_formatted}) is less than the estimated required RAM for data processing ({required_formatted}).")
            logging.warning("Script will continue, but may fail if system runs out of memory.")
        else:
            logging.info(f"Available RAM ({available_formatted}) meets estimated requirement ({required_formatted}).")
            
    except Exception as e:
        logging.error(f"Could not check RAM: {e}. (Is 'psutil' module installed?)")


def _log_time_estimate(total_requests):
    """Internal helper to log download time estimates."""
    # --- Calculate and display estimated time ---
    # Use both polite delay AND estimated overhead for a better prediction
    
    # ---  Use base delay + 1/2 jitter for estimation ---
    avg_jitter = POLITE_DOWNLOAD_DELAY_JITTER_SECONDS / 2.0
    avg_delay = POLITE_DOWNLOAD_DELAY_SECONDS + avg_jitter
    
    total_time_per_request = avg_delay + ESTIMATED_REQUEST_OVERHEAD_SECONDS
    estimated_seconds = total_requests * total_time_per_request
    estimated_time_str = format_time(estimated_seconds)
    
    time_message = (
        f"*** Estimated download time: ~{estimated_time_str} "
        f"({total_requests} requests @ ~{total_time_per_request:.1f}s/req)"
    )
    
    logging.info(time_message)
    print(time_message) # Print to console for visibility

def _process_file_content(content_bytes, sensor_id, date_str):
    """
    Internal helper to read CSV content into a DataFrame and log bad lines.
    Returns DataFrame or None if empty.
    """
    global all_cleaning_logs
    try:
        content_io = io.BytesIO(content_bytes)
        
        # --- Log skipped "bad" lines ---
        content_io.seek(0)
        total_lines = len(content_io.getvalue().splitlines())
        
        content_io.seek(0)
        df = pd.read_csv(content_io, delimiter=';', on_bad_lines='skip')
        
        num_read = len(df)
        num_skipped = 0

        if total_lines > 0:
            # -1 for the header row
            expected_data_rows = total_lines - 1
            num_skipped = expected_data_rows - num_read
            
            if num_skipped < 0:
                # This can happen if read_csv skips the header, etc.
                num_skipped = 0 

        if num_skipped > 0:
            all_cleaning_logs.append({
                'sensor_id': int(sensor_id),
                'date': pd.to_datetime(date_str), # Use date_str for grouping
                'bad_lines_skipped': num_skipped
            })
        
        if df.empty:
            return None
        return df
        
    except pd.errors.EmptyDataError:
        # This can happen if the file is 200 OK but completely empty
        return None
    except Exception as e:
        logging.error(f"  Error processing content for sensor {sensor_id} @ {date_str}: {e}")
        return None

def download_data_daily(session, sensor_ids, sensor_types, start_date, end_date):
    """
    Downloads data using DAILY archives.
    NEW: Includes robust retry logic.
    """
    logging.info(f"Downloading data for {len(sensor_ids)} sensor(s) using DAILY archives...")
    
    base_url = "https://archive.sensor.community"
    
    date_range = pd.date_range(start_date, end_date)
    all_dataframes = # <-- FIX: Initialized list
    
    total_requests = len(sensor_ids) * len(sensor_types) * len(date_range)
    logging.info(f"Total requests to make: {total_requests} (Delay: {POLITE_DOWNLOAD_DELAY_SECONDS}s + Jitter: {POLITE_DOWNLOAD_DELAY_JITTER_SECONDS}s)")
    _log_time_estimate(total_requests) # Log time estimate
    
    pbar = tqdm(total=total_requests, desc="Total Download Progress")
    

    for sensor_id in sensor_ids:
        for sensor_type in sensor_types:
            for date in date_range:
                pbar.set_description(f"Sensor {sensor_id} ({sensor_type}) @ {date.strftime('%Y-%m-%d')}")
                
                date_str = date.strftime("%Y-%m-%d")
                
                file_name = f"{date_str}_{sensor_type.lower()}_sensor_{sensor_id}.csv"
                file_url = f"{base_url}/{date_str}/{file_name}"
                
                # --- Retry Logic ---
                retries = 0
                success = False
                df = None
                
                while retries <= MAX_RETRIES and not success:
                    try:
                        r = session.get(file_url, timeout=30) # 30-second timeout
                        
                        # Case 1: File Not Found (404) - Not a failure, just doesn't exist.
                        if r.status_code == 404:
                            success = True # We "succeeded" in finding nothing.
                            df = None
                        
                        # Case 2: Success (200) - Process content
                        elif r.status_code == 200:
                            df = _process_file_content(r.content, sensor_id, date_str)
                            success = True
                        
                        # Case 3: Server Error (5xx, etc.) - Raise error to trigger retry
                        else:
                            r.raise_for_status() # Raises HTTPError

                    # This block catches 5xx errors, connection errors, timeouts
                    except (requests.exceptions.HTTPError, 
                            requests.exceptions.ConnectionError, 
                            requests.exceptions.Timeout) as e:
                        
                        retries += 1
                        if retries > MAX_RETRIES:
                            logging.error(f"  FAILED to download {file_url} after {MAX_RETRIES} retries: {e}")
                            df = None # Give up
                        else:
                            # Exponential backoff: 5s, 10s, 20s
                            wait_time = RETRY_INITIAL_DELAY_SECONDS * (2**(retries-1)) 
                            logging.warning(f"  Warning: Could not download {file_url}: {e}. Retrying in {wait_time}s... (Attempt {retries}/{MAX_RETRIES})")
                            time.sleep(wait_time)
                            
                    # This block catches other unexpected errors (e.g., pandas processing)
                    except Exception as e:
                         logging.error(f"  CRITICAL error processing {file_url}: {e}")
                         df = None
                         break # Break loop, don't retry on unknown error
                
                if df is not None:
                    all_dataframes.append(df)
                
                # --- END Retry Logic ---

    
                # --- MODIFIED: Add Jitter ---
                jitter = random.uniform(0, POLITE_DOWNLOAD_DELAY_JITTER_SECONDS)
                time.sleep(POLITE_DOWNLOAD_DELAY_SECONDS + jitter)
                pbar.update(1)
    
    pbar.close()
    return all_dataframes

def get_local_tz(lat, lon, tf_obj):
    """
    Finds the local timezone name from lat/lon.
    Defaults to 'EET' if not found (Request 5).
    """
    if pd.isna(lat) or pd.isna(lon):
        return 'EET' # Default as requested
    
    tz_name = tf_obj.timezone_at(lng=lon, lat=lat)
    
    if tz_name is None:
        return 'EET' # Default as requested
    
    return tz_name

def localize_group(group):
    """
    Applies timezone conversion to a DataFrame group.
    All timestamps must already be UTC-aware.
    """
    if group.empty:
        return group
        
    tz_name = group['local_tz'].iloc # <-- FIX: Added 
    try:
        # Convert UTC timestamp to the sensor's local timezone
        group['timestamp'] = group['timestamp'].dt.tz_convert(tz_name)
    except pytz.exceptions.UnknownTimeZoneError:
        # Fallback to EET if timezone name is invalid
        group['timestamp'] = group['timestamp'].dt.tz_convert('EET')
    return group

def clean_and_log_data(df, cleaning_logs_list, humidity_df=None, sensor_pairings=None,
                       spike_multiplier=10.0,
                       # --- Added time gap param (User Request) ---
                       spike_max_time_gap_minutes=16.0,
                       k_kappa=0.27, rh_low_bound=70.0, rh_high_bound=98.0):
    """
    Applies all data cleaning, k-Köhler humidity correction, and spike filtering.
    Generates a log of all excluded and corrected data.
    
    NEW: Implements 3-tiered humidity logic (Use As-Is, Correct, Discard).
    NEW: Adds log columns for corrected rows and fog-discarded rows.
    NEW: PM2.5 and PM10 columns are modified IN-PLACE.
    NEW: PM2.5_uncorrected and PM10_uncorrected are created and preserved.
    """
    logging.info("  Flagging data for cleaning...")
    
    # --- Request 2: Flag data for cleaning ---
    # Check if columns exist before trying to clean them
    if 'PM2.5' in df.columns:
        df['PM2.5'] = pd.to_numeric(df['PM2.5'], errors='coerce')
    if 'PM10' in df.columns:
        df['PM10'] = pd.to_numeric(df['PM10'], errors='coerce')

    # Flag 1: Negative values
    if 'PM2.5' in df.columns and 'PM10' in df.columns:
        df['negative_rows'] = ((df['PM2.5'] < 0) | (df['PM10'] < 0)).fillna(False)
    else:
        df['negative_rows'] = False # No PM data, so no negative rows
    
    # Flag 2: Values above 999
    if 'PM2.5' in df.columns and 'PM10' in df.columns:
        df['high_rows'] = ((df['PM2.5'] > 999) | (df['PM10'] > 999)).fillna(False)
    else:
        df['high_rows'] = False # No PM data
    
    # --- Flag 3: Spikes (Request 2 & 3: Configurable multiplier and acquittal logic) ---
    df = df.sort_values(by=['sensor_id', 'timestamp'])
    cols_to_drop_later = # <-- FIX: Initialized list
    
    if (spike_multiplier > 0 and 'PM2.5' in df.columns and 'PM10' in df.columns):
        logging.info(f"  Flagging potential spikes with multiplier {spike_multiplier}x...")
        df['pm25_prev'] = df.groupby('sensor_id')['PM2.5'].shift(1)
        df['pm10_prev'] = df.groupby('sensor_id')['PM10'].shift(1)
        
        # ---  Add timestamp shift for time-gap check (User Request) ---
        df['timestamp_prev'] = df.groupby('sensor_id')['timestamp'].shift(1)
        df['time_gap_minutes'] = (df['timestamp'] - df['timestamp_prev']).dt.total_seconds() / 60.0
        
        cols_to_drop_later.extend(['pm25_prev', 'pm10_prev', 'timestamp_prev', 'time_gap_minutes'])

        # A spike is > (multiplier) * the previous, and the previous value must exist and be > 0
        # --- MODIFIED: Added time-gap check (User Request) ---
        # The gap check ensures that a sensor coming online after a >16 min gap
        # does not trigger a false spike against its last known (and old) value.
        df['potential_spike_rows'] = (
            (
                (df['PM2.5'] > (df['pm25_prev'] * spike_multiplier)) & 
                df['pm25_prev'].notna() & 
                (df['pm25_prev'] > 0) &
                (df['time_gap_minutes'] <= spike_max_time_gap_minutes) 
            ) |
            (
                (df['PM10'] > (df['pm10_prev'] * spike_multiplier)) & 
                df['pm10_prev'].notna() & 
                (df['pm10_prev'] > 0) &
                (df['time_gap_minutes'] <= spike_max_time_gap_minutes) 
            )
        ).fillna(False)
        
        # ---  Acquittal Logic (Request 3) ---
        logging.info("  Applying spike acquittal logic...")
        # A spike is only a "true spike" if it's not corroborated by other sensors at the same time
        # 1. Count how many sensors spiked at the exact same timestamp
        df['timestamp_spike_count'] = df.groupby('timestamp')['potential_spike_rows'].transform('sum')
        
        # 2. A row is only flagged for *removal* if it's a potential spike AND it was the *only* one
        df['spike_rows'] = (df['potential_spike_rows'] == True) & (df['timestamp_spike_count'] == 1)
        
        cols_to_drop_later.extend(['potential_spike_rows', 'timestamp_spike_count'])
        
    else:
        if spike_multiplier <= 0:
            logging.info("  Spike filter disabled (multiplier is 0).")
        df['potential_spike_rows'] = False # Ensure column exists for later logic
        df['spike_rows'] = False # No PM data or filter disabled


    # --- *** NEW *** k-Köhler Humidity Correction & Filtering ---
    
    # 1. Create uncorrected columns to preserve raw data
    if 'PM2.5' in df.columns:
        df['PM2.5_uncorrected'] = df['PM2.5']
    if 'PM10' in df.columns:
        df['PM10_uncorrected'] = df['PM10']
    
    if (humidity_df is not None and not humidity_df.empty and
        'humidity' in humidity_df.columns and
        sensor_pairings is not None and sensor_pairings):
        
        logging.info("  Applying paired humidity merge...")
        
        humidity_df['humidity'] = pd.to_numeric(humidity_df['humidity'], errors='coerce')
        
        # 1. Get *all* humidity readings, keeping value and sensor_id
        # We merge all readings, not just high ones, to apply the 3-tier logic
        rh_df = humidity_df[['timestamp', 'humidity', 'sensor_id']].sort_values('timestamp')
        
        if not rh_df.empty:
            # Rename columns for clarity before merging
            rh_df.rename(columns={'humidity': 'humidity_rh_value', 'sensor_id': 'humidity_sensor_id_rh'}, inplace=True)
            rh_df['timestamp_rh'] = rh_df['timestamp']
            
            # 2. Map the paired sensor IDs to the main PM dataframe
            sensor_pairings_int_keys = {int(k): int(v) for k, v in sensor_pairings.items()}
            df['paired_humidity_sensor_id'] = df['sensor_id'].map(sensor_pairings_int_keys)

            # 3. Use merge_asof, matching on *both* time and the paired sensor ID
            # We use a 10-minute tolerance (as used in the old config) as a reasonable merge window.
            df = pd.merge_asof(
                df.sort_values('timestamp'),
                rh_df,
                on='timestamp',
                left_by='paired_humidity_sensor_id', 
                right_by='humidity_sensor_id_rh',
                direction='nearest', # Find the closest RH reading
                tolerance=pd.Timedelta(f'10 minutes') # Reasonable window
            )
        else:
            logging.warning("  Humidity sensor data was empty. Skipping humidity correction.")
            df['humidity_rh_value'] = np.nan
            df['timestamp_rh'] = pd.NaT
            df['paired_humidity_sensor_id'] = np.nan
            df['humidity_sensor_id_rh'] = np.nan
    else:
        if not sensor_pairings:
            logging.info("  No sensor pairings defined. Skipping humidity correction.")
        else:
            logging.info("  No humidity sensor data provided. Skipping humidity correction.")
        # Ensure the columns exist
        df['humidity_rh_value'] = np.nan
        df['timestamp_rh'] = pd.NaT
        df['paired_humidity_sensor_id'] = np.nan
        df['humidity_sensor_id_rh'] = np.nan

    # 4. --- Apply 3-Tiered k-Köhler Logic ---
    logging.info(f"  Applying k-Köhler correction (k={k_kappa}, Low={rh_low_bound}%, High={rh_high_bound}%)")
    
    # Define the new logic flags
    # We can only correct/discard if we have a valid humidity reading
    df['rows_corrected_k_kohler'] = (
        (df['humidity_rh_value'].notna()) &
        (df['humidity_rh_value'] >= rh_low_bound) &
        (df['humidity_rh_value'] < rh_high_bound)
    )
    
    # This is the new discard flag
    df['rows_discarded_fog'] = (
        (df['humidity_rh_value'].notna()) &
        (df['humidity_rh_value'] >= rh_high_bound)
    )
    
    # This is the old 'humidity_rows' flag, now repurposed for discarding
    df['humidity_rows'] = df['rows_discarded_fog']

    # 5. --- Apply the correction ---
    # Only apply to rows flagged for correction
    if df['rows_corrected_k_kohler'].any():
        # Calculate RH fraction
        rh_frac = df.loc[df['rows_corrected_k_kohler'], 'humidity_rh_value'] / 100.0
        
        # Calculate growth factor (GF_mass) based on k-Köhler theory
        # PM_corrected = PM_raw / (1 + k * ( (RH/100) / (1 - (RH/100)) ))
        growth_factor = 1.0 + k_kappa * (rh_frac / (1.0 - rh_frac))
        
        # Apply correction in-place to the main PM columns
        if 'PM2.5' in df.columns:
            df.loc[df['rows_corrected_k_kohler'], 'PM2.5'] = df.loc[df['rows_corrected_k_kohler'], 'PM2.5_uncorrected'] / growth_factor
        if 'PM10' in df.columns:
            df.loc[df['rows_corrected_k_kohler'], 'PM10'] = df.loc[df['rows_corrected_k_kohler'], 'PM10_uncorrected'] / growth_factor
            
    logging.info(f"    > Corrected {df['rows_corrected_k_kohler'].sum()} rows.")
    logging.info(f"    > Flagged {df['rows_discarded_fog'].sum()} rows for fog/saturation discard.")

    # --- Request 1 & 2: Create Cleaning Logs ---
    logging.info("  Generating cleaning logs...")
    
    # --- ADDED 'rows_corrected_k_kohler' to the log ---
    # 'humidity_rows' now represents 'rows_discarded_fog'
    log_cols = ['sensor_id', 'month', 'week', 'negative_rows', 'high_rows', 'spike_rows', 'humidity_rows', 'rows_corrected_k_kohler']
    
    # Group the newly flagged rows by sensor/month/week and sum them
    # Ensure all flag columns are numeric (bools) before summing
    # --- ADDED 'rows_corrected_k_kohler' ---
    flag_cols_to_sum = ['negative_rows', 'high_rows', 'spike_rows', 'humidity_rows', 'rows_corrected_k_kohler']
    for col in flag_cols_to_sum:
        if col in df.columns:
            df[col] = df[col].astype(bool)
        else:
            df[col] = False # Ensure column exists
        
    
    cleaning_log_df = df.groupby(['sensor_id', 'month', 'week'])[flag_cols_to_sum].sum().reset_index()

    # Process the 'bad_lines' log from downloads
    if cleaning_logs_list:
        bad_lines_log = pd.DataFrame(cleaning_logs_list)
        # Localize dates before creating month/week, default to EET
        try:
            bad_lines_log['date'] = bad_lines_log['date'].dt.tz_localize('UTC').dt.tz_convert('EET')
        except Exception:
            # Handle cases where tz_localize fails (e.g., already localized)
            try:
                bad_lines_log['date'] = bad_lines_log['date'].dt.tz_convert('EET')
            except TypeError: # Not a datetime object, localize first
                bad_lines_log['date'] = pd.to_datetime(bad_lines_log['date']).dt.tz_localize('EET')


        bad_lines_log['month'] = bad_lines_log['date'].dt.to_period('M')
        bad_lines_log['week'] = bad_lines_log['date'].dt.to_period('W')
        
        bad_lines_grouped = bad_lines_log.groupby(['sensor_id', 'month', 'week'])['bad_lines_skipped'].sum().reset_index()
        
        # Merge all logs together
        final_log_df = pd.merge(
            cleaning_log_df,
            bad_lines_grouped,
            on=['sensor_id', 'month', 'week'],
            how='outer'
        )
    else:
        final_log_df = cleaning_log_df
        final_log_df['bad_lines_skipped'] = 0

    # Fill NaNs with 0 for all log columns
    # --- ADDED 'rows_corrected_k_kohler' ---
    log_col_names = ['negative_rows', 'high_rows', 'spike_rows', 'humidity_rows', 'rows_corrected_k_kohler', 'bad_lines_skipped']
    final_log_df[log_col_names] = final_log_df[log_col_names].fillna(0).astype(int)
    
    # Rename 'humidity_rows' in the final log for clarity
    final_log_df.rename(columns={'humidity_rows': 'rows_discarded_fog'}, inplace=True)


    # --- Apply Cleaning ---
    logging.info("  Separating clean vs. discarded data...")
    original_rows = len(df)
    
    # --- Create a mask for all rows to be discarded ---
    # 'humidity_rows' is now the fog discard flag (>= 98%)
    flag_cols = ['negative_rows', 'high_rows', 'spike_rows', 'humidity_rows']
    discard_mask = df[flag_cols].any(axis=1)
    
    # 1. Create the `discarded_df`
    discarded_df = df[discard_mask].copy()
    
    # 2. Create the `cleaned_df`
    df_cleaned = df[~discard_mask].copy()
    
    # ---  Add 'discard_reason' column to the discarded_df ---
    if not discarded_df.empty:
        logging.info(f"  Saving {len(discarded_df)} rows to the discard bin...")
        
        # Function to efficiently find all reasons
        def get_reason(row):
            reasons = # <-- FIX: Initialized list
            if row['negative_rows']:
                reasons.append('negative_rows')
            if row['high_rows']:
                reasons.append('high_rows')
            if row['spike_rows']:
                # ---  Check if this was an *acquitted* spike ---
                reason_str = 'spike_rows'
                if row['potential_spike_rows'] == True:
                    reason_str = f"spike_rows (Uncorroborated spike)"
                reasons.append(reason_str)
            
            # ---  Update humidity reason ---
            if row['humidity_rows']: # This flag is now 'rows_discarded_fog'
                # ---  Add the specific paired sensor ID ---
                # Check if merge data is valid (it should be if humidity_rows is True)
                if pd.notna(row['humidity_rh_value']) and pd.notna(row['humidity_sensor_id_rh']):
                    reason_str = f"humidity_rows (Fog/Saturation: {row['humidity_rh_value']:.1f}% >= {rh_high_bound}%. Caused by sensor {row['humidity_sensor_id_rh']:.0f} at {row['timestamp_rh']})"
                else:
                    reason_str = f"humidity_rows (Fog/Saturation: {rh_high_bound}%+)"
                reasons.append(reason_str)
            
            return '; '.join(reasons)
        
        # Apply the function to create the new column
        discarded_df['discard_reason'] = discarded_df.apply(get_reason, axis=1)
    else:
        logging.info("  No rows matched the discard criteria.")
        # Create an empty 'discard_reason' column so the schema matches
        discarded_df['discard_reason'] = pd.Series(dtype='str')

    
    cleaned_rows = len(df_cleaned)
    logging.info(f"  Cleaning complete. Kept {cleaned_rows} rows. Discarded {original_rows - cleaned_rows} rows.")

    # Drop temporary columns from *both* DataFrames
    # ---  Added rows_corrected_k_kohler, rows_discarded_fog ---
    cols_to_drop = ['negative_rows', 'high_rows', 'spike_rows', 'humidity_rows', 
                    'rows_corrected_k_kohler', 'rows_discarded_fog',
                    'local_tz', 'humidity_rh_value', 'timestamp_rh',
                    'paired_humidity_sensor_id', 'humidity_sensor_id_rh'
                    ] + cols_to_drop_later
    
    df_cleaned = df_cleaned.drop(columns=cols_to_drop, errors='ignore')
    
    # Also drop them from the discarded df, but keep local_tz
    # ---  Added rows_corrected_k_kohler, rows_discarded_fog ---
    cols_to_drop_discarded = ['negative_rows', 'high_rows', 'spike_rows', 'humidity_rows',
                              'rows_corrected_k_kohler', 'rows_discarded_fog',
                              'humidity_rh_value', 'timestamp_rh',
                              'paired_humidity_sensor_id', 'humidity_sensor_id_rh'
                              ] + cols_to_drop_later
    discarded_df = discarded_df.drop(columns=cols_to_drop_discarded, errors='ignore')

    
    # Return all three DataFrames
    return df_cleaned, final_log_df, discarded_df


# --- HELPER FUNCTION 1 (LOGIC UPDATED FROM PDF) ---
def calculate_stats_for_series(data_series):
    """
    Calculates statistical and combined (adjusted) error for a given pandas Series,
    based on the methodology from 'Reporting Measurement Error in Air Quality.pdf'.
    --- MODIFIED FOR REQUEST 3 ---
    """
    
    # --- Basic Stats ---
    n = data_series.count()
    if n == 0:
        return pd.Series({
            'mean': np.nan, 'median': np.nan, 'std_dev': np.nan, 'n_samples': 0,
            'stat_ci_95_margin': np.nan, 'stat_ci_95_low': np.nan, 'stat_ci_95_high': np.nan,
            'adj_ci_95_margin': np.nan, 'adj_ci_95_low': np.nan, 'adj_ci_95_high': np.nan
        })
    
    mean = data_series.mean()
    median = data_series.median()
    std = data_series.std()
    
    # --- 1. Statistical Error (E_stat) ---
    # This is the 95% CI based on the data's variability 
    E_stat, stat_ci_low, stat_ci_high = np.nan, np.nan, np.nan
    
    if n > 1:
        se = std / (n**0.5)
        if se and pd.notna(se): 
            
            # --- Request 3: Use Student's t-distribution ---
            # Use t-distribution for small sample sizes (which is safer for all n)
            # instead of Z-distribution (norm.ppf)
            degrees_of_freedom = n - 1
            t_score = stats.t.ppf(0.975, df=degrees_of_freedom) 
            
            # E_stat is the 95% margin: t_score * (s / sqrt(n)) 
            E_stat = t_score * se 
            # --- End Request 3 ---
            
            stat_ci_low = mean - E_stat
            stat_ci_high = mean + E_stat
            
    # --- 2. Instrumental Error (E_instr) ---
    # This is the sensor's inherent accuracy error 
    E_instr = np.nan # Default to NaN (will be treated as 0 in combination)
    
    # This logic only applies to SDS011 PM sensors
    # --- MODIFIED: Check for uncorrected columns too ---
    if data_series.name in ['PM2.5', 'PM10', 'PM2.5_uncorrected', 'PM10_uncorrected']:
        error_pct = mean * 0.15   # 15% error 
        error_abs = 10            # 10 ug/m3 error 
        
        # Use whichever error is GREATER: max(mean * 0.15, 10) 
        E_instr = max(error_pct, error_abs)
            
    # --- 3. Combined (Adjusted) Error (E_comb) ---
    # Combine errors in quadrature: E_comb = sqrt(E_stat^2 + E_instr^2) 
    adj_ci_margin, adj_ci_low, adj_ci_high = np.nan, np.nan, np.nan
    
    # We can only combine if E_stat is a valid number
    if pd.notna(E_stat):
        # If E_instr is also valid (i.e., for PM2.5/PM10), combine them 
        if pd.notna(E_instr):
            adj_ci_margin = np.sqrt(E_stat**2 + E_instr**2)
        # Otherwise (e.g., for temperature), the adjusted margin is just the statistical margin
        else:
            adj_ci_margin = E_stat
            
        adj_ci_low = mean - adj_ci_margin
        adj_ci_high = mean + adj_ci_margin
    # If E_stat is NaN (e.g., n=1), we can't calculate any CI
    else:
        adj_ci_margin, adj_ci_low, adj_ci_high = np.nan, np.nan, np.nan

    return pd.Series({
        'mean': mean, 
        'median': median, 
        'std_dev': std,
        'n_samples': n,
        'stat_ci_95_margin': E_stat, 
        'stat_ci_95_low': stat_ci_low, 
        'stat_ci_95_high': stat_ci_high,
        'adj_ci_95_margin': adj_ci_margin, 
        'adj_ci_95_low': adj_ci_low, 
        'adj_ci_95_high': adj_ci_high
    })


# --- HELPER FUNCTION 2 ---
def calculate_stats_for_dataframe(data_frame):
    """
    Applies the series-level statistics function to each 
    column of the received DataFrame.
    """
    return data_frame.apply(calculate_stats_for_series, axis=0)

# ----------------------------------------

def main():
    # --- Setup Logging FIRST ---
    # This will also create the OUTPUT_DIR if it's missing
    setup_logging()
    
    start_time = datetime.now()
    logging.info(f"Script started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # --- Calculate total sensor-days for resource check ---
    try:
        start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
        end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
        num_days = (end_dt - start_dt).days + 1

        num_main_sensors = len(SENSOR_IDS_TO_DOWNLOAD)
        if SENSOR_PAIRINGS:
            num_bme_sensors = len(set(SENSOR_PAIRINGS.values()))
        else:
            num_bme_sensors = 0
        total_sensors = num_main_sensors + num_bme_sensors
        
        total_sensor_days = total_sensors * num_days
        logging.info("--- Download Job ---")
        logging.info(f"Total sensor-days to download: {total_sensor_days} ({total_sensors} sensors * {num_days} days)")

    except Exception as e:
        logging.error(f"Could not calculate job size: {e}.")
        # Need to set these variables for the next block
        num_main_sensors = len(SENSOR_IDS_TO_DOWNLOAD)
        if SENSOR_PAIRINGS:
            num_bme_sensors = len(set(SENSOR_PAIRINGS.values()))
        else:
            num_bme_sensors = 0
        total_sensors = num_main_sensors + num_bme_sensors
        num_days = 1 # Failsafe
    
    
    # --- System Requirement Check (UPDATED) ---
    logging.info("--- System Requirement Check ---")
    try:
        # 1. Calculate requirements (variables are from logic above)
        total_sensors_to_download = total_sensors # Renaming for clarity
        
        # --- Base Calculation ---
        # Constants (KB_PER_SENSOR_DAY, DISK_SAFETY_MARGIN, RAM_PROCESSING_MULTIPLIER)
        # are now defined in the USER CONFIGURATION section at the top.
        base_estimated_kb = KB_PER_SENSOR_DAY * total_sensors_to_download * num_days
        base_data_bytes = int(base_estimated_kb * 1024)
        
        # --- Disk Requirement ---
        # Base size * 20% safety margin for file size variance
        required_disk_bytes = int(base_data_bytes * DISK_SAFETY_MARGIN)
        
        # --- RAM Requirement ---
        # Base size * 3x-15x multiplier for pandas in-memory processing (concat, merge, copy)
        required_ram_bytes = int(base_data_bytes * RAM_PROCESSING_MULTIPLIER)
        
        # 2. Log and Print the calculation
        calc_message = (
            f"Estimated resource requirements based on config:\n"
            f"  Main Sensors: {num_main_sensors}\n"
            f"  BME280 Sensors: {num_bme_sensors}\n"
            f"  Total Sensors: {total_sensors_to_download}\n"
            f"  Date Range: {num_days} days\n"
            f"  Base data size (@{KB_PER_SENSOR_DAY}KB/sensor/day): {format_bytes(base_data_bytes)}\n"
            f"  --------------------------------------------------\n"
            f"  Est. Required DISK (Base * {DISK_SAFETY_MARGIN:.2f}x margin): {format_bytes(required_disk_bytes)}\n"
            f"  Est. Required RAM (Base * {RAM_PROCESSING_MULTIPLIER:.1f}x processing load): {format_bytes(required_ram_bytes)}"
        )
        logging.info(calc_message)
        # print(calc_message) # Redundant print removed

        # 3. Perform checks
        check_disk_space(required_disk_bytes)
        check_ram_space(required_ram_bytes)
        
    except Exception as e:
        logging.error(f"Could not perform system requirement check: {e}")
        logging.warning("Continuing script, but resource issues may occur.")
        
    # ---  Sensor-Day Threshold Warning (User Request) ---
    if total_sensor_days > SENSOR_DAY_WARNING_THRESHOLD:
        warning_msg = (
            f"\n*** WARNING: Job size ({total_sensor_days} sensor-days) exceeds the threshold ({SENSOR_DAY_WARNING_THRESHOLD}). ***\n"
            f"This may cause the script to fail with a 'MemoryError' during the final combination phase.\n"
            f"It is highly recommended to run this job in smaller batches (e.g., 5 sensors for 5 months).\n"
        )
        logging.warning(warning_msg)
        print(warning_msg)
        
        try:
            choice = input("Do you wish to continue anyway? (y/n): ").strip().lower()
            if choice!= 'y':
                logging.info("User aborted the script. Exiting.")
                print("Aborting script.")
                return # Exit main()
            else:
                logging.info("User chose to proceed despite the warning.")
        except EOFError: # Handle non-interactive environments
            logging.error("Cannot get user input. Aborting script to prevent potential failure.")
            print("Cannot get user input. Aborting.")
            return # Exit main()
    # --- END NEW SECTION ---
        
    logging.info("----------------------------------")
    # --- END NEW SECTION ---

    # Initialize TimezoneFinder (for Request 5)
    tf = TimezoneFinder()
    
    logging.info("--- Configuration ---")
    logging.info(f"Sensor IDs: {SENSOR_IDS_TO_DOWNLOAD}")
    logging.info(f"Sensor Types to try: {SENSOR_TYPES_TO_TRY}")
    logging.info(f"Date Range: {START_DATE} to {END_DATE}")
    if SENSOR_PAIRINGS:
        logging.info(f"Sensor Pairings: {SENSOR_PAIRINGS}")
    else:
        logging.info("No sensor pairings defined. No humidity filtering will be applied.")
    # ---  Log new filter settings ---
    logging.info(f"Humidity Logic: k-Köhler (k={KAPPA_PARAMETER})")
    logging.info(f"  > Use As-Is: RH < {RH_CORRECTION_LOWER_BOUND}%")
    logging.info(f"  > Correct: {RH_CORRECTION_LOWER_BOUND}% <= RH < {RH_CORRECTION_UPPER_BOUND}%")
    logging.info(f"  > Discard: RH >= {RH_CORRECTION_UPPER_BOUND}%")
    # ---  Log new spike filter setting (User Request) ---
    logging.info(f"Spike Filter: > {SPIKE_MULTIPLIER}x (0=disabled), with {SPIKE_MAX_TIME_GAP_MINUTES} min gap limit")
    # ---  Log retry settings ---
    logging.info(f"Download Retries: {MAX_RETRIES} attempts")
    logging.info(f"Initial Retry Delay: {RETRY_INITIAL_DELAY_SECONDS} seconds")
    logging.info(f"Delay Jitter: {POLITE_DOWNLOAD_DELAY_JITTER_SECONDS} seconds")
    
    logging.info(f"Output Directory: {OUTPUT_DIR}")
    logging.info("---------------------")


    # --- Phase 2: Download PM Sensor Data ---
    logging.info("--- Phase 2: Download Main Sensor Data ---")

    # ---  Create a shared requests.Session ---
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    logging.info(f"Using User-Agent: {USER_AGENT}")

    # ---  Add "From" header if provided ---
    if FROM_EMAIL:
        session.headers.update({"From": FROM_EMAIL})
        logging.info(f"Using 'From' header: {FROM_EMAIL}")

    all_dataframes = download_data_daily(
        session,
        SENSOR_IDS_TO_DOWNLOAD, 
        SENSOR_TYPES_TO_TRY, 
        START_DATE, 
        END_DATE
    )


    # --- Phase 2b: Download Humidity Sensor Data (Request 4) ---
    humidity_df = None
    # --- Download data for all sensors defined in the SENSOR_PAIRINGS values ---
    if SENSOR_PAIRINGS:
        # Get a unique list of all BME280 sensor IDs to download
        humidity_sensor_ids_to_download = list(set(SENSOR_PAIRINGS.values()))
        
        if humidity_sensor_ids_to_download:
            logging.info(f"--- Phase 2b: Downloading humidity data for {len(humidity_sensor_ids_to_download)} paired sensor(s): {humidity_sensor_ids_to_download} ---")
            
            # --- DOWNLOAD LOGIC SIMPLIFIED ---
            humidity_dfs = download_data_daily(
                session,
                humidity_sensor_ids_to_download, # Pass list of IDs
                ["bme280"], 
                START_DATE, 
                END_DATE
            )

            if humidity_dfs:
                humidity_df = pd.concat(humidity_dfs, ignore_index=True)
                logging.info(f"  Successfully downloaded {len(humidity_df)} total humidity records.")
            else:
                logging.warning(f"  Warning: No data found for humidity sensors {humidity_sensor_ids_to_download}.")
        else:
            logging.info("Phase 2b: SENSOR_PAIRINGS is empty. Skipping humidity download.")
    else:
        logging.info("Phase 2b: SENSOR_PAIRINGS is not defined. Skipping humidity download.")

    # --- Close the session ---
    session.close()


    # Phase 3: Combine, Save, and Analyze
    logging.info("--- Phase 3: Combining all data... ---")
    if not all_dataframes:
        logging.error("No data was downloaded for main sensors. Check your IDs, date range, and sensor types.")
        logging.info("Script will now exit.")
        return

    try:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        # Clear the list of DataFrames from memory
        del all_dataframes
        
        if 'timestamp' not in final_df.columns:
            logging.error("Error: 'timestamp' column not found. Cannot proceed with analysis.")
            return
        if 'sensor_id' not in final_df.columns:
            logging.error("Error: 'sensor_id' column not found. Cannot proceed with analysis.")
            return

        # --- Rename columns *BEFORE* cleaning ---
        logging.info("Renaming P1/P2 columns to PM10/PM2.5...")
        rename_map = {'P2': 'PM2.5', 'P1': 'PM10'}
        final_df.rename(columns=rename_map, inplace=True)
            
        # --- Request 5: Timezone Handling ---
        logging.info("Localizing timestamps...")
        
        # 1. Make all timestamps UTC-aware (data from archive is in UTC)
        # This step is now more robust. If data came from the monthly downloader,
        # it's already a datetime object. If from daily, it's a string.
        # pd.to_datetime handles both cases.
        final_df['timestamp'] = pd.to_datetime(final_df['timestamp'], utc=True)

        # 2. Find the local timezone for each sensor
        sensor_locations = final_df.groupby('sensor_id')[['lat', 'lon']].median()
        sensor_locations['local_tz'] = sensor_locations.apply(
            lambda row: get_local_tz(row['lat'], row['lon'], tf), axis=1
        )
        tz_map = sensor_locations['local_tz'].to_dict()
        
        # 3. Map the local timezone back to the main DataFrame
        final_df['local_tz'] = final_df['sensor_id'].map(tz_map)
        
        # 4. Apply the localization conversion
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            final_df = final_df.groupby('sensor_id', group_keys=False).apply(localize_group)
        logging.info("  Timestamps localized.")
        
        # 5. Localize humidity_df as well
        if humidity_df is not None and not humidity_df.empty:
            logging.info("  Localizing humidity timestamps...")
            humidity_df['timestamp'] = pd.to_datetime(humidity_df['timestamp'], utc=True)
            
            # Find timezone for *each* humidity sensor
            hum_sensor_locations = humidity_df.groupby('sensor_id')[['lat', 'lon']].median()
            hum_sensor_locations['local_tz'] = hum_sensor_locations.apply(
                lambda row: get_local_tz(row['lat'], row['lon'], tf), axis=1
            )
            hum_tz_map = hum_sensor_locations['local_tz'].to_dict()
            
            # Map the local timezone back to the humidity DataFrame
            humidity_df['local_tz'] = humidity_df['sensor_id'].map(hum_tz_map)
            
            # Apply localization
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                humidity_df = humidity_df.groupby('sensor_id', group_keys=False).apply(localize_group)
            
            # Drop the temp local_tz column
            humidity_df = humidity_df.drop(columns=['local_tz'], errors='ignore')

        # --- End Request 5 ---

        # Create temporal grouping columns *after* localization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            final_df['month'] = final_df['timestamp'].dt.to_period('M')
            final_df['week'] = final_df['timestamp'].dt.to_period('W')
    
    
        
        # ==================================================================
        # ---  Save raw BME280 data (per sensor)
        # ==================================================================
        MAX_ROWS_PER_FILE = 990000 # Max rows for Excel compatibility
        
        if os.path.isabs(FINAL_CSV_FILE):
            save_path_base = FINAL_CSV_FILE
        else:
            save_path_base = os.path.join(OUTPUT_DIR, FINAL_CSV_FILE)
            
        base_path, extension = os.path.splitext(save_path_base)
            
        if humidity_df is not None and not humidity_df.empty:
            logging.info("Saving raw BME280 data per sensor...")
            unique_bme_sensors = humidity_df['sensor_id'].unique()
            
            for sensor_id in unique_bme_sensors:
                logging.info(f"    Processing BME280 sensor: {sensor_id}")
                bme_sensor_df = humidity_df[humidity_df['sensor_id'] == sensor_id].copy()
                total_rows_sensor = len(bme_sensor_df)
                
                bme_save_base = f"{base_path}_sensor_BME280_{sensor_id}"
                
                if total_rows_sensor == 0:
                    logging.warning(f"      BME280 Sensor {sensor_id} has no data. File will not be saved.")
                    continue
                elif total_rows_sensor <= MAX_ROWS_PER_FILE:
                    sensor_file_path = f"{bme_save_base}{extension}"
                    logging.info(f"      Saving {total_rows_sensor} rows to {sensor_file_path}...")
                    bme_sensor_df.to_csv(sensor_file_path, index=False)
                else:
                    # Apply chunking if BME data is very large
                    num_files_sensor = int(np.ceil(total_rows_sensor / MAX_ROWS_PER_FILE))
                    logging.info(f"      Total sensor rows ({total_rows_sensor}) exceed {MAX_ROWS_PER_FILE}. Saving in {num_files_sensor} parts...")
                    
                    for i in range(num_files_sensor):
                        start_row = i * MAX_ROWS_PER_FILE
                        end_row = (i + 1) * MAX_ROWS_PER_FILE
                        chunk_df = bme_sensor_df.iloc[start_row:end_row]
                        part_file_path = f"{bme_save_base}_part_{i+1}{extension}"
                        
                        logging.info(f"        Saving part {i+1}/{num_files_sensor} ({len(chunk_df)} rows) to {part_file_path}...")
                        chunk_df.to_csv(part_file_path, index=False)
        # ==================================================================


        # --- Request 1, 2, 4: Clean and Log Data ---
        logging.info("Applying data cleaning rules and humidity correction...")
        
        # --- MODIFIED: Expect three return values, pass k-Köhler params ---
        cleaned_df, cleaning_logs_df, discarded_df = clean_and_log_data(
            final_df, 
            all_cleaning_logs, 
            humidity_df, 
            SENSOR_PAIRINGS,
            SPIKE_MULTIPLIER,            # <-- Pass spike multiplier
            # ---  Pass time gap (User Request) ---
            SPIKE_MAX_TIME_GAP_MINUTES,
            KAPPA_PARAMETER,             
            RH_CORRECTION_LOWER_BOUND,   
            RH_CORRECTION_UPPER_BOUND    
        )
        
        # Clear the large, un-cleaned DataFrame from memory
        del final_df
        
        # --- End Request 1, 2, 4 ---

        # --- Re-ordering (Rename step was moved) ---
        logging.info("Re-ordering columns...")
        # --- MODIFIED: Also check for uncorrected columns ---
        if 'PM2.5' in cleaned_df.columns and 'PM10' in cleaned_df.columns:
            all_cols = list(cleaned_df.columns)
            
            # Try to move PM2.5 next to PM10
            if 'PM10' in all_cols and 'PM2.5' in all_cols:
                try:
                    pm10_index = all_cols.index('PM10')
                    all_cols.remove('PM2.5')
                    all_cols.insert(pm10_index, 'PM2.5')
                except ValueError:
                    pass # Column already removed or not present
            
            # Try to move PM2.5_uncorrected next to PM10_uncorrected
            if 'PM10_uncorrected' in all_cols and 'PM2.5_uncorrected' in all_cols:
                try:
                    pm10_unc_index = all_cols.index('PM10_uncorrected')
                    all_cols.remove('PM2.5_uncorrected')
                    all_cols.insert(pm10_unc_index, 'PM2.5_uncorrected')
                except ValueError:
                    pass
            
            cleaned_df = cleaned_df[all_cols]
        
        
        
        # ==================================================================
        # ---  Save Discarded Data (with chunking) ---
        # ==================================================================
        logging.info("Saving discarded data...")
        
        discard_base_path = f"{base_path}_DISCARDED"
        total_rows_discarded = len(discarded_df)
        
        if total_rows_discarded == 0:
            logging.info("  No discarded data to save.")
        elif total_rows_discarded <= MAX_ROWS_PER_FILE:
            discard_file_path = f"{discard_base_path}{extension}"
            logging.info(f"  Saving discarded file ({total_rows_discarded} rows) to {discard_file_path}...")
            discarded_df.to_csv(discard_file_path, index=False)
        else:
            num_files_discarded = int(np.ceil(total_rows_discarded / MAX_ROWS_PER_FILE))
            logging.info(f"  Total discarded rows ({total_rows_discarded}) exceed {MAX_ROWS_PER_FILE}. Saving in {num_files_discarded} parts...")
            
            for i in range(num_files_discarded):
                start_row = i * MAX_ROWS_PER_FILE
                end_row = (i + 1) * MAX_ROWS_PER_FILE
                chunk_df = discarded_df.iloc[start_row:end_row]
                part_file_path = f"{discard_base_path}_part_{i+1}{extension}"
                
                logging.info(f"    Saving discarded part {i+1}/{num_files_discarded} ({len(chunk_df)} rows) to {part_file_path}...")
                chunk_df.to_csv(part_file_path, index=False)
        
        # --- Clear discarded_df from memory ---
        del discarded_df

        # ==================================================================
        # --- Save Cleaned Data (All Sensors and By Sensor) ---
        # ==================================================================
        logging.info("Saving cleaned data...")
        
        # --- 1. Save the MASTER file (all sensors) with chunking ---
        total_rows_all = len(cleaned_df)

        if total_rows_all == 0:
            logging.warning("No cleaned data available to save. Master file will be empty or not created.")
        elif total_rows_all <= MAX_ROWS_PER_FILE:
            logging.info(f"  Saving master file ({total_rows_all} rows) to {save_path_base}...")
            cleaned_df.to_csv(save_path_base, index=False)
        else:
            num_files_all = int(np.ceil(total_rows_all / MAX_ROWS_PER_FILE))
            logging.info(f"  Total rows ({total_rows_all}) exceed {MAX_ROWS_PER_FILE}. Saving master file in {num_files_all} parts...")
            
            for i in range(num_files_all):
                start_row = i * MAX_ROWS_PER_FILE
                end_row = (i + 1) * MAX_ROWS_PER_FILE
                chunk_df = cleaned_df.iloc[start_row:end_row]
                part_file_path = f"{base_path}_part_{i+1}{extension}"
                
                logging.info(f"    Saving master part {i+1}/{num_files_all} ({len(chunk_df)} rows) to {part_file_path}...")
                chunk_df.to_csv(part_file_path, index=False)

        # --- 2. Save by individual sensor (with chunking) ---
        logging.info("  Saving cleaned data per sensor...")
        unique_sensor_ids = cleaned_df['sensor_id'].unique()
        
        for sensor_id in unique_sensor_ids:
            logging.info(f"    Processing sensor: {sensor_id}")
            
            sensor_df = cleaned_df[cleaned_df['sensor_id'] == sensor_id].copy()
            total_rows_sensor = len(sensor_df)
            
            sensor_save_base = f"{base_path}_sensor_{sensor_id}"
            
            if total_rows_sensor == 0:
                logging.warning(f"    Sensor {sensor_id} has no cleaned data. File will not be saved.")
                continue
            elif total_rows_sensor <= MAX_ROWS_PER_FILE:
                sensor_file_path = f"{sensor_save_base}{extension}"
                logging.info(f"      Saving {total_rows_sensor} rows to {sensor_file_path}...")
                sensor_df.to_csv(sensor_file_path, index=False)
            else:
                num_files_sensor = int(np.ceil(total_rows_sensor / MAX_ROWS_PER_FILE))
                logging.info(f"      Total sensor rows ({total_rows_sensor}) exceed {MAX_ROWS_PER_FILE}. Saving in {num_files_sensor} parts...")
                
                for i in range(num_files_sensor):
                    start_row = i * MAX_ROWS_PER_FILE
                    end_row = (i + 1) * MAX_ROWS_PER_FILE
                    chunk_df = sensor_df.iloc[start_row:end_row]
                    part_file_path = f"{sensor_save_base}_part_{i+1}{extension}"
                    
                    logging.info(f"        Saving part {i+1}/{num_files_sensor} ({len(chunk_df)} rows) to {part_file_path}...")
                    chunk_df.to_csv(part_file_path, index=False)
        
        
        # ==================================================================
        # --- Statistical Analysis (Unchanged) ---
        # ==================================================================
        
        if cleaned_df.empty and (cleaning_logs_df is None or cleaning_logs_df.empty):
            logging.warning("Cleaned dataframe and log dataframe are empty. Skipping statistical analysis.")
        else:
            logging.info("Calculating summary statistics...")
            
            # --- MODIFIED: Add uncorrected columns to skip list ---
            # This ensures stats (mean, median, t-dist) are ONLY run on the
            # corrected PM2.5 and PM10 columns, as requested.
            cols_to_skip = ['sensor_id', 'location', 'lat', 'lon', 
                            'PM2.5_uncorrected', 'PM10_uncorrected']
            
            analysis_cols = [
                col for col in cleaned_df.columns 
                if pd.api.types.is_numeric_dtype(cleaned_df[col]) 
                and col not in cols_to_skip
            ]
            
            if not analysis_cols and cleaned_df.empty:
                logging.warning("No numeric data columns found and cleaned_df is empty. Stats files will only contain log data.")
                # Create a dummy stats_df function that returns an empty structure
                # This is tricky, easier to let it create an empty stats_df
                pass # Let it proceed, stats_df will be empty
            
            logging.info(f"Analyzing columns: {analysis_cols}")
            
            # --- MODIFIED: Add new log column names ---
            log_col_names = ['negative_rows', 'high_rows', 'spike_rows', 
                             'rows_discarded_fog', 'rows_corrected_k_kohler', 
                             'bad_lines_skipped']

            # 3a. Per Sensor
            stats_sensor = cleaned_df.groupby('sensor_id')[analysis_cols].apply(calculate_stats_for_dataframe)
            sensor_logs = cleaning_logs_df.drop(columns=['month', 'week']).groupby('sensor_id').sum().reset_index()
            # --- NEW FIX: Re-index stats to include all sensors from logs ---
            if not stats_sensor.empty:
                stat_names = stats_sensor.index.get_level_values(1).unique() # Index is 'sensor_id', level 1
            else:
                dummy_stats = calculate_stats_for_series(pd.Series([1]))
                stat_names = dummy_stats.index

            stats_groups = stats_sensor.reset_index()[['sensor_id']].drop_duplicates()
            logs_groups = sensor_logs[['sensor_id']].drop_duplicates()
            all_groups = pd.merge(stats_groups, logs_groups, on='sensor_id', how='outer')

            if not all_groups.empty:
                new_index = pd.MultiIndex.from_product(
                    [all_groups['sensor_id'], stat_names],
                    names=['sensor_id', 'level_1']
                )
                stats_sensor_reindexed = stats_sensor.reindex(new_index)
            else:
                stats_sensor_reindexed = stats_sensor # No groups to reindex
            
            stats_sensor_merged = stats_sensor_reindexed.reset_index().merge(
                sensor_logs, 
                on='sensor_id', 
                how='left'
            )
            
            stats_sensor_merged[log_col_names] = stats_sensor_merged[log_col_names].fillna(0).astype(int)

            if not stats_sensor.empty or not cleaned_df.empty:
                n_samples_mask = stats_sensor_merged['level_1'] == 'n_samples'
                stat_value_cols = stats_sensor.columns
                stats_sensor_merged.loc[n_samples_mask, stat_value_cols] = stats_sensor_merged.loc[n_samples_mask, stat_value_cols].fillna(0)
            
            stats_sensor_merged.round(1).to_csv(os.path.join(OUTPUT_DIR, "stats_per_sensor.csv"), index=False)
            logging.info("    > Saved stats_per_sensor.csv (with cleaning logs)")

            # 3b. Per Month
            stats_month = cleaned_df.groupby('month')[analysis_cols].apply(calculate_stats_for_dataframe)
            month_logs = cleaning_logs_df.drop(columns=['sensor_id', 'week']).groupby('month').sum().reset_index()
            # --- NEW FIX: Re-index stats to include all months from logs ---
            if not stats_month.empty:
                stat_names = stats_month.index.get_level_values(1).unique() # Index is 'month', level 1
            else:
                dummy_stats = calculate_stats_for_series(pd.Series([1]))
                stat_names = dummy_stats.index

            stats_groups = stats_month.reset_index()[['month']].drop_duplicates()
            logs_groups = month_logs[['month']].drop_duplicates()
            all_groups = pd.merge(stats_groups, logs_groups, on='month', how='outer')

            if not all_groups.empty:
                new_index = pd.MultiIndex.from_product(
                    [all_groups['month'], stat_names],
                    names=['month', 'level_1']
                )
                stats_month_reindexed = stats_month.reindex(new_index)
            else:
                stats_month_reindexed = stats_month # No groups to reindex
            
            stats_month_merged = stats_month_reindexed.reset_index().merge(
                month_logs, 
                on='month', 
                how='left'
            )
            
            stats_month_merged[log_col_names] = stats_month_merged[log_col_names].fillna(0).astype(int)

            if not stats_month.empty or not cleaned_df.empty:
                n_samples_mask = stats_month_merged['level_1'] == 'n_samples'
                stat_value_cols = stats_month.columns
                stats_month_merged.loc[n_samples_mask, stat_value_cols] = stats_month_merged.loc[n_samples_mask, stat_value_cols].fillna(0)

            stats_month_merged.round(1).to_csv(os.path.join(OUTPUT_DIR, "stats_per_month.csv"), index=False)
            logging.info("    > Saved stats_per_month.csv (with cleaning logs)")
            
            # 3c. Per Sensor and Per Month
            stats_sensor_month = cleaned_df.groupby(['sensor_id', 'month'])[analysis_cols].apply(calculate_stats_for_dataframe)
            sensor_month_logs = cleaning_logs_df.drop(columns=['week']).groupby(['sensor_id', 'month']).sum().reset_index()
            # --- NEW FIX: Re-index stats to include all sensors/months from logs ---
            if not stats_sensor_month.empty:
                stat_names = stats_sensor_month.index.get_level_values(2).unique() # Index is 'sensor_id', 'month', level 2
            else:
                # If no stats were calculated at all, get names from the function
                dummy_stats = calculate_stats_for_series(pd.Series([1]))
                stat_names = dummy_stats.index
            
            # 2. Get all (sensor_id, month) combos from *both* dataframes
            stats_groups = stats_sensor_month.reset_index()[['sensor_id', 'month']].drop_duplicates()
            logs_groups = sensor_month_logs[['sensor_id', 'month']].drop_duplicates()
            all_groups = pd.merge(stats_groups, logs_groups, on=['sensor_id', 'month'], how='outer')

            # 3. Create the full MultiIndex
            if not all_groups.empty:
                all_sensor_ids = all_groups['sensor_id'].unique()
                all_months = all_groups['month'].unique()
                
                new_index = pd.MultiIndex.from_product(
                    [all_sensor_ids, all_months, stat_names],
                    names=['sensor_id', 'month', 'level_2']
                )
                # Filter new_index to only include combinations present in all_groups
                all_groups_idx = pd.MultiIndex.from_frame(all_groups)
                new_index_filtered = new_index[new_index.droplevel('level_2').isin(all_groups_idx)]
                stats_sensor_month_reindexed = stats_sensor_month.reindex(new_index_filtered)
            else:
                stats_sensor_month_reindexed = stats_sensor_month # No groups to reindex
            
            # 5. Reset index and merge with logs. Use 'left' since left side is now complete.
            stats_sensor_month_merged = stats_sensor_month_reindexed.reset_index().merge(
                sensor_month_logs, 
                on=['sensor_id', 'month'], 
                how='left' # Use 'left' to broadcast logs to all 10 stat rows
            )
            
            # 6. Fill NaNs for log columns
            stats_sensor_month_merged[log_col_names] = stats_sensor_month_merged[log_col_names].fillna(0).astype(int)

            # 7. SPECIAL CASE: For rows where level_2 is 'n_samples', fill NaN values with 0
            # This fills the PM2.5, PM10, etc. columns for the 'n_samples' row
            if not stats_sensor_month.empty or not cleaned_df.empty:
                n_samples_mask = stats_sensor_month_merged['level_2'] == 'n_samples'
                # Get all columns *except* the index and log columns
                stat_value_cols = stats_sensor_month.columns
                stats_sensor_month_merged.loc[n_samples_mask, stat_value_cols] = stats_sensor_month_merged.loc[n_samples_mask, stat_value_cols].fillna(0)

            stats_sensor_month_merged.round(1).to_csv(os.path.join(OUTPUT_DIR, "stats_per_sensor_per_month.csv"), index=False)
            logging.info("    > Saved stats_per_sensor_per_month.csv (with cleaning logs)")

            # 3d. Per Week
            stats_week = cleaned_df.groupby('week')[analysis_cols].apply(calculate_stats_for_dataframe)
            week_logs = cleaning_logs_df.drop(columns=['sensor_id', 'month']).groupby('week').sum().reset_index()
            # --- NEW FIX: Re-index stats to include all weeks from logs ---
            if not stats_week.empty:
                stat_names = stats_week.index.get_level_values(1).unique() # Index is 'week', level 1
            else:
                dummy_stats = calculate_stats_for_series(pd.Series([1]))
                stat_names = dummy_stats.index

            stats_groups = stats_week.reset_index()[['week']].drop_duplicates()
            logs_groups = week_logs[['week']].drop_duplicates()
            all_groups = pd.merge(stats_groups, logs_groups, on='week', how='outer')

            if not all_groups.empty:
                new_index = pd.MultiIndex.from_product(
                    [all_groups['week'], stat_names],
                    names=['week', 'level_1']
                )
                stats_week_reindexed = stats_week.reindex(new_index)
            else:
                stats_week_reindexed = stats_week # No groups to reindex
            
            stats_week_merged = stats_week_reindexed.reset_index().merge(
                week_logs, 
                on='week', 
                how='left'
            )
            
            stats_week_merged[log_col_names] = stats_week_merged[log_col_names].fillna(0).astype(int)

            if not stats_week.empty or not cleaned_df.empty:
                n_samples_mask = stats_week_merged['level_1'] == 'n_samples'
                stat_value_cols = stats_week.columns
                # --- THIS IS THE FIX ---
                stats_week_merged.loc[n_samples_mask, stat_value_cols] = stats_week_merged.loc[n_samples_mask, stat_value_cols].fillna(0)

            stats_week_merged.round(1).to_csv(os.path.join(OUTPUT_DIR, "stats_per_week.csv"), index=False)
            logging.info("    > Saved stats_per_week.csv (with cleaning logs)")

        # ==================================================================
        
        end_time = datetime.now()
        logging.info("--- SCRIPT FINISHED SUCCESSFULLY ---")
        logging.info(f"All files saved to {OUTPUT_DIR}")
        logging.info(f"Total cleaned data rows for analysis: {len(cleaned_df)}")
        logging.info(f"Total time: {end_time - start_time}")

    except Exception as e:
        logging.exception("--- SCRIPT FAILED WITH AN UNHANDLED ERROR ---")
        # This will automatically log the full traceback

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # This will catch any error that happens *before* logging is set up
        print(f"CRITICAL SCRIPT FAILURE: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n-------------------------------------")
        input("Press ENTER to exit...")