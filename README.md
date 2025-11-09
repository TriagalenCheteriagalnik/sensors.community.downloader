# sensors.community.downloader - CURRENTLY IN BETA!!!
What is this? 

This is a piece of Python software that allows for you to do the following:

1. Download specific sensors.community SDS011 air quality sensors data, for specific dates or time periods of your choosing
2. Account for humidity via pairing to humidity sensors and an automated Kohler algorithm implementation (configurable)
3. Remove bad data points (including corrupted data and all data outside the SDS011s operating parameters)
4. Account for the instrumental sensor error of the SDS011 particle sensor
5. Remove single-sensor spikes and statistical outliers (with an acquittal mechanism to prevent correct readings from being deleted)
6. Generate weekly and monthly reports, with calculated average (mean), median, standard deviation and confidence interval values
7. Keep separate logs of discarded data, in case you want to fact-check the algorithm

It's easy to use and requires only initial user configuration. Everything else is automated. 

It also has several other features, such as:
- Time and system resource estimations
- Polite download practices to avoid spamming the servers and getting rate limited/banned
- Automatic timezone recognition based on automatically-detected sensor location
- Automatic error checking and logging
- Easy user configuration

Keep in mind: The academic literature states that the SDS011 sensor is accurate enough for scientific readings of PM2.5 particulates when humidity-controlled (which we do!), but that the PM10 readings are not reliable enough to be used academically. I recommend focusing on PM2.5 data. Still, this script offers you the ability to download the less accurate PM10 data, too. 

How to use this: 
1. Make sure Python3 or greater is installed (from https://www.python.org/downloads/) 
2. Make sure to download all the required modules from the "requirements.txt" file. 
To do so, open a command window (open the start menu and type in "cmd") and type in "python3 -m pip install" followed by the modules listed in that file.
3. Select the air quality sensors you wish to use. This can be done by going to sensors.community and clicking on any sensor while in PM2.5 mode to view their IDs. Write these down. 
4. (Optional but HIGHTLY recommended) Select the humidity sensors you wish to use. This can be done by going to sensors.community and clicking on any sensor while in humidity mode to view their IDs. Write these down. Ideally, these should be co-located, meaning in the same location as the air quality sensors. 
5. Right-click on the .py file and edit it with a text editor.
6. Paste the IDs of all the air quality sensors you wish to use in the relevant part of the configuration.
7. (Optional but HIGHLY recommended) Pair them to humidity sensors in the configuration by pasting the humidity sensor IDs opposite each sensor.
8. Go through the other parts of the configuration settings. Make sure to set a path to which to save the files!!! I recommend setting a "from" email. You may leave settings you don't understand as-is. The defaults have been optimized.
9. Save the file and exit the text editor.
10. Run the script.
11. Open the target directory and enjoy!

How to interpret the data (keep in mind different files are associated with different data):
- In summary files, the PM2.5 and PM10 values represent the Kohler-adjusted air pollution values for the relevant time period
- "Negative rows" means how many data rows were discarded due to having a negative value (pollutant values cannot be negative, so any negative values are misreadings)
- "High rows" shows you how many data rows were discarded due to showing a particulate concentration >999. This is above the specified maximum the SDS011 sensor can read, so any such reading is erroneous.
- "Spike_rows" shows you how many data rows were deleted due to being above the configurable exclusion modifier (Default: 10x) and not being "acquitted" by another sensor
- "rows_discarded_fog" show you how many data rows were discarded due to being associated with a humidity reading at or above 98%. At this high level of humidity (fog), the readings are so inaccurate that not even the Kohler formula can save them.
- "rows_corrected_k_kohler" shows you how many data rows were corrected using the k-value from the settings due to being associated a relative humidity higher than 70% but lower than 98%.
- "bad_lines_skipped" represents how many data rows were discarded due to being "bad" - i.e. blank, NaN, or filled with non-numerical values. This is corrupted data.
- the "stat_ci" rows show only the statistical error, which assumes all particle sensor readings are perfectly accurate once humidity-adjusted. 
- the adj_ci" rows also add in the instrumental measurement error (±15% and ±10 μg /m³ for the SDS011 sensor; whichever is higher). Use these for your error bars.
- "n_samples" simply shows how many non-discarded data points are left after processing has been finished for the relevant sensor and time period. These are the data points that were used for the statistical calculations.
- The timestamps in the individual sensor files show you the exact time when converted to the local sensor timezone. This is noted when compared to UTC. For example "2024-01-01 02:00:00+02:00" means that the reading was done at 02:00:00 local sensor time, and the local sensor time zone is UTC+2. This also should take into account daylight savings.
- Only particle sensor files have PM2.5 and PM10 data. Only humidity sensor files have temperature and humidity readings. This is normal and is the reason we "pair" them in the configuration. 

Common issues: 
- The script may launch briefly and crash if you don't have the required modules installed. See - instruction 2.
- The script may crash if you don't have enough system memory. I have a built-in feature to try and warn you about this, but it isn't perfect. You should have several megabytes of RAM for each sensor and each day you wish to check.
- The script may also fail if you have misconfigured the saving path. Make sure there is a lowercase letter "r" before the path, and make sure the path is in quotation marks, as the configuration indicates.
- The script will fail if sensor IDs are not typed in fully or correctly. Likewise, if the date is wrong.
- If the sensors you're using have bad location settings, the timezone will default to EET. This should very rarely happen.
- It is possible for the timezone to be misinterpreted if your sensor is very close to a timezone boundary.
- Issues can arise if you pair an air quality sensor with a humidity sensor from another timezone. This is not a good idea, anyway, as the humidity readings are unlikely to be very relevant in this case.
- The humidity sensors I'm assuming you're using are BME280 sensors. I have not yet tested the software with other types of sensors.
- Likewise, I'm assuming all air quality readings are coming from SDS011 sensors. Other sensors may yield incorrect data.
- Other issues may be present, as this software is currently in beta. The author would appreciate any bug reports. 


  Good luck and have fun! 
