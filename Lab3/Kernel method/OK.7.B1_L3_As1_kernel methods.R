# Set the seed for reproducibility
set.seed(1234567890)

# Load the geosphere package for distance calculations
library(geosphere)

# Read station and temperature data
stations <- read.csv("../Data/stations.csv", fileEncoding = "latin1")
temps <- read.csv("../Data/temps50k.csv")

# Merge station and temperature data by station number
st <- merge(stations, temps, by = "station_number")

# Parameters for Gaussian kernel calculations
h_distance <- 300000  # Bandwidth for distance kernel (in meters)
h_date <- 40          # Bandwidth for date kernel (in days)
h_time <- 2           # Bandwidth for time kernel (in hours)

# Coordinates of the point of interest
a <- 58.4274          # Latitude
b <- 14.826           # Longitude

# Date and times of interest for prediction
date <- "2013-07-04"
times_input <- c("04:00:00", "06:00:00", "08:00:00", "10:00:00",
                 "12:00:00", "14:00:00", "16:00:00", "18:00:00",
                 "20:00:00", "22:00:00", "24:00:00")
temp <- vector(length = length(times_input))  # Initialize predicted temperatures

# Prepare coordinates and time data
coordinate <- append(st$longitude, st$latitude)  # Combine longitude and latitude
#coordinate_2<-data.frame(st$longitude, st$latitude)
st_coordiante <- matrix(coordinate, ncol = 2)    # Convert to matrix of coordinates
st_date <- as.Date(st$date)                      # Convert station dates to Date objects
date <- as.Date(date)                            # Convert prediction date to Date object
st_time <- as.POSIXct(st$time, format = "%H:%M:%S")  # Convert station times to POSIXct
times <- as.POSIXct(times_input, format = "%H:%M:%S")  # Convert times of interest to POSIXct

# Initialize lists and vectors for calculations
k_result <- list()             # Store kernel results for each time
pred_numerator_mult <- c()     # Numerator for multiplication method
pred_denominator_mult <- c()   # Denominator for multiplication method
pred_numerator_sum <- c()      # Numerator for summation method
pred_denominator_sum <- c()    # Denominator for summation method
pred_temp_mult <- c()          # Predicted temperatures (multiplication method)
pred_temp_sum <- c()           # Predicted temperatures (summation method)

# Loop over each time of interest
for (j in 1:length(times)) {
  # Initialize matrix for kernel calculations
  k_result[[j]] <- matrix(NA, nrow = nrow(st_coordiante), ncol = 7)
  colnames(k_result[[j]]) <- c("Distance", 'k_distance', 'Day_diff', 'k_day',
                               'Time_diff', "k_time", 'k_temp')

  # Loop over each station
  for (i in (1:dim(st_coordiante)[1])) {
    # Calculate physical distance from station to point of interest
    k_result[[j]][i, 1] <- distHaversine(st_coordiante[i, ], c(b, a))
    k_result[[j]][i, 2] <- exp(-k_result[[j]][i, 1]^2 / 2 / h_distance^2)  # Gaussian kernel

    # Calculate difference in days and apply Gaussian kernel
    k_result[[j]][i, 3] <- abs(as.numeric(format(st_date[i], "%j")) - as.numeric(format(date, "%j")))
    k_result[[j]][i, 3] <- min(k_result[[j]][i, 3], 365 - k_result[[j]][i, 3])
    k_result[[j]][i, 4] <- exp(-k_result[[j]][i, 3]^2 / 2 / h_date^2)

    # Calculate difference in hours and apply Gaussian kernel
    k_result[[j]][i, 5] <- abs(as.numeric(format(st_time[i], "%H")) - as.numeric(format(times[j], "%H")))
    k_result[[j]][i, 5] <- min(k_result[[j]][i, 5], 24 - k_result[[j]][i, 5])
    k_result[[j]][i, 6] <- exp(-k_result[[j]][i, 5]^2 / 2 / h_time^2)

    # Store air temperature
    k_result[[j]][i, 7] <- st$air_temperature[i]
  }

  # Exclude rows with timestamps after the prediction time
  exclude_rows <- rep(FALSE, nrow(k_result[[j]]))
  for (k in (1:nrow(k_result[[j]]))) {
    if (as.numeric(st_date[k] - date) > 0 ||
        ((as.numeric(st_date[k] - date) == 0) &&
         (as.numeric(difftime(st_time[k], times[j]))) > 0)) {
      exclude_rows[k] <- TRUE
    }
  }
  k_result[[j]] <- k_result[[j]][!exclude_rows, ]

  # Calculate predictions using multiplication method
  pred_numerator_mult[j] <- sum(k_result[[j]][, 2] * k_result[[j]][, 4] * k_result[[j]][, 6] * k_result[[j]][, 7])
  pred_denominator_mult[j] <- sum(k_result[[j]][, 2] * k_result[[j]][, 4] * k_result[[j]][, 6])
  pred_temp_mult[j] <- pred_numerator_mult[j] / pred_denominator_mult[j]

  # Calculate predictions using summation method
  pred_numerator_sum[j] <- sum((k_result[[j]][, 2] + k_result[[j]][, 4] + k_result[[j]][, 6]) * k_result[[j]][, 7])
  pred_denominator_sum[j] <- sum(k_result[[j]][, 2] + k_result[[j]][, 4] + k_result[[j]][, 6])
  pred_temp_sum[j] <- pred_numerator_sum[j] / pred_denominator_sum[j]
}

# Plot the results of kernel functions and predictions
library(ggplot2)

# Plot for distance kernel
k_result_df_mul <- as.data.frame(k_result[[1]])
e <- ggplot(k_result_df_mul, aes(x = Distance, y = k_distance)) +
  geom_line(color = 'red') +
  geom_point()

# Plot for day kernel
f <- ggplot(k_result_df_mul, aes(x = Day_diff, y = k_day)) +
  geom_line(color = 'red') +
  geom_point()

# Plot for time kernel
g <- ggplot(k_result_df_mul, aes(x = Time_diff, y = k_time)) +
  geom_line(color = 'red') +
  geom_point()

# Combine prediction results into a dataframe
hours <- as.numeric(as.character(substr(times_input, 1, 2)))
pred_temp_df <- cbind(hours, pred_temp_mult, pred_temp_sum)

# Plot predicted temperatures over hours
h <- ggplot(pred_temp_df) +
  geom_line(aes(x = hours, y = pred_temp_mult, group = 1), color = 'red') +
  geom_point(aes(x = hours, y = pred_temp_mult, group = 1), color = 'red') +
  geom_line(aes(x = hours, y = pred_temp_sum, group = 2), color = 'blue') +
  geom_point(aes(x = hours, y = pred_temp_sum, group = 2), color = 'blue') +
  geom_text(
    aes(x = hours[length(hours)], y = pred_temp_mult[length(pred_temp_mult)]),
    label = "Multiplication Method", color = "red", vjust = -0.5, hjust = 0
  ) +
  geom_text(
    aes(x = hours[length(hours)], y = pred_temp_sum[length(pred_temp_sum)]),
    label = "Summation Method", color = "blue", vjust = -0.5, hjust = 0
  )

# Display the final plot
print(h)


