#' ---
#' title: "sdms_data_piping"
#' author: "Daniel Furman"
#' date: "2020"
#' ---
#' 
#' This R script pipes presence and absence data for joshua trees
#' 
#' Data citation:
#' GBIF.org (01 November 2020) GBIF Occurrence
#' Download https://doi.org/10.15468/dl.g6swrm 
## ------------------------------------------------------------------------

# Load the necessary libraries for spatial analysis
library(raster)     # For working with raster data
library(rgdal)      # For reading/writing geospatial data
library(dismo)      # For species distribution modeling tools
library(maptools)   # For tools to handle spatial objects

## ------------------------------------------------------------------------

# Load GBIF data (presence data for Joshua trees)
jt_raw <- read.csv('data/GBIF_raw.csv', header = TRUE)  # Read in the GBIF occurrence data for Joshua trees from a CSV file

# Filter the data to keep only records from the US
jt_raw <- jt_raw[which(jt_raw$countryCode=='US'),]

# Create a data frame with two columns for longitude and latitude
jt <- data.frame(matrix(ncol = 2, nrow = length(jt_raw$decimalLongitude)))

# Populate the longitude and latitude columns
jt[,1] <- jt_raw$decimalLongitude   # First column is longitude
jt[,2] <- jt_raw$decimalLatitude    # Second column is latitude

# Remove duplicate rows
jt <- unique(jt)  # Keep only unique coordinates to remove duplicates

# Remove rows with missing values (NA)
jt <- jt[complete.cases(jt),]  # Ensure there are no NA values in the data

# Set column names for the data frame
colnames(jt) <- c('lon','lat')  # Rename columns to 'lon' (longitude) and 'lat' (latitude)

## ------------------------------------------------------------------------

# Set the geographic extent of the study area (longitude and latitude range)
e <- extent(-120,-110,32,38.5)  # Study area: from -120 to -110 longitude and from 32 to 38.5 latitude

# Filter the data to keep only points within the extent
jt <- jt[which(jt$lon>=e[1] & jt$lon<=e[2]),]  # Keep points within the longitude range
jt <- jt[which(jt$lat>=e[3] & jt$lat<=e[4]),]  # Keep points within the latitude range

# Download WorldClim bioclimatic variables for the study area
bioclim.data <- getData(name = "worldclim",  # Name of the dataset (WorldClim)
                        var = "bio",         # Bioclimatic variables (temperature, precipitation, etc.)
                        res = 2.5,           # Resolution of the data (2.5 arc-minutes)
                        path = "data/")      # Save the downloaded data to the 'data' folder

# Crop the bioclimatic data to the study area extent
bioclim.data <- crop(bioclim.data, e*1.25)  # Crop the data, slightly expanding the extent (by 1.25x)

# Write each bioclimatic variable as a separate raster file to the 'data' folder
for (i in c(1:19)){
  writeRaster(bioclim.data[[i]], paste('data/bclim', i, sep = ''),
              format="ascii", overwrite=TRUE)  # Save rasters as ASCII files, one for each bioclimatic variable
}

## ------------------------------------------------------------------------

# Sample background points for model training (randomly distributed points)
# The number of background points is double the number of presence points
bg <- randomPoints(bioclim.data[[1]], length(jt)*2, ext=e, extf = 1.25)  
colnames(bg) <- c('lon','lat')  # Set column names for background points ('lon' and 'lat')

# Combine presence points (Joshua tree data) and background points
train <- rbind(jt, bg)  # Combine presence points (jt) and background points (bg) into one dataset

# Create a column for presence/absence (1 for presence, 0 for absence)
pa_train <- c(rep(1, nrow(jt)), rep(0, nrow(bg)))  # 1 for presence points, 0 for background points

# Combine the presence/absence labels with the longitude and latitude data
train <- data.frame(cbind(CLASS=pa_train, train))  # Combine labels (CLASS) with coordinates (lon, lat)

## ------------------------------------------------------------------------

# Create spatial points data with the same projection as the bioclimatic data
crs <- crs(bioclim.data[[1]])  # Extract the coordinate reference system from the bioclimatic data

# Randomize the order of the training data points
train <- train[sample(nrow(train)),]  # Shuffle the rows of the data

# Create a data frame for the presence/absence class labels
class.pa <- data.frame(train[,1])  # Extract the 'CLASS' column (presence/absence labels)
colnames(class.pa) <- 'CLASS'  # Name the column 'CLASS'

# Create a spatial points data frame for the training data (coordinates + class labels)
dataMap.jt  <- SpatialPointsDataFrame(train[,c(2,3)], class.pa,  # Use the lon/lat columns and class labels
                                      proj4string = crs)  # Apply the coordinate reference system (crs)

# Write the spatial points data as a shapefile for use in GIS applications
writeOGR(dataMap.jt, 'data/jtree.shp','jtree', driver='ESRI Shapefile')  # Save the data as a shapefile in the 'data' folder

## ------------------------------------------------------------------------

# Plot the first bioclimatic variable with Joshua tree points and background points
plot(bioclim.data[[1]], main='Bioclim 1')  # Plot the first bioclimatic variable
points(bg, col='red', pch = 16,cex=.3)  # Plot background points in red
points(jt, col='black', pch = 16,cex=.3)  # Plot Joshua tree points in black

# Overlay a world map outline on the plot
plot(wrld_simpl, add=TRUE, border='dark grey')  # Add country borders from a world map, with dark grey borders