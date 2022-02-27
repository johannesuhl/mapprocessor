# mapprocessor: A pipeline to convert color + texture information from the USGS historical topographic map collection into aggregated, seamless geospatial gridded surfaces

This script allows for accessing >200,000 scanned and georeferenced historical topographic maps from the United States Geological Survey (USGS) historical topographic map collection (HTMC), available in GeoTIFF format on AWS S3.
Specifically, the user can set up a query (e.g., searching for maps of a specific scale, geographic region, or time period). These map sheets are then downloaded, and the map collar will be removed automatically.
Then, the GeoTIFF files will be aggregated (color moments, and optionally, Local Binary Pattern texture descriptors) into coarser resolution grids.
Lastly, the aggregated map data can be converted into seamless layers using a mosaicking technique.
This pipeline aims to facilitate the mass processing of the USGS HTMC historical map archive and has been employed in several projects, such as:

Uhl, J. H., Leyk, S., Li, Z., Duan, W., Shbita, B., Chiang, Y. Y., & Knoblock, C. A. (2021). Combining remote-sensing-derived data and historical maps for long-term back-casting of urban extents. Remote Sensing, 13(18), 3672. https://doi.org/10.3390/rs13183672

Uhl, J. H., Leyk, S., Chiang, Y. Y., & Knoblock, C. A. (2022). Towards the automated large-scale reconstruction of past road networks from historical maps. arXiv preprint arXiv:2202.04883. https://arxiv.org/abs/2202.04883

Uhl, Johannes H. (2021). How the U.S. was mapped - visualizing 130 years of topographic mapping in the conterminous U.S.. figshare. Media. https://doi.org/10.6084/m9.figshare.17209433.v2 

The LinkedMap project: https://usc-isi-i2.github.io/linked-maps/

<img width="1000" alt="Example: fully automatically generated seamless layer of 1:62,500 USGS topo maps in Rhode Island from around 1890. Data source: USGS HTMC. Basemap source: The OpenStreetMap Contributors." src="https://github.com/johannesuhl/mapprocessor/blob/master/RI.jpg">
