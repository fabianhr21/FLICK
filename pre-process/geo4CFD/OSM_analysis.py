import gc
import os
import math
import warnings
import geopandas as gpd
import osmnx as ox
import pandas as pd
import numpy as np
from shapely.geometry import box, Polygon, LineString, MultiLineString, Point
from shapely import affinity
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import glob

# Try importing rasterio for height extraction and rasterization
try:
    import rasterio
    from rasterio import features
    from rasterio.mask import mask
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: 'rasterio' library not found. Raster height extraction and Alignedness metrics will be disabled.")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class UrbanMorphologyAnalyzer:
    def __init__(self, city_name, ua_file_path=None, eubucco_file_path=None, height_raster_path=None, grid_size_km=1.0):
        self.city_name = city_name
        self.ua_file_path = ua_file_path
        self.eubucco_file_path = eubucco_file_path
        self.height_raster_path = height_raster_path
        self.grid_size = grid_size_km * 1000  # meters
        
        # Initialize CRS as None; it will be determined dynamically in generate_grid
        self.crs_proj = None  
        self.eubucco_crs = None
        
        self.output_dir = f"output_data_NEWPARAMS/output_{self.city_name.replace(' ', '_').replace(',', '')}"
        self.city_boundary = None
        self.grid = None
        
        # Road width defaults
        self.street_widths = {
            'motorway': 20, 'trunk': 15, 'primary': 12, 'secondary': 10,
            'tertiary': 8, 'residential': 6, 'living_street': 5,
            'pedestrian': 4, 'service': 4, 'cycleway': 2, 'footway': 2, 'path': 2
        }

        # Configure OSMnx
        ox.settings.timeout = 1800
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_grid(self):
        """Generates the analysis grid."""
        print(f"1. Fetching boundary for {self.city_name}...")
        try:
            # 1. Get boundary in WGS84 (Lat/Lon)
            gdf_wgs84 = ox.geocode_to_gdf(self.city_name)
            
            if gdf_wgs84.empty:
                print(f"   Error: Could not find location for '{self.city_name}'")
                return None

            # 2. Automatically project to the best local UTM zone (meters)
            self.crs_proj = gdf_wgs84.estimate_utm_crs()
            self.city_boundary = gdf_wgs84.to_crs(self.crs_proj)
            
            print(f"   -> Detected optimal projection: {self.crs_proj}")
            
            minx, miny, maxx, maxy = self.city_boundary.total_bounds
            
            cols = np.arange(minx, maxx, self.grid_size)
            rows = np.arange(miny, maxy, self.grid_size)
            
            polygons = []
            for x in cols:
                for y in rows:
                    poly = box(x, y, x + self.grid_size, y + self.grid_size)
                    if poly.intersects(self.city_boundary.geometry.iloc[0]):
                        polygons.append(poly)
            
            if not polygons:
                print("   Error: No grid cells created (city might be smaller than grid size).")
                return None

            self.grid = gpd.GeoDataFrame({'geometry': polygons}, crs=self.crs_proj)
            self.grid['grid_id'] = range(len(self.grid))
            # Calculate centroids once
            self.grid['center_lat'] = self.grid.to_crs("EPSG:4326").centroid.y
            self.grid['center_lon'] = self.grid.to_crs("EPSG:4326").centroid.x
            self.grid['cell_area_m2'] = self.grid.area
            
            # Check EUBUCCO file CRS if path is provided
            if self.eubucco_file_path and os.path.exists(self.eubucco_file_path):
                try:
                    import fiona
                    with fiona.open(self.eubucco_file_path) as src:
                        self.eubucco_crs = src.crs
                except Exception:
                    self.eubucco_crs = "EPSG:3035" # Default fallback

            print(f"   -> Generated {len(self.grid)} grid cells.")
            return self.grid
        except Exception as e:
            print(f"Error during grid generation: {e}")
            return None

    def calculate_projected_width(self, poly: Polygon, wind_dir_deg: float):
        """
        Return projected width of polygon perpendicular to wind direction.
        wind_dir_deg: wind direction in degrees where 0 = wind from North (towards South).
        """
        if not isinstance(poly, Polygon):
            return 0.0
        try:
            xs, ys = poly.exterior.coords.xy
            pts = np.vstack((xs, ys)).T
            # Rotate points to align wind axis with X-axis
            theta = np.deg2rad(-(90 - wind_dir_deg))
            R = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
            rotated = pts.dot(R.T)
            minx, maxx = rotated[:,0].min(), rotated[:,0].max()
            return maxx - minx
        except Exception:
            return 0.0
            
    def calculate_orientation_entropy(self, buildings_gdf, num_bins=36):
        """
        Calculates the normalized orientation entropy (phi) of building facets.
        Based on Shannon entropy relative to a square (min) and circle (max).
        """
        if buildings_gdf.empty:
            return 0.0

        try:
            # 1. Collect all wall segments from all buildings in the cell
            all_lengths = []
            all_angles = []

            # Vectorized approach to extracting coords might be complex due to jagged arrays
            # Iterating geometry is acceptable for single grid cell scale
            for geom in buildings_gdf.geometry:
                if not isinstance(geom, Polygon): continue
                
                # Get exterior coordinates
                # Note: last point duplicates first in Polygon
                coords = np.array(geom.exterior.coords)
                if len(coords) < 2: continue
                
                # Vectors (dx, dy)
                diffs = np.diff(coords, axis=0) # shape (N-1, 2)
                dx = diffs[:, 0]
                dy = diffs[:, 1]
                
                # Lengths
                lengths = np.hypot(dx, dy)
                
                # Angles (0 to 360 degrees, clockwise from North to match typical compass, or cartesian?)
                # np.arctan2 returns -pi to pi (0 is East, positive CCW)
                # Let's standardize to 0-360 mathematical (East=0, CCW) or Compass?
                # Entropy is invariant to rotation reference as long as consistent.
                # Using standard math degrees (0-360)
                angles = np.degrees(np.arctan2(dy, dx))
                angles = angles % 360
                
                all_lengths.append(lengths)
                all_angles.append(angles)
            
            if not all_lengths:
                return 0.0
                
            # Flatten arrays
            all_lengths = np.concatenate(all_lengths)
            all_angles = np.concatenate(all_angles)
            
            # 2. Binning
            # F = 36 bins (10 degrees each)
            hist, _ = np.histogram(all_angles, bins=num_bins, range=(0, 360), weights=all_lengths)
            
            # 3. Calculate Probabilities P(w_f)
            total_length = np.sum(hist)
            if total_length == 0: return 0.0
            
            probs = hist / total_length
            
            # Remove zero probabilities for log calculation
            probs = probs[probs > 0]
            
            # 4. Shannon Entropy H0
            H0 = -np.sum(probs * np.log(probs))
            
            # 5. Normalization
            # H_max = ln(F) -> Uniform distribution (Circle)
            # H_min = ln(4) -> 4 orthogonal bins (Square)
            
            H_max = np.log(num_bins)
            H_min = np.log(4) 
            
            # Clamp H0 between min and max theoretically
            # (H0 can be < H_min if buildings align to just 1 or 2 directions perfectly, 
            # effectively H_min for a "square" implies 4 directions. A single wall is more ordered than a square.)
            # However, the formula provided is specific.
            # Using the prompt's formula directly:
            
            # Avoid division by zero
            if H_max == H_min: 
                return 0.0
            
            # Note: If H0 < H_min (e.g. all parallel walls), the term inside square could be negative.
            # The prompt implies a range [0, 1].
            # We assume H_min here refers to the "disorder" of a regular grid.
            # Usually entropy 0 is perfectly aligned (1 bin). 
            # If the user defines H_min = ln(4), they define a square as the baseline "zero" point?
            # Or is it a range mapping?
            # Prompt: "normalized by two theoretical extremes Hmax ... and minimum Hmin"
            # Equation: phi ~ ((H0 - Hmin) / (Hmax - Hmin))^2
            # If H0 < Hmin, we clamp to 0? Or abs? "0 <= phi <= 1.0" suggests clamping.
            
            val = (H0 - H_min) / (H_max - H_min)
            val = max(0.0, min(1.0, val)) # Clamp to [0, 1] before squaring
            
            phi = val ** 2
            return phi

        except Exception as e:
            # print(f"Entropy Calc Error: {e}")
            return 0.0
    
    def parse_height_str(self, s):
        """Parses height string to float, handling units and ranges."""
        if pd.isna(s): return np.nan
        s = str(s).strip().lower()
        s = s.replace('approx', '').replace('~', '')
        
        # 1. Ranges (e.g. '10-12')
        m_range = re.search(r'^([0-9]+(?:\.[0-9]+)?)\s*(?:[-–]|to)\s*([0-9]+(?:\.[0-9]+)?)', s)
        if m_range:
            try:
                return (float(m_range.group(1)) + float(m_range.group(2))) / 2.0
            except ValueError: pass

        # 2. Single number with units
        m_single = re.search(r'^([0-9]+(?:\.[0-9]+)?)', s)
        if m_single:
            try:
                val = float(m_single.group(1))
                if 'ft' in s or 'feet' in s or "'" in s:
                    return val * 0.3048
                return val
            except ValueError: pass
                
        return np.nan

    def enrich_with_raster_heights(self, buildings_gdf):
        """Samples building heights from the provided TIF raster."""
        if not HAS_RASTERIO or not self.height_raster_path or not os.path.exists(self.height_raster_path):
            return buildings_gdf

        try:
            with rasterio.open(self.height_raster_path) as src:
                # Reproject buildings to Raster CRS
                buildings_reproj = buildings_gdf.to_crs(src.crs)
                raster_heights = []
                
                for geom in buildings_reproj.geometry:
                    try:
                        out_image, _ = mask(src, [geom], crop=True, nodata=np.nan)
                        data = out_image[0]
                        if src.nodata is not None:
                            valid_data = data[data != src.nodata]
                        else:
                            valid_data = data[~np.isnan(data)]
                        
                        valid_data = valid_data[valid_data > 0]
                        
                        if valid_data.size > 0:
                            raster_heights.append(np.nanmean(valid_data))
                        else:
                            raster_heights.append(np.nan)
                    except Exception:
                        raster_heights.append(np.nan)
                
                buildings_gdf['raster_height'] = raster_heights
                
        except Exception as e:
            print(f"      Warning: Failed to extract raster heights: {e}")
            
        return buildings_gdf
    
    def estimate_heights(self, df):
        # 1. Parse OSM tags
        for col in ['height', 'roof:height', 'maxheight']:
            if col in df.columns:
                df[f'{col}_m'] = df[col].astype(str).apply(self.parse_height_str)
        
        # 2. Parse Levels
        if 'building:levels' in df.columns:
            df['levels'] = pd.to_numeric(df['building:levels'], errors='coerce')
        elif 'levels' in df.columns:
            df['levels'] = pd.to_numeric(df['levels'], errors='coerce')
        else:
            df['levels'] = np.nan

        # Floor height logic
        def floor_height(row):
            btype = str(row.get('building', '')).lower()
            if 'residential' in btype: return 2.9
            if 'commercial' in btype or 'industrial' in btype: return 3.5
            return 3.0

        df['floor_h'] = df.apply(floor_height, axis=1)
        df['height_from_levels'] = df['levels'] * df['floor_h']

        # 3. Priority: Raster > OSM Tag > OSM Level
        if 'raster_height' in df.columns:
            df['est_height'] = pd.to_numeric(df['raster_height'], errors='coerce')
        else:
            df['est_height'] = np.nan

        fallback_cols = ['height_m', 'roof:height_m', 'maxheight_m', 'height_from_levels']
        for col in fallback_cols:
            if col in df.columns:
                df['est_height'] = df['est_height'].fillna(df[col])
            
        # 4. Fallback median
        median_h = df['est_height'].median()
        df['est_height'] = df['est_height'].fillna(median_h if not np.isnan(median_h) else 9.0)
        return df

    # def calculate_volumetric_porosity(self,buildings_gdf, total_area_m2, fixed_height=None):
    #     """
    #     Calculates Volumetric Porosity (0.0 to 1.0).
        
    #     Args:
    #         buildings_gdf (GeoDataFrame): Polygons with a 'height' column.
    #         total_area_m2 (float): The area of the ROI (Region of Interest) circle or polygon.
    #         fixed_height (float): Optional. If None, uses max(building_height).
    #                             For comparing different districts, distinct H_ref is 
    #                             usually better (Canopy Porosity).
    #     """
    #     # 1. Calculate Building Volume
    #     # Ensure we have a height. If missing, assume generic 3m or fail.
    #     if 'height' not in buildings_gdf.columns:
    #         raise ValueError("GeoDataFrame must have a 'height' column for volumetric calcs.")
        
    #     # Volume = Area * Height
    #     print(buildings_gdf.head())
    #     print(buildings_gdf.geometry.area)
    #     print(buildings_gdf['height'])
    #     buildings_gdf['volume'] = buildings_gdf.geometry.area * buildings_gdf['height']
    #     total_bldg_vol = buildings_gdf['volume'].sum()
        
    #     # 2. Define Domain Volume
    #     # If no fixed height is given, the "Canopy Layer" ends at the tallest building.
    #     if fixed_height:
    #         h_ref = fixed_height
    #     else:
    #         h_ref = buildings_gdf['height'].max()
        
    #     total_domain_vol = total_area_m2 * h_ref
        
    #     # 3. Calculate Porosity (Void Fraction)
    #     # Porosity = (Total Vol - Bldg Vol) / Total Vol
    #     porosity = (total_domain_vol - total_bldg_vol) / total_domain_vol
        
    #     return porosity, h_ref


    def calculate_aspect_ratio(self, avg_height, avg_street_width):
        """
        Calculates the aspect ratio (H/W) between building height and street width.
        Handles division by zero.
        """
        # Ensure we are working with numpy arrays or pandas Series for vectorization
        h = np.asarray(avg_height, dtype=float)
        w = np.asarray(avg_street_width, dtype=float)
        
        # Calculate ratio where width > 0, else 0
        ratio = np.where(w > 0, h / w, 0.0)
        return ratio
    
    def calculate_alignedness_metrics(self, cell_poly, buildings_gdf, wind_dir=0, resolution=2.0):
        """
        Calculates Mean Alignedness (gamma_m) and Modified Alignedness (gamma_m_star).
        Uses rasterization for efficiency.
        """
        if not HAS_RASTERIO or buildings_gdf.empty:
            return 0.0, 0.0

        try:
            # 1. Rotate everything so Wind Direction aligns with X-axis (Left->Right)
            origin = cell_poly.centroid
            rot_angle = 90 - wind_dir 
            cell_rot = affinity.rotate(cell_poly, rot_angle, origin=origin)
            
            bldgs_rot = buildings_gdf.copy()
            bldgs_rot.geometry = bldgs_rot.geometry.apply(lambda g: affinity.rotate(g, rot_angle, origin=origin))
            
            # 2. Rasterize
            minx, miny, maxx, maxy = cell_rot.bounds
            width = int((maxx - minx) / resolution)
            height = int((maxy - miny) / resolution)
            
            if width <= 0 or height <= 0: return 0.0, 0.0

            transform = from_origin(minx, maxy, resolution, resolution)
            shapes = ((geom, val) for geom, val in zip(bldgs_rot.geometry, bldgs_rot['est_height']))
            
            height_grid = features.rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0, 
                default_value=1,
                dtype='float32'
            )
            
            # 3. Analyze Rows
            row_means = []
            row_mods = []
            domain_length = width * resolution 
            
            for row in height_grid:
                is_open = (row == 0)
                if not np.any(is_open):
                    row_means.append(0.0)
                    row_mods.append(0.0)
                    continue
                
                padded = np.pad(is_open, (1, 1), 'constant', constant_values=False)
                diff = np.diff(padded.astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                lengths_m = (ends - starts) * resolution
                
                if len(lengths_m) == 0:
                    row_means.append(0.0)
                    row_mods.append(0.0)
                    continue
                
                # Mean Alignedness
                row_means.append(np.max(lengths_m) / domain_length)
                
                # Modified Alignedness
                ratios = []
                for i, l_val in enumerate(lengths_m):
                    start_idx = starts[i]
                    if start_idx == 0: continue
                    h_upwind = row[start_idx - 1]
                    if h_upwind > 0:
                        ratios.append(l_val / h_upwind)
                
                if ratios:
                    row_mods.append(np.max(ratios))
                else:
                    row_mods.append(0.0)
            
            return np.mean(row_means), np.mean(row_mods)

        except Exception:
            return 0.0, 0.0

    def _fetch_data_for_batch(self, batch_poly):
        """Downloads data for batch. Uses EUBUCCO for buildings if available."""
        poly_wgs84 = gpd.GeoSeries([batch_poly], crs=self.crs_proj).to_crs("EPSG:4326").iloc[0]
        data = {}

        # A. Buildings (EUBUCCO vs OSM)
        try:
            if self.eubucco_file_path and os.path.exists(self.eubucco_file_path):
                target_crs = self.eubucco_crs if self.eubucco_crs else "EPSG:3035"
                poly_eubucco = gpd.GeoSeries([batch_poly], crs=self.crs_proj).to_crs(target_crs).iloc[0]
                bldgs = gpd.read_file(self.eubucco_file_path, bbox=poly_eubucco.bounds)
                if not bldgs.empty:
                    bldgs = bldgs.to_crs(self.crs_proj)
                    data['buildings'] = bldgs[bldgs.geom_type.isin(['Polygon', 'MultiPolygon'])]
                else:
                    data['buildings'] = gpd.GeoDataFrame(crs=self.crs_proj, geometry=[])
            else:
                bldgs = ox.features_from_polygon(poly_wgs84, tags={"building": True})
                if not bldgs.empty:
                    bldgs = bldgs.to_crs(self.crs_proj)
                    data['buildings'] = bldgs[bldgs.geom_type.isin(['Polygon', 'MultiPolygon'])]
                else:
                    data['buildings'] = gpd.GeoDataFrame(crs=self.crs_proj, geometry=[])
        except Exception:
            data['buildings'] = gpd.GeoDataFrame(crs=self.crs_proj, geometry=[])

        # B. Streets (OSM)
        try:
            streets = ox.features_from_polygon(poly_wgs84, tags={"highway": ["motorway", "trunk", "primary", 
                                                                           "secondary", "tertiary", "residential", 
                                                                           "living_street", "pedestrian", "service"]})
            if not streets.empty:
                streets = streets.to_crs(self.crs_proj)
                data['streets'] = streets[streets.geom_type.isin(['LineString', 'MultiLineString'])]
            else:
                data['streets'] = gpd.GeoDataFrame(crs=self.crs_proj, geometry=[])
        except:
            data['streets'] = gpd.GeoDataFrame(crs=self.crs_proj, geometry=[])

        # C. Land Use (OSM)
        try:
            lu = ox.features_from_polygon(poly_wgs84, tags={"landuse": ["grass", "forest", "meadow", "orchard", "recreation_ground", "village_green", 
                                                                      "industrial", "commercial", "retail"],
                                                          "natural": ["wood", "scrub", "heath", "grassland"],
                                                          "leisure": ["park", "garden", "golf_course", "pitch"]})
            if not lu.empty:
                lu = lu.to_crs(self.crs_proj)
                data['landuse'] = lu[lu.geom_type.isin(['Polygon', 'MultiPolygon'])]
            else:
                data['landuse'] = gpd.GeoDataFrame(crs=self.crs_proj, geometry=[])
        except:
            data['landuse'] = gpd.GeoDataFrame(crs=self.crs_proj, geometry=[])
            
        return data



    # --- Example Usage ---
    # area_roi = 500 * 500 * 3.14159... (if using radius) or gdf.total_bounds area
    # p, h = calculate_volumetric_porosity(gdf, area_roi)
    # print(f"Porosity: {p:.3f} (within canopy height {h}m)")

    def run_analysis_optimized(self, batch_size=500):
        if self.grid is None:
            if self.generate_grid() is None:
                return None
        
        total_cells = len(self.grid)
        num_batches = int(math.ceil(total_cells / batch_size))
        
        print(f"2. Starting streaming analysis in {num_batches} batches (Batch Size: {batch_size})...")
        if self.eubucco_file_path:
            print("   Source: EUBUCCO (Local)")
        else:
            print("   Source: OpenStreetMap (Internet)")
        if self.height_raster_path:
            print(f"   Height Source: Raster ({os.path.basename(self.height_raster_path)}) > OSM Tags")
        
        final_results_list = []
        count_plots = 0

        for i in tqdm(range(num_batches), desc="Processing Batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_cells)
            batch_grid = self.grid.iloc[start_idx:end_idx].copy()
            
            minx, miny, maxx, maxy = batch_grid.total_bounds
            batch_poly = box(minx, miny, maxx, maxy)
            osm_data = self._fetch_data_for_batch(batch_poly)
            batch_results = batch_grid[['grid_id', 'geometry', 'center_lat', 'center_lon', 'cell_area_m2']].copy()
            
            # --- BUILDINGS ---
            if not osm_data['buildings'].empty:
                bldgs_overlay = gpd.overlay(osm_data['buildings'], batch_grid[['geometry', 'grid_id']], how='intersection')
                
                if not bldgs_overlay.empty:
                    # Enrich and Estimate
                    bldgs_overlay = self.enrich_with_raster_heights(bldgs_overlay)
                    bldgs_overlay = self.estimate_heights(bldgs_overlay)
                    
                    bldgs_overlay['footprint_area'] = bldgs_overlay.area
                    pai_stats = bldgs_overlay.groupby('grid_id')['footprint_area'].sum()
                    dens_stats = bldgs_overlay.groupby('grid_id').size()
                    h_mean = bldgs_overlay.groupby('grid_id')['est_height'].mean()
                    h_std = bldgs_overlay.groupby('grid_id')['est_height'].std().fillna(0)
                    
                    bldgs_overlay['proj_width'] = bldgs_overlay.geometry.apply(lambda geom: self.calculate_projected_width(geom, wind_dir_deg=0))
                    bldgs_overlay['frontal_area'] = bldgs_overlay['proj_width'] * bldgs_overlay['est_height']
                    fai_stats = bldgs_overlay.groupby('grid_id')['frontal_area'].sum()
                                        # --- ADDED: Volumetric Porosity Prep ---
                    # Volume = Footprint * Height
                    bldgs_overlay['volume'] = bldgs_overlay['footprint_area'] * bldgs_overlay['est_height']
                    vol_stats = bldgs_overlay.groupby('grid_id')['volume'].sum()
                    # Domain Height (Reference) = Max Building Height
                    max_h_stats = bldgs_overlay.groupby('grid_id')['est_height'].max()

                    # ALIGNEDNESS & ENTROPY CALCULATION (Per Cell)
                    mean_align_list = []
                    mod_align_list = []
                    entropy_list = []
                    
                    grouped_bldgs = dict(tuple(bldgs_overlay.groupby('grid_id')))
                    
                    for gid in batch_results['grid_id']:
                        if gid in grouped_bldgs:
                            cell_poly = batch_grid.loc[batch_grid['grid_id'] == gid, 'geometry'].values[0]
                            # Alignedness
                            ma, moa = self.calculate_alignedness_metrics(cell_poly, grouped_bldgs[gid], wind_dir=0)
                            mean_align_list.append(ma)
                            mod_align_list.append(moa)
                            # Entropy
                            phi = self.calculate_orientation_entropy(grouped_bldgs[gid])
                            entropy_list.append(phi)
                            # Volumetric Porosity
                            # vol_pot, h_ref = self.calculate_volumetric_porosity(grouped_bldgs[gid], total_area_m2=cell_poly.area)
                        else:
                            mean_align_list.append(0.0)
                            mod_align_list.append(0.0)
                            entropy_list.append(0.0)
                            
                    batch_results['mean_alignedness'] = mean_align_list
                    batch_results['modified_alignedness'] = mod_align_list
                    batch_results['orientation_entropy'] = entropy_list
                    # batch_results['volumetric_porosity'] = vol_pot
                    
                    batch_results = batch_results.merge(pai_stats.rename("total_footprint"), on='grid_id', how='left').fillna(0)
                    batch_results = batch_results.merge(dens_stats.rename("bldg_count"), on='grid_id', how='left').fillna(0)
                    batch_results = batch_results.merge(h_mean.rename("height_mean"), on='grid_id', how='left').fillna(0)
                    batch_results = batch_results.merge(h_std.rename("height_std"), on='grid_id', how='left').fillna(0)
                    batch_results = batch_results.merge(fai_stats.rename("total_frontal"), on='grid_id', how='left').fillna(0)
                    batch_results = batch_results.merge(vol_stats.rename("total_volume"), on='grid_id', how='left').fillna(0)
                    batch_results = batch_results.merge(max_h_stats.rename("max_height"), on='grid_id', how='left').fillna(0)
                    batch_results['PAI_BuiltPlan'] = batch_results['total_footprint'] / batch_results['cell_area_m2']
                    batch_results['FAI_FrontalArea'] = batch_results['total_frontal'] / batch_results['cell_area_m2']
                    batch_results['bldg_density_km2'] = batch_results['bldg_count'] / (batch_results['cell_area_m2'] / 1e6)
                    # --- ADDED: Volumetric Porosity Calculation ---
                    # Porosity = (Total Domain Vol - Building Vol) / Total Domain Vol
                    # Domain Vol = Cell Area * Max Height
                    # If Max Height is 0 (no buildings), Porosity is 1.0 (empty space).
                    
                    domain_vol = batch_results['cell_area_m2'] * batch_results['max_height']
                    # Handle division by zero where max_height is 0
                    batch_results['volumetric_porosity'] = np.where(
                        domain_vol > 0,
                        (domain_vol - batch_results['total_volume']) / domain_vol,
                        1.0 # If domain volume is 0, assume empty space (porosity 1.0)
                    )
                    # Clamp to 0-1 range just in case
                    batch_results['volumetric_porosity'] = batch_results['volumetric_porosity'].clip(0, 1)
                    # Visualization
                    for _, row in batch_results.iterrows():
                        gid = row['grid_id']
                        if gid in grouped_bldgs:
                            if row['PAI_BuiltPlan'] > 0.25 and row['height_mean'] > 7:
                                self.visualize_cell(row['geometry'], grouped_bldgs[gid], gid, 
                                                  row['PAI_BuiltPlan'], row['FAI_FrontalArea'])
                                count_plots += 1
                else:
                    for col in ['PAI_BuiltPlan', 'FAI_FrontalArea', 'bldg_density_km2', 'height_mean', 'height_std', 'mean_alignedness', 'modified_alignedness', 'orientation_entropy']:
                        batch_results[col] = 0.0
            else:
                for col in ['PAI_BuiltPlan', 'FAI_FrontalArea', 'bldg_density_km2', 'height_mean', 'height_std', 'mean_alignedness', 'modified_alignedness', 'orientation_entropy']:
                    batch_results[col] = 0.0

            # --- STREETS ---
            if not osm_data['streets'].empty:
                streets_overlay = gpd.overlay(osm_data['streets'], batch_grid[['geometry', 'grid_id']], how='intersection')
                if not streets_overlay.empty:
                    streets_overlay['length'] = streets_overlay.length
                    
                    def map_width(row):
                        hw = row['highway']
                        if isinstance(hw, list): hw = hw[0]
                        return self.street_widths.get(hw, 6.0)
                    
                    streets_overlay['width_val'] = streets_overlay.apply(map_width, axis=1)
                    len_sum = streets_overlay.groupby('grid_id')['length'].sum()
                    streets_overlay['w_x_l'] = streets_overlay['width_val'] * streets_overlay['length']
                    width_sum = streets_overlay.groupby('grid_id')['w_x_l'].sum()
                    
                    def get_angle(geom):
                        if geom.geom_type == 'MultiLineString':
                            if geom.is_empty: return 0
                            geom = max(geom.geoms, key=lambda g: g.length)
                        if geom.geom_type == 'LineString':
                            coords = geom.coords
                            if len(coords) < 2: return 0
                            dx = coords[-1][0] - coords[0][0]
                            dy = coords[-1][1] - coords[0][1]
                            deg = math.degrees(math.atan2(dx, dy))
                            if deg < 0: deg += 360
                            return deg % 180
                        return 0
                    
                    streets_overlay['angle'] = streets_overlay.geometry.apply(get_angle)
                    angle_mean = streets_overlay.groupby('grid_id')['angle'].mean()
                    angle_std = streets_overlay.groupby('grid_id')['angle'].std().fillna(0)
                    
                    batch_results = batch_results.merge(len_sum.rename("street_len_total"), on='grid_id', how='left').fillna(0)
                    batch_results = batch_results.merge(width_sum.rename("width_weighted_sum"), on='grid_id', how='left').fillna(0)
                    batch_results = batch_results.merge(angle_mean.rename("street_angle_mean"), on='grid_id', how='left')
                    batch_results = batch_results.merge(angle_std.rename("street_angle_std"), on='grid_id', how='left')
                    
                    mask = batch_results['street_len_total'] > 0
                    batch_results.loc[mask, 'street_avg_width'] = batch_results.loc[mask, 'width_weighted_sum'] / batch_results.loc[mask, 'street_len_total']
            
            if 'height_mean' in batch_results.columns and 'street_avg_width' in batch_results.columns:
                batch_results['street_aspect_ratio'] = self.calculate_aspect_ratio(
                    batch_results['height_mean'], 
                    batch_results['street_avg_width']
                )
            else:
                batch_results['street_aspect_ratio'] = 0.0

            # --- LAND USE ---
            if not osm_data['landuse'].empty:
                lu_overlay = gpd.overlay(osm_data['landuse'], batch_grid[['geometry', 'grid_id']], how='intersection')
                if not lu_overlay.empty:
                    lu_overlay['area'] = lu_overlay.area
                    
                    green_types = ["grass", "forest", "meadow", "orchard", "recreation_ground", "village_green", "wood", "scrub", "heath", "grassland", "park", "garden", "golf_course", "pitch"]
                    ind_types = ["industrial", "commercial", "retail"]
                    
                    cols = lu_overlay.columns
                    is_green = pd.Series(False, index=lu_overlay.index)
                    is_ind = pd.Series(False, index=lu_overlay.index)
                    
                    for col in ['landuse', 'natural', 'leisure']:
                        if col in cols:
                            is_green |= lu_overlay[col].apply(lambda x: x in green_types if isinstance(x, str) else False)
                            is_ind |= lu_overlay[col].apply(lambda x: x in ind_types if isinstance(x, str) else False)

                    green_stats = lu_overlay[is_green].groupby('grid_id')['area'].sum()
                    ind_stats = lu_overlay[is_ind].groupby('grid_id')['area'].sum()
                    
                    batch_results = batch_results.merge(green_stats.rename("green_area_sqm"), on='grid_id', how='left').fillna(0)
                    batch_results = batch_results.merge(ind_stats.rename("ind_area_sqm"), on='grid_id', how='left').fillna(0)
                    
                    batch_results['green_area_ratio'] = batch_results['green_area_sqm'] / batch_results['cell_area_m2']
                    batch_results['industrial_area_ratio'] = batch_results['ind_area_sqm'] / batch_results['cell_area_m2']

            final_results_list.append(batch_results)
            
            del osm_data
            if 'bldgs_overlay' in locals(): del bldgs_overlay
            if 'streets_overlay' in locals(): del streets_overlay
            if 'lu_overlay' in locals(): del lu_overlay
            gc.collect()

        if not final_results_list:
            print("No results generated.")
            return None
            
        results_df = pd.concat(final_results_list, ignore_index=True)
        
        # --- URBAN ATLAS ---
        if self.ua_file_path and os.path.exists(self.ua_file_path):
            print("4. Processing Urban Atlas...")
            try:
                ua_data = gpd.read_file(self.ua_file_path).to_crs(self.crs_proj)
                ua_overlay = gpd.overlay(ua_data, self.grid[['geometry', 'grid_id']], how='intersection')
                ua_overlay['area'] = ua_overlay.area
                col_name = next((c for c in ua_overlay.columns if 'class' in c.lower() or 'item' in c.lower()), None)
                if col_name:
                    dominant = ua_overlay.sort_values('area', ascending=False).drop_duplicates('grid_id')
                    results_df = results_df.merge(dominant[['grid_id', col_name]], on='grid_id', how='left')
                    results_df.rename(columns={col_name: 'UA_LandUse'}, inplace=True)
            except Exception as e:
                print(f"   Warning: UA processing failed: {e}")
        else:
            results_df['UA_LandUse'] = "N/A"

        # Cleanup columns # DROPED  'mean_alignedness', 'green_area_ratio', 'industrial_area_ratio',  'street_angle_mean', 
        final_cols = ['grid_id', 'geometry', 'center_lat', 'center_lon', 
                      'PAI_BuiltPlan', 'FAI_FrontalArea', 'bldg_density_km2', 
                      'height_mean', 'height_std','modified_alignedness', 'orientation_entropy', 'street_aspect_ratio', 'volumetric_porosity',
                      'street_len_total', 'street_avg_width','street_angle_std','UA_LandUse', 'green_area_ratio']
        
        for c in final_cols:
            if c not in results_df.columns:
                results_df[c] = 0 if 'ratio' in c or 'mean' in c else None

        self.gdf_results = gpd.GeoDataFrame(results_df[final_cols], crs=self.crs_proj)
        
        out_file = os.path.join(self.output_dir, f"{self.city_name.split(',')[0]}_morphology_optimized.geojson")
        self.gdf_results.to_file(out_file, driver='GeoJSON')
        
        self.visualize_city_grid(self.gdf_results)
        
        print(f"Analysis complete. Saved to {out_file}")
        print(f"Generated {count_plots} candidate grid plots.")
        return self.gdf_results

    def visualize_cell(self, cell_poly, buildings, cell_id, pai, fai):
        """Generates visual plot for a single cell."""
        plot_dir = os.path.join(self.output_dir, "plots")
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)
        
        try:
            fig, ax = plt.subplots(figsize=(6, 6))
            gpd.GeoSeries([cell_poly]).plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2)
            if not buildings.empty:
                buildings.plot(ax=ax, facecolor="black", alpha=0.7)
            ax.set_title(f"Grid {cell_id}\nPAI: {pai:.2f} | FAI: {fai:.2f}")
            ax.axis("off")
            plt.savefig(os.path.join(plot_dir, f"grid_{cell_id}.png"), dpi=80)
            plt.close(fig)
        except:
            plt.close(fig)

    def visualize_city_grid(self, processed_gdf):
        """Generates overview map."""
        plot_dir = os.path.join(self.output_dir, "plots")
        if not os.path.exists(plot_dir): os.makedirs(plot_dir)
        
        if processed_gdf is None or processed_gdf.empty:
            return

        try:
            fig, ax = plt.subplots(figsize=(10, 10))
            
            if self.city_boundary is not None:
                self.city_boundary.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2)
            
            col_to_plot = 'PAI_BuiltPlan'
            
            if col_to_plot in processed_gdf.columns:
                if processed_gdf[col_to_plot].nunique() <= 1:
                     processed_gdf.plot(column=col_to_plot, ax=ax, color='lightgray', legend=False)
                else:
                    processed_gdf.plot(column=col_to_plot, ax=ax, cmap='OrRd', alpha=0.6, legend=True)
                
                count_high_density = 0
                for _, row in processed_gdf.iterrows():
                    if row.geometry and not row.geometry.is_empty:
                        pai = row.get('PAI_BuiltPlan', 0)
                        h_mean = row.get('height_mean', 0)
                        
                        if pai > 0.25 and h_mean > 7:
                            c = row.geometry.centroid
                            ax.text(c.x, c.y, str(row['grid_id']), fontsize=5, ha='center', color='blue', weight='bold')
                            count_high_density += 1
                print(f"   Highlighted {count_high_density} grid cells in city map.")

            plt.title(f"Urban Morphology: {self.city_name} (PAI Heatmap)")
            plt.axis('off')
            plt.savefig(os.path.join(plot_dir, "city_overview_optimized.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"   Warning: Failed to generate city overview plot. Error: {e}")

if __name__ == "__main__":
    cities = [
        "Amsterdam, Netherlands",
        "Barcelona, Spain",
        "Basel, Switzerland",
        "Berlin, Germany",
        "Bilbao, Spain",
        "Brussel, Belgium",
        "Lisboa, Portugal",
        "London, UK",
        "Madrid, Spain",
        "Paris, France",
        "Porto, Portugal",
        "Praha, Czechia",
        "Roma, Italy",
        "Sevilla, Spain",
        "Valencia, Spain",
        "Wien, Austria",
        "Zaragoza, Spain",
        "Mexico City, Mexico",
        "Monterrey, Mexico",
        "Buenos Aires, Argentina",
        "Santiago, Chile",
        "Caracas, Venezuela"
        "Lausanne, Switzerland",
        "Sarajevo, Bosnia and Herzegovina",
        "Gävle, Sweden",
        "Barakaldo, Spain"
    ]
    
    HEIGHTS_ROOT = "/home/fabianh/vscode/post_SOD2D/nn/dataset_geometries/cities/heights/"

    for CITY in cities:
        print(f"\n=== Analyzing {CITY} ===")
        
        current_raster_path = None
        city_key = CITY.split(',')[0].strip()
        
        # Replaced glob.glob with os.listdir logic to avoid namespace collision
        if os.path.exists(HEIGHTS_ROOT):
            try:
                subdirs = [d for d in os.listdir(HEIGHTS_ROOT) if os.path.isdir(os.path.join(HEIGHTS_ROOT, d))]
                matched_dir = next((d for d in subdirs if city_key.lower() in d.lower()), None)
                
                if matched_dir:
                    dir_path = os.path.join(HEIGHTS_ROOT, matched_dir)
                    if os.path.exists(os.path.join(dir_path, "Data")):
                        dir_path = os.path.join(dir_path, "Data")
                    elif os.path.exists(os.path.join(dir_path, "Dataset")):
                        dir_path = os.path.join(dir_path, "Dataset")
                        
                    tif_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.tif')]
                    
                    if tif_files:
                        current_raster_path = tif_files[0]
                        print(f"   -> Found raster: {os.path.basename(current_raster_path)}")
                    else:
                        print(f"   -> Warning: No .tif file found in {matched_dir}")
                else:
                    print(f"   -> Warning: No directory found containing '{city_key}' in {HEIGHTS_ROOT}")
            except Exception as e:
                print(f"   -> Error searching for raster: {e}")
        else:
            print(f"   -> Warning: Heights root directory does not exist: {HEIGHTS_ROOT}")

        analyzer = UrbanMorphologyAnalyzer(CITY, None, height_raster_path=current_raster_path)
        results = analyzer.run_analysis_optimized(batch_size=500)

        del analyzer
        del results
        gc.collect()