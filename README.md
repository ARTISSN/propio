Run "python facemeshmarker.py --input_dir morephotosofdaniel --output_dir maps"

facemeshmarker.py generates landmarks and passes them to drawing_utils.py
drawing_utils.py calculates and saves the normal map data
lighting_utils.py was my attempt at calculating lighting coefficients
coordinate_utils.py is the helper file with all coordinate transformations
normal_map_generator.py is how the normal map is generated from the input image
