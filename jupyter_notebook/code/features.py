import tifffile
import numpy as np
from skimage import measure
from pathlib import Path
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from hdaf_filter import hdaf

def get_features(folder_path, channel, radius, noise_size = 8):
    # Initialize the data dictionary to store features
    data = {
        'list_files': [f.name for f in Path(folder_path).iterdir() if f.suffix in {'.tif'}],
        'num objects': [],
        'intensity_mean_objects': [],
        'img_mean': [],
        'mean_tamano_objetos': [],
        'distance': [],
        'img': [],
        'segmentation_labeled': []
    }

    # Process each image file
    for file_name in data['list_files']:
        print(f"Running Channel {channel} --- file = {file_name}")
        # Read image
        img = tifffile.imread(Path(folder_path, file_name), key = channel) 

        # Store image
        data['img'].append(img)

        # Segment bright structures
        segmentation_labeled = detectar_estructuras_brillantes(img, radius, noise_size)
        data['segmentation_labeled'].append(segmentation_labeled)

        # Feature 1: Number of detected objects
        n_objects = np.max(segmentation_labeled)
        data['num objects'].append(n_objects)
            
        # Feature 2: calcular la intensidad media de todos los objetos
        # Feature 4: promedio de tamaño de objetos (en pixeles)
        mean_intensity_object = np.zeros(n_objects)
        tamano_object = np.zeros(n_objects)
        
        for ii in range(img.shape[0]):
            for jj in range(img.shape[1]):
                label = data['segmentation_labeled'][-1][ii, jj]
                if label > 0:
                    mean_intensity_object[label - 1] += img[ii, jj]  # Adjust for 0-indexing
                    tamano_object[label - 1] += 1  # Adjust for 0-indexing            

        # Calculate the mean intensity for each object
        mean_intensity_object = np.divide(mean_intensity_object, tamano_object, where=tamano_object != 0)
        
        data['intensity_mean_objects'].append(np.mean(mean_intensity_object))
        data['mean_tamano_objetos'].append(np.mean(tamano_object))

        # Feature 3: Mean intensity of the image
        data['img_mean'].append(np.mean(img))

        # Feature 5: Distance to the center of the segmentation
        distance = distance_transform_edt(segmentation_labeled > 0)
        
        distance = distance[distance > 0]
        q = np.quantile(distance, 0.50)
        data['distance'].append(np.mean(distance[distance > q]))
        
        # distance = np.sort(distance[distance > 0])[::-1]
        # data['distance'].append(np.mean(distance[:2000]))

    return data

def detectar_estructuras_brillantes(img, radius, noise_size):
    # Normalize the image
    img = img.astype(np.float32) / np.max(img)

    # create an object to calculate multiscale filter
    img_filter = hdaf.filt(img)
    
    # Compute the multiscale Laplacian of the image
    lap = img_filter.apply_filter("laplacian_multiscale", radius)

    # Segment bright structures based on a threshold
    I_background = lap > 0
    threshold = np.mean(img[I_background]) + 2 * np.std(img[I_background])
    seg = img > threshold

    # Label connected components
    label = measure.label(seg, connectivity=2)

    # Initialize the label_removed array
    label_removed = np.zeros_like(label)
    
    # Filter out small objects
    count = 1
    for region in measure.regionprops(label):
        if region.area > noise_size:
            label_removed[region.coords[:, 0], region.coords[:, 1]] = count
            count += 1
    
    return label_removed


# Example usage:
# folder_path = r'D:\dataset\CMNSXXI_Psoriasis\2024_06_Pacientes_Sanos_Claudina_1_y_4\Sanos_test'
# channel = 0  # Example channel
# radius = [4,7,10]   # Example radius
# data = get_features(folder_path, channel, radius)
